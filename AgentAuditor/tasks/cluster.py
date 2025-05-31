import json
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from finch import FINCH
import logging
import traceback
import os
import pickle

# --- Hyperparameters ---
NOMIC_MODEL = 'nomic-ai/nomic-embed-text-v1.5'
MATRYOSHKA_DIM = 512

# --- PCA Configuration ---
USE_PCA = True # Set to True to enable PCA (only if multiple weights > 0)

# --- Constants for Text Processing ---
PLACEHOLDER_TEXT = "" # Text used if original field is missing/empty/error

# Define individual error constants FIRST
ERROR_VALUE = "LLM_GENERATION_ERROR"
NO_CONTENT_VALUE = "NO_CONTENT_PROVIDED"
PARSING_ERROR_PREFIX = "PARSING_ERROR" # Prefix for parsing errors
NO_VALID_JSON_VALUE = "NO_VALID_JSON_FOUND"
PROCESSING_ERROR_PREFIX = "PROCESSING_ERROR" # Prefix for processing errors
MISSING_KEY_PREFIX = "MISSING_KEY" # Prefix for missing keys in parsed JSON

# Define the set of full error strings for direct matching (optional, but used in code)
LLM_ERROR_INDICATORS = {
    ERROR_VALUE,
    NO_CONTENT_VALUE,
    NO_VALID_JSON_VALUE
    # You can add specific PARSING/PROCESSING/MISSING key errors here if needed for exact matching
    # e.g., "PARSING_ERROR_JSONDecodeError" if you want to match it exactly in the set
}

# Define expected keys from LLM output (used in previous script, ensure consistency)
EXPECTED_KEYS = ["application_scenario", "risk_type", "failure_mode"]

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeightedDialogueClusterer:

    def __init__(self, model_name: str = NOMIC_MODEL, matryoshka_dim: int = MATRYOSHKA_DIM):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        try:
            self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)
            logger.info(f"SentenceTransformer model '{model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{model_name}': {e}", exc_info=True)
            raise

        self.matryoshka_dim = matryoshka_dim if matryoshka_dim else self.model.get_sentence_embedding_dimension()
        if matryoshka_dim:
             logger.info(f"Using Matryoshka dimension: {self.matryoshka_dim}")

    def _extract_text_field(self, item: Dict, field_name: str) -> str:
        field_value = item.get(field_name)
        if field_value is None:
            logger.debug(f"Field '{field_name}' missing in item ID {item.get('id', 'N/A')}. Using placeholder.")
            return PLACEHOLDER_TEXT
        if isinstance(field_value, str):
            if field_value in LLM_ERROR_INDICATORS or field_value.startswith(PARSING_ERROR_PREFIX) or \
               field_value.startswith(PROCESSING_ERROR_PREFIX) or field_value.startswith(MISSING_KEY_PREFIX):
                 logger.debug(f"Field '{field_name}' contains error indicator '{field_value}' in item ID {item.get('id', 'N/A')}. Using placeholder.")
                 return PLACEHOLDER_TEXT
            return field_value.strip()
        else:
            # Handle non-string fields if necessary, e.g., contents list
            if field_name == 'contents':
                return self._extract_dialogue_text_from_contents(field_value)
            else:
                 logger.warning(f"Field '{field_name}' is not a string (type: {type(field_value)}) in item ID {item.get('id', 'N/A')}. Converting to string.")
                 return str(field_value).strip()


    def _extract_dialogue_text_from_contents(self, contents: Union[List[Dict], Any]) -> str:
        text_parts = []
        if not contents:
            return ""

        if not isinstance(contents, list):
            logger.warning(f"Expected 'contents' to be a list, but got {type(contents)}. Attempting string conversion.")
            return str(contents)

        try:
            if contents and isinstance(contents[0], dict):
                for turn in contents:
                    if turn and isinstance(turn, dict):
                        content = str(turn.get('content', '') or '')
                        thought = str(turn.get('thought', '') or '')
                        action = str(turn.get('action', '') or '')
                        turn_text = " ".join(filter(None, [content, thought, action])).strip()
                        if turn_text:
                            text_parts.append(turn_text)
            elif contents and isinstance(contents[0], list):
                 logger.warning("Detected nested list structure in 'contents'. Processing inner lists.")
                 for dialogue in contents:
                       if dialogue and isinstance(dialogue, list):
                           for turn in dialogue:
                               if turn and isinstance(turn, dict):
                                   content = str(turn.get('content', '') or '')
                                   thought = str(turn.get('thought', '') or '')
                                   action = str(turn.get('action', '') or '')
                                   turn_text = " ".join(filter(None, [content, thought, action])).strip()
                                   if turn_text:
                                       text_parts.append(turn_text)
            else:
                 # Handle cases where contents is a list but not of dicts or lists
                 logger.warning(f"Unexpected structure within 'contents' list (first item type: {type(contents[0])}). Joining items as strings.")
                 text_parts = [str(item) for item in contents]

        except IndexError:
            logger.warning(f"IndexError while processing contents snippet: {str(contents)[:200]}...")
            return ""
        except Exception as e:
            logger.error(f"Error extracting text from contents: {e}. Contents snippet: {str(contents)[:200]}...", exc_info=True)
            return ""

        full_text = " ".join(text_parts).strip()
        full_text = ' '.join(full_text.split())
        return full_text


    def get_embeddings(self, texts: List[str], description: str) -> np.ndarray:
        logger.info(f"Generating embeddings for {len(texts)} items ({description})...")
        if not texts:
            logger.warning(f"Received empty list of texts for embedding generation ({description}).")
            target_dim = self.matryoshka_dim if self.matryoshka_dim else self.model.get_sentence_embedding_dimension()
            return np.empty((0, target_dim), dtype=np.float32)

        embeddings_list = []
        batch_size = 16 if self.device.type == 'cuda' else 8
        logger.info(f"Using batch size: {batch_size}")

        total_batches = (len(texts) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                processed_batch = [text if text and text.strip() else PLACEHOLDER_TEXT for text in batch_texts]

                try:
                    batch_embeddings = self.model.encode(processed_batch, convert_to_tensor=True, device=self.device, batch_size=len(processed_batch))

                    if self.matryoshka_dim and self.matryoshka_dim < batch_embeddings.shape[1]:
                         batch_embeddings = batch_embeddings[:, :self.matryoshka_dim]

                    # Perform normalization here - crucial for weighting and cosine similarity
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

                    embeddings_list.append(batch_embeddings.cpu().numpy())

                except Exception as e:
                    logger.error(f"Error encoding batch for {description} starting at index {i}: {e}", exc_info=True)
                    error_shape = (len(processed_batch), self.matryoshka_dim if self.matryoshka_dim else self.model.get_sentence_embedding_dimension())
                    logger.warning(f"Appending zero vectors for failed batch {i // batch_size + 1}/{total_batches}")
                    embeddings_list.append(np.zeros(error_shape, dtype=np.float32))

                if (i // batch_size + 1) % (max(1, total_batches // 10)) == 0 or (i // batch_size + 1) == total_batches:
                     logger.info(f"Processed batch {i // batch_size + 1}/{total_batches} for {description}")

        if not embeddings_list:
             logger.error(f"Embedding generation for {description} resulted in an empty list.")
             target_dim = self.matryoshka_dim if self.matryoshka_dim else self.model.get_sentence_embedding_dimension()
             return np.empty((0, target_dim), dtype=np.float32)

        all_embeddings = np.vstack(embeddings_list)

        if not np.all(np.isfinite(all_embeddings)):
            logger.warning(f"Non-finite values (NaN or Inf) found in {description} embeddings. Replacing with zeros.")
            all_embeddings = np.nan_to_num(all_embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"{description} embedding generation complete, final shape: {all_embeddings.shape}")
        return all_embeddings.astype(np.float32)


    def _calculate_centers_and_distances(self, embeddings: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if embeddings.size == 0 or labels.size == 0:
            logger.warning("Cannot calculate centers/distances with empty embeddings or labels.")
            return np.array([]), np.array([])

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]
        n_clusters_found = len(unique_labels)
        if n_clusters_found == 0:
             logger.warning("No non-negative cluster labels found. Cannot calculate centers.")
             return np.array([]), np.array([])

        embedding_dim = embeddings.shape[1]
        logger.info(f"Calculating centers for {n_clusters_found} clusters using embeddings of shape {embeddings.shape}...")

        centers = np.zeros((n_clusters_found, embedding_dim), dtype=embeddings.dtype)
        cluster_map = {label: i for i, label in enumerate(unique_labels)}

        for original_label, cluster_idx in cluster_map.items():
            cluster_mask = (labels == original_label)
            cluster_points = embeddings[cluster_mask]

            if len(cluster_points) > 0:
                center = np.mean(cluster_points, axis=0)
                # Normalize the center (centroid) as well
                norm = np.linalg.norm(center)
                if norm > 1e-9:
                    centers[cluster_idx] = center / norm
                else:
                    # Handle potential zero vectors
                    centers[cluster_idx] = center
                    logger.warning(f"Cluster center for label {original_label} has zero norm.")
            else:
                logger.warning(f"Cluster with original label {original_label} has no points assigned!")
                # Center remains zero vector

        logger.info("Calculating distances to centers...")
        # Initialize distances to max possible (2.0 for 1-cosine with normalized vectors)
        distances = np.full((embeddings.shape[0], n_clusters_found), 2.0, dtype=embeddings.dtype)
        if centers.shape[0] > 0:
            try:
                # Embeddings are already normalized, centers are normalized
                similarity_matrix = cosine_similarity(embeddings, centers)
                # Clip ensures values are within [0, 2] after 1 - similarity
                distances = np.clip(1.0 - similarity_matrix, 0.0, 2.0)
            except Exception as e:
                logger.error(f"Error calculating cosine similarity for distances: {e}", exc_info=True)

        return centers, distances


    def perform_clustering(self, embeddings: np.ndarray, target_n_clusters: Optional[int] = None) -> Dict:
        logger.info("Performing FINCH clustering...")

        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            logger.error(f"Invalid embeddings shape for clustering: {embeddings.shape}. Need (N, D) with N > 0.")
            return {'cluster_labels': np.array([], dtype=int), 'cluster_centers': np.array([]), 'distances': np.array([]), 'num_clusters_found': 0}

        if embeddings.dtype != np.float32:
            logger.warning(f"Embeddings dtype is {embeddings.dtype}, converting to float32 for FINCH.")
            embeddings = embeddings.astype(np.float32)

        # FINCH expects float32
        if not np.all(np.isfinite(embeddings)):
             logger.error("Non-finite values detected in embeddings immediately before FINCH call. Cannot proceed.")
             return {'cluster_labels': np.array([], dtype=int), 'cluster_centers': np.array([]), 'distances': np.array([]), 'num_clusters_found': 0}

        logger.info(f"Input embeddings shape to FINCH: {embeddings.shape}")

        try:
            # Use distance='cosine' as embeddings should be normalized
            c, num_clust, _ = FINCH(embeddings, distance='cosine', ensure_early_exit=True, verbose=False)
            if isinstance(c, np.ndarray):
                 logger.info(f"FINCH returned c shape: {c.shape}, num_clust list: {num_clust}")
            else:
                 logger.warning(f"FINCH output 'c' is not a numpy array. Type: {type(c)}. Cannot determine shape.")
                 return {'cluster_labels': np.array([], dtype=int), 'cluster_centers': np.array([]), 'distances': np.array([]), 'num_clusters_found': 0}
        except Exception as e:
            logger.error(f"FINCH clustering failed with an exception: {e}", exc_info=True)
            return {'cluster_labels': np.array([], dtype=int), 'cluster_centers': np.array([]), 'distances': np.array([]), 'num_clusters_found': 0}

        if not num_clust or c.ndim != 2 or c.shape[0] != embeddings.shape[0] or c.shape[1] != len(num_clust):
             logger.error(f"FINCH output validation failed. Expected c shape ({embeddings.shape[0]}, {len(num_clust) if num_clust else 'N/A'}), got {c.shape}. Or num_clust is empty: {num_clust}")
             return {'cluster_labels': np.array([], dtype=int), 'cluster_centers': np.array([]), 'distances': np.array([]), 'num_clusters_found': 0}

        best_partition_idx = -1
        if target_n_clusters is not None and target_n_clusters > 0:
            logger.info(f"Targeting approximately {target_n_clusters} clusters.")
            min_diff = float('inf')

            valid_partitions = [(i, n) for i, n in enumerate(num_clust) if n > 0]
            if not valid_partitions:
                 logger.warning("FINCH found no partitions with clusters. Using the last partition by default.")
                 best_partition_idx = c.shape[1] - 1
            else:
                # Find partition closest to target, preferring slightly more clusters in case of tie
                # And prefer later partition if difference and direction preference is same
                best_n_clusters = -1
                for i, n_clusters in valid_partitions:
                    diff = abs(n_clusters - target_n_clusters)
                    if diff < min_diff:
                        min_diff = diff
                        best_partition_idx = i
                        best_n_clusters = n_clusters
                    elif diff == min_diff:
                         # Prefer partitions with >= target clusters over those < target
                        if n_clusters >= target_n_clusters and best_n_clusters < target_n_clusters:
                            best_partition_idx = i
                            best_n_clusters = n_clusters
                        # If both are >= target or both are < target, prefer the one with more clusters (usually later partition)
                        elif (n_clusters >= target_n_clusters and best_n_clusters >= target_n_clusters) or \
                             (n_clusters < target_n_clusters and best_n_clusters < target_n_clusters):
                             if n_clusters > best_n_clusters: # Prefer more clusters if diff is same
                                best_partition_idx = i
                                best_n_clusters = n_clusters
                             elif n_clusters == best_n_clusters and i > best_partition_idx: # Prefer later partition if clusters also same
                                best_partition_idx = i
                                best_n_clusters = n_clusters


                if best_partition_idx == -1: # Should not happen if valid_partitions is not empty
                    logger.warning("Could not determine best partition index. Defaulting to last partition.")
                    best_partition_idx = c.shape[1] - 1
                else:
                    # Final check on index bounds
                    if best_partition_idx >= len(num_clust):
                        logger.error(f"Calculated best_partition_idx {best_partition_idx} is out of bounds for num_clust (len {len(num_clust)}). Defaulting to last.")
                        best_partition_idx = c.shape[1] - 1
                    logger.info(f"Selected partition index {best_partition_idx} with {num_clust[best_partition_idx]} clusters (target was {target_n_clusters}, difference {min_diff}).")
        else:
            logger.info("No target number of clusters specified or target <= 0. Using the last partition found by FINCH.")
            best_partition_idx = c.shape[1] - 1

        # Ensure index is valid before accessing 'c'
        if best_partition_idx < 0 or best_partition_idx >= c.shape[1]:
             logger.error(f"Final best_partition_idx {best_partition_idx} is out of bounds for c matrix columns ({c.shape[1]}). Cannot extract labels.")
             return {'cluster_labels': np.array([], dtype=int), 'cluster_centers': np.array([]), 'distances': np.array([]), 'num_clusters_found': 0}

        try:
            final_labels = c[:, best_partition_idx]
            logger.info(f"Accessed labels using c[:, {best_partition_idx}]")
        except IndexError as ie:
            logger.error(f"IndexError accessing FINCH labels c[:, {best_partition_idx}]. c shape: {c.shape}. Error: {ie}", exc_info=True)
            return {'cluster_labels': np.array([], dtype=int), 'cluster_centers': np.array([]), 'distances': np.array([]), 'num_clusters_found': 0}
        except Exception as e:
            logger.error(f"Unexpected error accessing FINCH labels c[:, {best_partition_idx}]. Error: {e}", exc_info=True)
            return {'cluster_labels': np.array([], dtype=int), 'cluster_centers': np.array([]), 'distances': np.array([]), 'num_clusters_found': 0}

        # Use the count from num_clust corresponding to the chosen partition
        if best_partition_idx < len(num_clust):
            num_clusters_found = num_clust[best_partition_idx]
        else:
             # Fallback if index somehow became invalid for num_clust after selection
             logger.warning(f"best_partition_idx {best_partition_idx} is out of bounds for num_clust list (len {len(num_clust)}). Calculating clusters from labels.")
             num_clusters_found = len(np.unique(final_labels[final_labels >= 0]))


        if final_labels.ndim != 1 or final_labels.shape[0] != embeddings.shape[0]:
            logger.error(f"CRITICAL SHAPE MISMATCH! Embeddings shape: {embeddings.shape}, final_labels shape: {final_labels.shape}.")
            return {'cluster_labels': np.array([], dtype=int), 'cluster_centers': np.array([]), 'distances': np.array([]), 'num_clusters_found': 0}

        logger.info(f"Selected partition results in {num_clusters_found} clusters. Label array shape: {final_labels.shape}")

        logger.info("Calculating cluster centers and distances post-FINCH...")
        # Pass the *same* embeddings used for clustering to calculate centers/distances
        cluster_centers, distances = self._calculate_centers_and_distances(embeddings, final_labels)
        logger.info(f"Center calculation complete (shape: {cluster_centers.shape}). Distance calculation complete (shape: {distances.shape}).")

        effective_clusters = len(np.unique(final_labels[final_labels >= 0]))
        if effective_clusters != num_clusters_found:
            logger.warning(f"Number of clusters reported ({num_clusters_found}) differs from unique non-negative labels found ({effective_clusters}). Trusting unique labels count.")
            num_clusters_found = effective_clusters

        # Ensure distance matrix columns matches the final cluster count
        if distances.shape[1] != num_clusters_found:
             logger.warning(f"Mismatch between distance matrix columns ({distances.shape[1]}) and final effective cluster count ({num_clusters_found}). Using distance matrix columns count.")
             num_clusters_found = distances.shape[1] # Trust the shape of the calculated distances

        return {
            'cluster_labels': final_labels,
            'cluster_centers': cluster_centers,
            'distances': distances,
            'num_clusters_found': num_clusters_found
        }


    def find_representatives(self, json_data: List[Dict], clustering_results: Dict, original_indices: List[int]) -> List[Dict]:
        logger.info("Finding representative dialogues for each cluster...")
        cluster_labels = clustering_results.get('cluster_labels')
        distances = clustering_results.get('distances')
        num_clusters_found = clustering_results.get('num_clusters_found', 0)

        if cluster_labels is None or distances is None or num_clusters_found <= 0 or cluster_labels.size == 0 or distances.size == 0:
             logger.warning(f"Cannot find representatives: Missing/empty clustering results (num_clusters={num_clusters_found}).")
             return []

        if distances.shape[1] != num_clusters_found:
             logger.warning(f"Adjusting num_clusters_found from {num_clusters_found} to {distances.shape[1]} based on distance matrix.")
             num_clusters_found = distances.shape[1]
             if num_clusters_found <= 0: return []

        if len(json_data) != len(original_indices):
             logger.error(f"Length mismatch: json_data ({len(json_data)}) vs original_indices ({len(original_indices)}).")
             return []
        if len(cluster_labels) != len(json_data):
             logger.error(f"Length mismatch: cluster_labels ({len(cluster_labels)}) vs json_data ({len(json_data)}).")
             return []
        if distances.shape[0] != len(json_data):
            logger.error(f"Length mismatch: distances rows ({distances.shape[0]}) vs json_data ({len(json_data)}).")
            return []

        representatives = []
        processed_original_indices = set()
        unique_original_labels = np.unique(cluster_labels[cluster_labels >= 0])

        logger.info(f"Iterating through {num_clusters_found} clusters (based on distance matrix columns) to find representatives...")

        # We iterate based on the columns of the distance matrix (0 to K-1)
        # which should correspond to the calculated centers.
        for cluster_idx in range(num_clusters_found):
            # Find the original label value this center corresponds to.
            # This relies on the order preserved in _calculate_centers_and_distances
            if cluster_idx >= len(unique_original_labels):
                logger.warning(f"Skipping cluster index {cluster_idx}, as it exceeds the number of unique non-negative labels found ({len(unique_original_labels)}). Possible empty cluster.")
                continue
            original_label = unique_original_labels[cluster_idx]

            # Find indices (in the filtered data) belonging to this cluster label
            indices_in_cluster_mask = (cluster_labels == original_label)
            indices_in_cluster_relative = np.where(indices_in_cluster_mask)[0]

            if len(indices_in_cluster_relative) == 0:
                logger.warning(f"Cluster with original label {original_label} (center index {cluster_idx}) is empty. Skipping.")
                continue

            # Get distances *for points in this cluster* to *this cluster's center*
            cluster_distances_for_points = distances[indices_in_cluster_relative, cluster_idx]

            # Find the point within this cluster that has the minimum distance to the center
            min_dist_idx_within_cluster = np.argmin(cluster_distances_for_points)
            # Get its index relative to the filtered data (which matches labels/distances rows)
            representative_relative_idx = indices_in_cluster_relative[min_dist_idx_within_cluster]

            # Map back to the original full dataset index using the provided mapping
            if representative_relative_idx >= len(original_indices):
                 logger.error(f"Representative relative index {representative_relative_idx} out of bounds for original_indices map (len {len(original_indices)}). Skipping.")
                 continue
            representative_original_idx = original_indices[representative_relative_idx]

            # Check if this original index has already been selected (should be rare with FINCH?)
            if representative_original_idx in processed_original_indices:
                logger.warning(f"Representative original index {representative_original_idx} already chosen. Finding next best for cluster {cluster_idx} (label {original_label}).")
                sorted_dist_indices_within_cluster = np.argsort(cluster_distances_for_points)
                found_alternative = False
                for i in range(1, len(sorted_dist_indices_within_cluster)):
                    alt_relative_idx = indices_in_cluster_relative[sorted_dist_indices_within_cluster[i]]
                    if alt_relative_idx < len(original_indices):
                        alt_original_idx = original_indices[alt_relative_idx]
                        if alt_original_idx not in processed_original_indices:
                             # Found an alternative, use it
                            representative_original_idx = alt_original_idx
                            logger.info(f"  > Found alternative representative: original index {representative_original_idx}")
                            found_alternative = True
                            break # Stop searching for alternatives
                    else:
                         logger.error(f"Invalid alternative relative index {alt_relative_idx} found.")
                         break # Stop if index mapping fails

                if not found_alternative:
                    logger.error(f"  > Could not find unique alternative representative for cluster {cluster_idx}. Skipping.")
                    continue # Skip this cluster if no unique representative found

            # Add the chosen representative (original or alternative)
            # Ensure the index is valid for the *original* full json_data list
            # Note: We pass the filtered_json_data to this function, but need original index for tracking
            # Let's pass the FULL json data to main and use the original index here.
            # Correction needed in main() call.
            # For now, assume json_data IS the filtered list corresponding to original_indices
            if representative_relative_idx < len(json_data): # Index must be valid for the json_data list passed in
                 representative_data = json_data[representative_relative_idx].copy() # Get data using relative index
                 # Add cluster info maybe?
                #  representative_data['cluster_assigned_label'] = int(original_label)
                #  representative_data['cluster_center_index'] = int(cluster_idx)
                 representatives.append(representative_data)
                 processed_original_indices.add(representative_original_idx) # Track original index
                 # logger.info(f"Cluster {cluster_idx} (label {original_label}): Found representative - original index {representative_original_idx} ({len(indices_in_cluster_relative)} points in cluster)")
            else:
                 logger.error(f"Representative relative index {representative_relative_idx} out of bounds for filtered json_data (len {len(json_data)}). Skipping.")
                 continue # Skip if index is invalid

        logger.info(f"Found {len(representatives)} representative dialogues out of {num_clusters_found} identified clusters.")
        return representatives


def load_and_filter_data(input_path: str, clusterer: WeightedDialogueClusterer) -> Tuple[List[Dict], List[int], List[str], List[str], List[str], List[str]]:
    logger.info(f"Loading data from {input_path}...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            full_json_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file: {input_path} - {e}")
        raise
    except Exception as e:
       logger.error(f"An unexpected error occurred while loading data: {e}", exc_info=True)
       raise

    logger.info(f"Loaded {len(full_json_data)} entries initially.")

    if not isinstance(full_json_data, list) or not full_json_data:
       logger.error("Input data is not a list or is empty.")
       raise ValueError("Input data must be a non-empty list.")

    texts_c, texts_s, texts_r, texts_f = [], [], [], []
    valid_indices = []
    filtered_json_data = []

    for i, item in enumerate(full_json_data):
        if not isinstance(item, dict):
            logger.warning(f"Item at index {i} is not a dictionary, skipping.")
            continue

        # Extract texts, using placeholder for errors/missing
        text_c = clusterer._extract_text_field(item, 'contents')
        text_s = clusterer._extract_text_field(item, 'application_scenario')
        text_r = clusterer._extract_text_field(item, 'risk_type')
        text_f = clusterer._extract_text_field(item, 'failure_mode')

        # We cluster based on the combination, so include even if some parts are empty/placeholders
        # But maybe filter if CONTENTS is empty? Let's assume we cluster all valid entries.
        texts_c.append(text_c)
        texts_s.append(text_s)
        texts_r.append(text_r)
        texts_f.append(text_f)
        valid_indices.append(i) # Original index
        filtered_json_data.append(item) # Keep the original data for this valid entry

    logger.info(f"Kept {len(filtered_json_data)} entries for processing.")
    if len(filtered_json_data) == 0:
         logger.error("No valid entries found after initial check. Cannot proceed.")
         raise ValueError("No valid entries to process.")

    return filtered_json_data, valid_indices, texts_c, texts_s, texts_r, texts_f


def cluster_main(dataset):
    logger.info("--- Starting Weighted Dialogue Clustering ---")
    global USE_PCA

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, f"../temp/{dataset}/memory.json")
    output_path = os.path.join(script_dir, f"../temp/{dataset}/cluster.json")

    logger.info(f"Input file: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Use PCA: {USE_PCA}")


    script_dir = os.path.dirname(os.path.abspath(__file__))
    # --- Load Parameters ---
    with open(os.path.join(script_dir, '../params/clus_param.pkl'), 'rb') as f:
        params = pickle.load(f)

    try:
        clusterer = WeightedDialogueClusterer()
        filtered_data, original_indices, texts_c, texts_s, texts_r, texts_f = load_and_filter_data(input_path, clusterer)

        TARGET_N_CLUSTERS = int(len(filtered_data) / 10)

        logger.info("--- Generating Embeddings ---")
        Ec = clusterer.get_embeddings(texts_c, "Contents (Ec)")
        Es = clusterer.get_embeddings(texts_s, "Scenario (Es)")
        Er = clusterer.get_embeddings(texts_r, "Risk Type (Er)")
        Ef = clusterer.get_embeddings(texts_f, "Failure Mode (Ef)")

        # Basic shape validation
        expected_len = len(filtered_data)
        if not (Ec.shape[0] == expected_len and Es.shape[0] == expected_len and Er.shape[0] == expected_len and Ef.shape[0] == expected_len):
             logger.error("Mismatch in embedding array lengths after generation. Cannot proceed.")
             return
        if not (Ec.shape[1] == Es.shape[1] == Er.shape[1] == Ef.shape[1]):
             logger.error("Mismatch in embedding dimensions between different fields. Cannot proceed.")
             return

        logger.info("--- Combining Embeddings ---")
        # Normalization happened within get_embeddings

        active_embeddings = []
        active_weights_count = 0
        if params[0] > 1e-6 : # Use small threshold instead of == 0 for float comparison
             active_embeddings.append(params[0] * Ec)
             active_weights_count += 1
        if params[1] > 1e-6 :
             active_embeddings.append(params[1] * Es)
             active_weights_count += 1
        if params[2] > 1e-6 :
             active_embeddings.append(params[2] * Er)
             active_weights_count += 1
        if params[3] > 1e-6 :
             active_embeddings.append(params[3] * Ef)
             active_weights_count += 1

        if not active_embeddings:
            logger.error("All weights are effectively zero. Cannot perform clustering.")
            return

        # --- Special Case Handling & PCA ---
        apply_pca_now = USE_PCA and active_weights_count > 1
        skip_pca_info = ""

        if active_weights_count == 1:
            final_embeddings_for_clustering = active_embeddings[0]
            if USE_PCA:
                 skip_pca_info = "(PCA skipped as only one embedding type is active)"
            logger.info(f"Using single active embedding type. Final shape: {final_embeddings_for_clustering.shape} {skip_pca_info}")
        else:
             # Concatenate only active weighted embeddings
            combined_embeddings_highd = np.concatenate(active_embeddings, axis=1)
            logger.info(f"Concatenated active embeddings shape: {combined_embeddings_highd.shape}")

            if apply_pca_now:
                logger.info("Applying PCA for dimensionality reduction...")
                n_components = params[5]
                if n_components is None:
                     n_components = params[4]
                     if not (0 < n_components <= 1.0):
                          logger.warning(f"out of range (0, 1]. Using default PCA behavior.")
                          n_components = None # Let PCA decide based on variance

                try:
                    pca = PCA(n_components=n_components)
                    final_embeddings_for_clustering = pca.fit_transform(combined_embeddings_highd)
                    logger.info(f"PCA applied. Reduced embedding shape: {final_embeddings_for_clustering.shape}")
                    logger.info(f"PCA - Explained variance ratio sum: {np.sum(pca.explained_variance_ratio_):.4f}")
                    logger.info(f"PCA - Number of components chosen: {pca.n_components_}")
                except Exception as e:
                     logger.error(f"PCA failed: {e}. Falling back to using non-reduced combined embeddings.", exc_info=True)
                     final_embeddings_for_clustering = combined_embeddings_highd
                     apply_pca_now = False # Mark PCA as not applied

            else:
                 final_embeddings_for_clustering = combined_embeddings_highd
                 logger.info(f"Using concatenated embeddings without PCA. Shape: {final_embeddings_for_clustering.shape}")

        # Ensure final embeddings are contiguous float32 for FINCH
        final_embeddings_for_clustering = np.ascontiguousarray(final_embeddings_for_clustering, dtype=np.float32)

        logger.info("--- Starting Clustering Step ---")
        results = clusterer.perform_clustering(final_embeddings_for_clustering, target_n_clusters=TARGET_N_CLUSTERS)
        logger.info("--- Finished Clustering Step ---")

        if not results or results.get('num_clusters_found', 0) <= 0 or results.get('cluster_labels') is None:
            logger.error("Clustering did not produce valid results. Cannot find representatives.")
            return

        num_clusters_found = results['num_clusters_found']

        logger.info("--- Starting Representative Finding Step ---")
        # Pass the filtered_data list that corresponds row-wise to the embeddings used
        representatives = clusterer.find_representatives(filtered_data, results, original_indices)
        logger.info("--- Finished Representative Finding Step ---")

        if not representatives:
             logger.warning("No representative dialogues were found.")
        else:
            logger.info(f"Saving {len(representatives)} representative dialogues to {output_path}...")
            try:
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(representatives, f, indent=4, ensure_ascii=False)
                logger.info(f"Successfully saved representatives to {output_path}")
            except IOError as e:
                logger.error(f"Error writing output file {output_path}: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"An unexpected error occurred during saving: {e}", exc_info=True)

        print("\n--- Clustering Summary ---")
        print(f"Input file read: {input_path}")
        print(f"Entries processed: {len(filtered_data)}")
        print(f"PCA Applied: {apply_pca_now}")
        print(f"Embedding dimensions before FINCH: {final_embeddings_for_clustering.shape[1]}")
        if TARGET_N_CLUSTERS is not None and TARGET_N_CLUSTERS > 0:
            print(f"Target number of clusters: {TARGET_N_CLUSTERS}")
        print(f"FINCH Clustering found: {num_clusters_found} clusters")
        print(f"Number of representative dialogues found: {len(representatives)}")
        if representatives:
            print(f"Representative dialogues saved to: {output_path}")
        else:
            print(f"No representatives saved.")
        print("--------------------------\n")

        max_overview = 10
        if representatives:
            print(f"\nRepresentative dialogues overview (showing up to {max_overview}):")
            for i, rep in enumerate(representatives[:max_overview]):
                print(f"\nCluster Representative {i} (Cluster Label {rep.get('cluster_assigned_label', 'N/A')}):")
                print(f"  Original ID: {rep.get('id', 'N/A')}")
                print(f"  Scenario: {rep.get('application_scenario', 'N/A')}")
                print(f"  Risk Type: {rep.get('risk_type', 'N/A')}")
                print(f"  Failure Mode: {rep.get('failure_mode', 'N/A')}")
                content_snippet = clusterer._extract_text_field(rep, 'contents')
                print(f"  Content Snippet: {content_snippet[:150]}...")
                print("-" * 50)
            if len(representatives) > max_overview:
                print(f"\n... and {len(representatives) - max_overview} more cluster representatives.")


    except Exception as e:
        logger.error(f"An unhandled error occurred in main execution: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred. Please check the log file for details.")
