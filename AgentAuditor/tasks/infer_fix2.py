import json
import time
import requests
import os
from typing import Dict, List, Optional, Any
from tqdm import tqdm

class GPTConfig:
    def __init__(self):
        self.API_KEY = "sk-xxxx"  # Replace with your actual API key
        self.API_BASE = "ENDPOINT"
        self.MODEL = "gpt-4.1-2025-04-14"
        self.TEMPERATURE = 0
        self.TOP_P = 0.7
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 5
        
class LLMHandler:
    def __init__(self, config: GPTConfig):
        self.config = config

    def call_llm_api(self, prompt: str, item_id: int) -> Optional[str]:
        # (Code from previous answer - handles API calls, retries, errors)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.API_KEY}"
        }
        data = {
            "model": self.config.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.TEMPERATURE,
            "top_p": self.config.TOP_P
        }
        for attempt in range(self.config.MAX_RETRIES):
            try:
                print(f"\n[ID: {item_id}] Calling Correction API (Attempt {attempt + 1}/{self.config.MAX_RETRIES})...")
                response = requests.post(
                    f"{self.config.API_BASE}/chat/completions",
                    headers=headers, json=data, timeout=60
                )
                response.raise_for_status()
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0 and \
                   "message" in result["choices"][0] and \
                   "content" in result["choices"][0]["message"]:
                    content = result["choices"][0]["message"]["content"]
                    print(f"[ID: {item_id}] Correction API call successful.")
                    return content
                else:
                    print(f"[ID: {item_id}] Correction API response format incomplete: {result}")
            except requests.exceptions.RequestException as e:
                print(f"[ID: {item_id}] Correction API call failed (Attempt {attempt + 1}): {str(e)}")
                if response is not None and response.status_code in [401, 403, 429]:
                    print(f"[ID: {item_id}] Status code {response.status_code}, stopping retries.")
                    break
                if attempt < self.config.MAX_RETRIES - 1:
                    print(f"Waiting {self.config.RETRY_DELAY} seconds before retry...")
                    time.sleep(self.config.RETRY_DELAY)
                else:
                    print(f"[ID: {item_id}] Max retries reached.")
            except Exception as e:
                 print(f"[ID: {item_id}] Unexpected error during correction API call (Attempt {attempt + 1}): {str(e)}")
                 if attempt < self.config.MAX_RETRIES - 1:
                    print(f"Waiting {self.config.RETRY_DELAY} seconds before retry...")
                    time.sleep(self.config.RETRY_DELAY)
                 else:
                    print(f"[ID: {item_id}] Max retries reached.")
        print(f"[ID: {item_id}] All correction API call attempts failed.")
        return None

# --- [Paste validate_output_format function here] ---
def validate_output_format(output_data: Any) -> bool:
    # (Code from previous answer - validates the structure)
    if not isinstance(output_data, dict): return False
    chain_of_thought = output_data.get("chain_of_thought")
    if not isinstance(chain_of_thought, dict): return False
    final_output = chain_of_thought.get("Output")
    if final_output not in ["0 (safe)", "1 (unsafe)"]: return False
    return True

# --- [Paste parse_llm_output function here] ---
# This function tries to extract the JSON content from the LLM response string
def parse_llm_output(llm_string: str, item_id: int) -> Optional[Dict]:
    # (Code from previous answer - cleans and parses LLM JSON output)
     # Prioritize finding JSON within markdown blocks
    json_blocks = []
    in_block = False
    current_block = ""
    for line in llm_string.splitlines():
        if line.strip().startswith("```json"):
            in_block = True
            current_block = ""
            continue
        elif line.strip().startswith("```") and in_block:
             in_block = False
             if current_block:
                 json_blocks.append(current_block.strip())
             current_block = ""
             continue
        elif in_block:
            current_block += line + "\n"

    # If no JSON block found, assume the whole string might be JSON
    if not json_blocks:
         json_blocks.append(llm_string.strip())

    for block in json_blocks:
        try:
            parsed = json.loads(block)
            # Check if LLM returned the full structure or just the inner CoT
            if isinstance(parsed, dict):
                if "chain_of_thought" in parsed and isinstance(parsed["chain_of_thought"], dict):
                    # Returned the full structure, extract CoT
                    print(f"[ID: {item_id}] Parsed full structure from LLM response.")
                    return parsed["chain_of_thought"]
                elif "Output" in parsed: # Heuristic: check if it looks like the inner CoT content
                    # Returned just the inner CoT object
                    print(f"[ID: {item_id}] Parsed inner CoT structure from LLM response.")
                    return parsed
                else:
                    # Returned a dictionary, but not in expected format
                    print(f"[ID: {item_id}] Parsed JSON dict, but not the expected structure: {list(parsed.keys())}")
                    # Fallback or further checks could be added here
            else:
                print(f"[ID: {item_id}] Parsed JSON, but it's not a dictionary.")

        except json.JSONDecodeError:
            print(f"[ID: {item_id}] Failed to parse block as JSON.")
            continue # Try next block if any

    print(f"[ID: {item_id}] Could not parse valid target JSON structure from LLM output.")
    return None # Return None if no block yielded the desired structure

# --- [NEW create_correction_prompt function pasted here] ---
def create_correction_prompt(fewshot_demos: List[Dict], problematic_output_str: str) -> str:
    # (Code defined above)
    prompt_parts = []
    prompt_parts.append(
        "You are an expert JSON formatter and data corrector. "
        "Your task is to analyze the provided 'Input Data' and correct it so that it strictly matches the required JSON format. "
        "The final output MUST be a valid JSON object containing exactly one key: 'chain_of_thought'. "
        "The value of 'chain_of_thought' MUST be a JSON object containing various analysis steps (keys can vary) "
        "and MUST conclude with a key 'Output' whose value is exactly the string '0 (safe)' or '1 (unsafe)'. "
        "If the input data is completely unusable or missing the core analysis, try to reconstruct a minimal valid structure containing at least the 'Output' key with a default value like '0 (safe)' if unsure, but prioritize fixing the input if possible."
    )
    prompt_parts.append("\nRequired Format Structure Example:")
    prompt_parts.append("""
{
  "chain_of_thought": {
    "Start": "Analysis begins...",
    "...": "...",
    "Final Judgment": "Based on the analysis...",
    "Output": "0 (safe)" OR "Output": "1 (unsafe)"
  }
}""")
    if fewshot_demos:
        prompt_parts.append("\nHere are examples demonstrating the desired *final* JSON structure:")
        for i, demo in enumerate(fewshot_demos):
            example_content = demo.get('A', '{}')
            if isinstance(example_content, dict):
                 example_str = json.dumps(example_content, indent=2, ensure_ascii=False)
            else:
                 example_str = str(example_content)
            prompt_parts.append(f"\nExample {i+1}:\n{example_str}")
    prompt_parts.append("\n---")
    prompt_parts.append("Input Data (This might be malformed, incomplete, a plain string, or incorrect JSON):")
    prompt_parts.append(f"```\n{problematic_output_str}\n```")
    prompt_parts.append("---\n")
    prompt_parts.append("Corrected JSON Output (Provide *only* the valid JSON object according to the required format):")
    return "\n".join(prompt_parts)


# --- Main Function Modified ---
def fix_json_outputs_by_correction(input_json_path: str, error_ids: List[int], output_json_path: str, failed_items_path: str):
    """
    Reads JSON, identifies items by error_ids, and uses LLM to *correct*
    their existing 'output' field if malformed. Saves results.
    """
    try:
        print(f"Reading input file: {input_json_path}...")
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Read {len(data)} records.")
    except Exception as e:
        print(f"Error reading input file {input_json_path}: {e}")
        return

    error_id_set = set(error_ids)
    print(f"Targeting {len(error_id_set)} IDs for potential correction.")

    config = GPTConfig()
    llm_handler = LLMHandler(config)

    processed_data = []
    failed_api_calls = [] # Store info about items where API call failed during correction

    print("\nStarting processing...")
    for item in tqdm(data, desc="Processing items"):
        item_id = item.get('id')
        original_item_copy = item.copy() # Keep a copy for the failed list if needed

        if item_id is None:
            processed_data.append(item) # Keep items without ID
            continue

        # Clean up potential error flags from previous runs *before* validation
        item.pop('llm_parse_error', None)
        item.pop('llm_format_error', None)
        item.pop('raw_llm_output', None)
        item.pop('llm_format_error_after_correction', None)
        item.pop('llm_parse_error_after_correction', None)

        if item_id in error_id_set:
            current_output = item.get('output')
            needs_correction = not validate_output_format(current_output)

            if needs_correction:
                print(f"\n[ID: {item_id}] Invalid format detected. Attempting LLM correction...")

                # Prepare input for the correction prompt
                # Use json.dumps for better string representation of dicts/lists, else just str()
                if current_output is None:
                   problematic_output_str = "None"
                elif isinstance(current_output, (dict, list)):
                   try:
                       problematic_output_str = json.dumps(current_output, indent=2, ensure_ascii=False)
                   except TypeError: # Handle potential non-serializable data if any
                       problematic_output_str = str(current_output)
                else:
                   problematic_output_str = str(current_output)

                fewshot_demos = item.get('fewshot_demos', []) # Use few-shots if available

                # Create the specific correction prompt
                correction_prompt = create_correction_prompt(fewshot_demos, problematic_output_str)
                # print(f"[ID: {item_id}] Correction Prompt Snippet:\n{correction_prompt[:500]}...") # For debugging

                # Call LLM for correction
                llm_response_str = llm_handler.call_llm_api(correction_prompt, item_id)

                if llm_response_str:
                    # Parse the LLM's corrected output
                    parsed_cot = parse_llm_output(llm_response_str, item_id) # Expects the inner CoT dict

                    if parsed_cot:
                        corrected_output_structure = {"chain_of_thought": parsed_cot}
                        # Validate the structure LLM returned
                        if validate_output_format(corrected_output_structure):
                            print(f"[ID: {item_id}] LLM correction successful and validated. Updating item.")
                            item['output'] = corrected_output_structure # Replace with corrected structure
                            # Error flags already removed at the start of loop for this item
                        else:
                            # LLM response parsed but still invalid
                            print(f"[ID: {item_id}] Warning: LLM response parsed BUT failed validation. Storing raw LLM output.")
                            item['output'] = {'raw_llm_output': llm_response_str, 'llm_format_error_after_correction': True}
                    else:
                        # Failed to parse LLM response
                        print(f"[ID: {item_id}] Warning: Failed to parse LLM correction response. Storing raw LLM output.")
                        item['output'] = {'raw_llm_output': llm_response_str, 'llm_parse_error_after_correction': True}
                else:
                    # LLM API call failed completely for correction
                    print(f"[ID: {item_id}] Correction API call failed. Adding original item to failed list.")
                    failure_info = original_item_copy # Use the unmodified copy
                    failure_info['failure_reason'] = "API call failed during correction attempt"
                    failed_api_calls.append(failure_info)
                    # Keep the original item (with its potentially broken output) in the main results for now
                    # Or you could decide to put items with failed API calls *only* in the failed list

            # else: # Item is in error_ids but already valid
            #    print(f"[ID: {item_id}] Format is already valid, no correction needed.")

        # Append the item (original, corrected, or with new error flags) to the main results
        processed_data.append(item)


    # --- Final Saving ---
    print("\nSaving results...")
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f_out:
            json.dump(processed_data, f_out, indent=2, ensure_ascii=False)
        print(f"Processed data saved to: {output_json_path}")
    except Exception as e:
        print(f"Error saving output file {output_json_path}: {e}")

    if failed_api_calls:
        print(f"\n{len(failed_api_calls)} items had failed API calls during correction attempt.")
        try:
            with open(failed_items_path, 'w', encoding='utf-8') as f_fail:
                json.dump(failed_api_calls, f_fail, indent=2, ensure_ascii=False)
            print(f"Failed API call item info saved to: {failed_items_path}")
        except Exception as e:
            print(f"Error saving failed items file {failed_items_path}: {e}")
    else:
        print("\nNo failed API calls recorded during correction attempts.")


def fix2_main(dataset):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_JSON = os.path.join(script_dir, f"../temp/{dataset}/output-k3_fixed.json")
    OUTPUT_JSON = os.path.join(script_dir, f"../temp/{dataset}/output-k3_corrected.json")
    FAILED_ITEMS_JSON = os.path.join(script_dir, f"../temp/{dataset}/failed_correction.json")

    ERROR_IDS_TO_FIX = [
        83, 102, 163, 189, 415, 648, 1438, 1492
    ]

    # Run the correction function
    fix_json_outputs_by_correction(INPUT_JSON, ERROR_IDS_TO_FIX, OUTPUT_JSON, FAILED_ITEMS_JSON)

    print("\nScript finished.")