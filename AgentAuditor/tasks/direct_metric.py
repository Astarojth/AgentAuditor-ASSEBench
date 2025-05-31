import json
from typing import List, Dict, Union
import os

def calculate_metrics(data: List[Dict[str, Union[int, str, list, dict]]]) -> Dict[str, Union[float, bool, set, Dict]]:
    """
    Calculate accuracy, precision, recall, and F1 score for binary classification results.
    Also handle anomalous values, with special treatment for -1 values.

    Args:
        data: List of dictionaries containing 'label' and 'pred' fields

    Returns:
        Dictionary containing standard metrics, anomaly information, and metrics for different -1 handling strategies
    """
    # Check for anomalous prediction values
    anomalous_values = set()
    has_negative_one = False

    for item in data:
        # Ensure 'pred' exists and handle potential non-numeric types gracefully if needed,
        # though the type hint suggests it should be int/str/list/dict.
        # Assuming 'pred' will be numeric (0, 1, -1, etc.) for calculations.
        pred_val = item.get('pred')
        if pred_val not in [0, 1]:
            anomalous_values.add(pred_val)
            if pred_val == -1:
                has_negative_one = True

    has_anomalies = len(anomalous_values) > 0
    total_items = len(data)

    if total_items == 0:
        # Handle empty data case
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'has_anomalies': False,
            'anomalous_values': set()
        }

    # --- Standard metrics calculation (Original method) ---
    true_positives = sum(1 for item in data if item['label'] == 1 and item['pred'] == 1)
    false_positives = sum(1 for item in data if item['label'] == 0 and item['pred'] == 1)
    false_negatives = sum(1 for item in data if item['label'] == 1 and item['pred'] == 0)
    # Calculate True Negatives needed for accuracy
    true_negatives = sum(1 for item in data if item['label'] == 0 and item['pred'] == 0)

    # Calculate Accuracy
    accuracy = (true_positives + true_negatives) / total_items

    # Calculate precision
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    # Calculate recall
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    result = {
        'accuracy': accuracy, # Added accuracy
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'has_anomalies': has_anomalies,
        'anomalous_values': anomalous_values
    }

    # --- If -1 is found, calculate metrics with both treatment strategies ---
    if has_negative_one:
        # Strategy 1: Treat all -1 as 0
        data_neg_as_zero = []
        for item in data:
            new_item = item.copy()
            if new_item.get('pred') == -1: # Use .get() for safety
                new_item['pred'] = 0
            # Only include items where prediction is now 0 or 1 for metric calculation
            if new_item.get('pred') in [0, 1]:
                data_neg_as_zero.append(new_item)

        # Calculate metrics for -1 as 0
        tp_neg_as_zero = sum(1 for item in data_neg_as_zero if item['label'] == 1 and item['pred'] == 1)
        fp_neg_as_zero = sum(1 for item in data_neg_as_zero if item['label'] == 0 and item['pred'] == 1)
        fn_neg_as_zero = sum(1 for item in data_neg_as_zero if item['label'] == 1 and item['pred'] == 0)
        tn_neg_as_zero = sum(1 for item in data_neg_as_zero if item['label'] == 0 and item['pred'] == 0) # Added TN

        accuracy_neg_as_zero = (tp_neg_as_zero + tn_neg_as_zero) / total_items # Total items remains the same denominator

        precision_neg_as_zero = tp_neg_as_zero / (tp_neg_as_zero + fp_neg_as_zero) if (tp_neg_as_zero + fp_neg_as_zero) > 0 else 0
        recall_neg_as_zero = tp_neg_as_zero / (tp_neg_as_zero + fn_neg_as_zero) if (tp_neg_as_zero + fn_neg_as_zero) > 0 else 0
        f1_neg_as_zero = 2 * (precision_neg_as_zero * recall_neg_as_zero) / (precision_neg_as_zero + recall_neg_as_zero) if (precision_neg_as_zero + recall_neg_as_zero) > 0 else 0

        # Strategy 2: Treat all -1 as 1
        data_neg_as_one = []
        for item in data:
            new_item = item.copy()
            if new_item.get('pred') == -1: # Use .get() for safety
                new_item['pred'] = 1
            # Only include items where prediction is now 0 or 1 for metric calculation
            if new_item.get('pred') in [0, 1]:
                 data_neg_as_one.append(new_item)

        # Calculate metrics for -1 as 1
        tp_neg_as_one = sum(1 for item in data_neg_as_one if item['label'] == 1 and item['pred'] == 1)
        fp_neg_as_one = sum(1 for item in data_neg_as_one if item['label'] == 0 and item['pred'] == 1)
        fn_neg_as_one = sum(1 for item in data_neg_as_one if item['label'] == 1 and item['pred'] == 0)
        tn_neg_as_one = sum(1 for item in data_neg_as_one if item['label'] == 0 and item['pred'] == 0) # Added TN

        accuracy_neg_as_one = (tp_neg_as_one + tn_neg_as_one) / total_items # Total items remains the same denominator

        precision_neg_as_one = tp_neg_as_one / (tp_neg_as_one + fp_neg_as_one) if (tp_neg_as_one + fp_neg_as_one) > 0 else 0
        recall_neg_as_one = tp_neg_as_one / (tp_neg_as_one + fn_neg_as_one) if (tp_neg_as_one + fn_neg_as_one) > 0 else 0
        f1_neg_as_one = 2 * (precision_neg_as_one * recall_neg_as_one) / (precision_neg_as_one + recall_neg_as_one) if (precision_neg_as_one + recall_neg_as_one) > 0 else 0

        # Add these results to the return dictionary
        result['neg_one_as_zero'] = {
            'accuracy': accuracy_neg_as_zero, # Added accuracy
            'precision': precision_neg_as_zero,
            'recall': recall_neg_as_zero,
            'f1': f1_neg_as_zero
        }

        result['neg_one_as_one'] = {
            'accuracy': accuracy_neg_as_one, # Added accuracy
            'precision': precision_neg_as_one,
            'recall': recall_neg_as_one,
            'f1': f1_neg_as_one
        }

    return result


def direct_metric_main(dataset):
    # Read the JSON file
    # <<< Make sure this path is correct for your environment >>>
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, f"../direct_temp/{dataset}_output.json")
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Please ensure the path is correct.")
        # Example fallback: create dummy data for testing
        print("Using dummy data for demonstration.")
        data = [
            {'label': 1, 'pred': 1}, {'label': 0, 'pred': 0}, {'label': 1, 'pred': 0},
            {'label': 0, 'pred': 1}, {'label': 1, 'pred': 1}, {'label': 0, 'pred': -1},
            {'label': 1, 'pred': -1}, {'label': 0, 'pred': 0}, {'label': 1, 'pred': 2}
        ]
        # return # Optionally exit if file not found and no fallback desired
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return # Exit if JSON is invalid
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return # Exit on other errors

    # Calculate metrics
    metrics = calculate_metrics(data)

    # Print results with formatted output
    print("\nClassification Metrics (Original - Anomalies Ignored):")
    print("-" * 50)
    # Added Accuracy print statement
    print(f"Accuracy:  {metrics.get('accuracy', 0.0):.4f}")
    print(f"Precision: {metrics.get('precision', 0.0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0.0):.4f}")
    print(f"F1 Score:  {metrics.get('f1', 0.0):.4f}")


    # Print anomaly information
    if metrics.get('has_anomalies', False):
        print("\nWARNING: Anomalous prediction values detected!")
        # Use .get() for safer access, providing default empty set
        print(f"Found values other than 0 and 1 in 'pred': {metrics.get('anomalous_values', set())}")

        # If -1 is one of the anomalous values, print additional metrics
        if -1 in metrics.get('anomalous_values', set()):
            # Check if the specific keys exist before trying to access them
            if 'neg_one_as_zero' in metrics:
                print("\nMetrics when treating -1 as 0:")
                print("-" * 50)
                # Added Accuracy print statement
                print(f"Accuracy:  {metrics['neg_one_as_zero'].get('accuracy', 0.0):.4f}")
                print(f"Precision: {metrics['neg_one_as_zero'].get('precision', 0.0):.4f}")
                print(f"Recall:    {metrics['neg_one_as_zero'].get('recall', 0.0):.4f}")
                print(f"F1 Score:  {metrics['neg_one_as_zero'].get('f1', 0.0):.4f}")

            if 'neg_one_as_one' in metrics:
                print("\nMetrics when treating -1 as 1:")
                print("-" * 50)
                # Added Accuracy print statement
                print(f"Accuracy:  {metrics['neg_one_as_one'].get('accuracy', 0.0):.4f}")
                print(f"Precision: {metrics['neg_one_as_one'].get('precision', 0.0):.4f}")
                print(f"Recall:    {metrics['neg_one_as_one'].get('recall', 0.0):.4f}")
                print(f"F1 Score:  {metrics['neg_one_as_one'].get('f1', 0.0):.4f}")
    elif not metrics.get('has_anomalies', True): # Handles case where anomalies aren't present but key might be missing
         print("\nNo anomalous prediction values detected.")
