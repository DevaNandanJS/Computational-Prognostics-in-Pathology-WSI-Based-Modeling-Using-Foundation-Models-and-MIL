import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

def find_optimal_threshold(results_file):
    """
    Loads validation results and finds the optimal classification threshold.
    """
    print(f"Loading results from: {results_file}")
    with open(results_file, 'rb') as f:
        # The results file is a dictionary where keys are slide_ids
        # and values are dicts with 'prob' and 'label'
        results = pickle.load(f)

    # Consolidate probabilities and labels from the dictionary into numpy arrays
    y_probs = []
    y_true = []
    for slide_id, data in results.items():
        # Assuming the positive class is at index 1
        y_probs.append(data['prob'][0][1]) 
        y_true.append(data['label'])
    
    y_probs = np.array(y_probs)
    y_true = np.array(y_true)

    print(f"\nFound {len(y_true)} samples in the validation set.")
    print(f"Class distribution: {np.bincount(y_true)}")

    # Iterate through a range of potential thresholds
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_f1 = 0
    best_threshold = 0

    for threshold in thresholds:
        # Get predictions based on the current threshold
        y_pred = (y_probs >= threshold).astype(int)
        
        # Calculate F1 score
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print("\n--- Optimal Threshold Search (based on F1-Score) ---")
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Best F1-Score on Val Set: {best_f1:.4f}")

    # Evaluate the performance using the best threshold
    y_pred_optimal = (y_probs >= best_threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred_optimal)
    
    cm = confusion_matrix(y_true, y_pred_optimal)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    print("\n--- Performance with Optimal Threshold ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall for Class 1): {sensitivity:.4f} ({tp}/{tp+fn})")
    print(f"Specificity (Recall for Class 0): {specificity:.4f} ({tn}/{tn+fp})")
    print("\nConfusion Matrix:")
    print(f"         [Pred 0] [Pred 1]")
    print(f"[True 0]    {tn}        {fp}")
    print(f"[True 1]    {fn}        {tp}")

if __name__ == '__main__':
    # Path to the results file from our best run
    # This assumes the script is run from the root 'C:\thesis project' directory
    best_run_results_path = 'CLAM/results/brca_5yr_classification_clam_fold_0_focal_loss_bweight_0.9_s1/split_0_results.pkl'
    find_optimal_threshold(best_run_results_path)
