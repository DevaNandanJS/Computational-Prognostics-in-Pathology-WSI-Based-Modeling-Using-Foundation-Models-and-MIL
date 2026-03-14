import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import glob
from pathlib import Path

# Setup plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def load_clam_results(clam_results_dir, folds=[0, 1, 2, 3, 4]):
    """
    Load CLAM results from pickle files.
    """
    all_fold_results = []
    
    for fold in folds:
        pkl_path = os.path.join(clam_results_dir, f'split_{fold}_results.pkl')
        if not os.path.exists(pkl_path):
            print(f"Warning: CLAM result for fold {fold} not found at {pkl_path}")
            continue
            
        with open(pkl_path, 'rb') as f:
            # In CLAM/main.py, split_X_results.pkl contains just the patient_results dict
            patient_results = pickle.load(f)
            
            slide_ids = []
            probs = []
            labels = []
            
            for slide_id, res in patient_results.items():
                slide_ids.append(slide_id)
                # CLAM saves prob as [1, 2], we want prob for class 1 (tumor/deceased)
                # Ensure we handle different shapes [2] or [1, 2]
                p = res['prob']
                if hasattr(p, 'shape'):
                    if len(p.shape) > 1:
                        probs.append(p[0, 1])
                    else:
                        probs.append(p[1])
                else:
                    # If it's a list or something else
                    probs.append(p[1])
                labels.append(res['label'])
            
            fold_df = pd.DataFrame({
                'slide_id': slide_ids,
                'prob': probs,
                'label': labels,
                'fold': fold
            })
            all_fold_results.append(fold_df)
            
    if not all_fold_results:
        return None
    return pd.concat(all_fold_results)

def load_transmil_results(transmil_logs_dir, folds=[0, 1, 2, 3, 4]):
    """
    Load TransMIL results from CSV files.
    """
    all_fold_results = []
    
    for fold in folds:
        # TransMIL structure: logs/TCGA_BRCA/foldX/test_predictions.csv
        csv_path = os.path.join(transmil_logs_dir, f'fold{fold}', 'test_predictions.csv')
        if not os.path.exists(csv_path):
            # Try recursive search just in case
            matches = list(Path(transmil_logs_dir).glob(f'**/fold{fold}/test_predictions.csv'))
            if matches:
                csv_path = str(matches[0])
            else:
                print(f"Warning: TransMIL result for fold {fold} not found at {csv_path}")
                continue
                
        fold_df = pd.read_csv(csv_path)
        fold_df['fold'] = fold
        # TransMIL saves prob_0 and prob_1
        fold_df = fold_df.rename(columns={'prob_1': 'prob'})
        all_fold_results.append(fold_df)
        
    if not all_fold_results:
        return None
    return pd.concat(all_fold_results)

def calculate_metrics(df):
    """
    Calculate performance metrics from a dataframe of results.
    """
    y_true = df['label'].values
    y_prob = df['prob'].values
    y_pred = (y_prob >= 0.5).astype(int)
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    metrics = {
        'AUC': roc_auc,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return metrics, fpr, tpr

def plot_roc_curves(model_results, save_path='roc_comparison.png'):
    """
    Plot ROC curves for multiple models.
    model_results is a dict: {model_name: results_df}
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, df in model_results.items():
        # Overall ROC
        metrics, fpr, tpr = calculate_metrics(df)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {metrics["AUC"]:.3f})')
        
        # Calculate per-fold AUC for mean/std
        fold_aucs = []
        for fold in df['fold'].unique():
            fold_df = df[df['fold'] == fold]
            if len(fold_df['label'].unique()) > 1:
                fold_m, _, _ = calculate_metrics(fold_df)
                fold_aucs.append(fold_m['AUC'])
        
        if fold_aucs:
            print(f"{model_name} Mean Fold AUC: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Comparison')
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC comparison plot saved to {save_path}")

def plot_confusion_matrices(model_results, save_dir='.'):
    """
    Plot confusion matrices for each model.
    """
    for model_name, df in model_results.items():
        y_true = df['label'].values
        y_pred = (df['prob'].values >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix: {model_name}')
        plt.savefig(os.path.join(save_dir, f'confusion_matrix_{model_name.lower()}.png'), dpi=300)
        plt.close()

def main():
    # Paths (adjust based on actual execution results)
    # CLAM results directory (the one containing split_X_results.pkl)
    clam_results_dir = 'CLAM/results/brca_5yr_classification_clam_fold_0_focal_loss_s1'
    # TransMIL logs directory (the one containing foldX/test_predictions.csv)
    transmil_logs_dir = 'TransMIL/logs/TCGA_BRCA'
    
    print("Loading results...")
    clam_df = load_clam_results(clam_results_dir)
    transmil_df = load_transmil_results(transmil_logs_dir)
    
    model_results = {}
    if clam_df is not None:
        model_results['CLAM'] = clam_df
        clam_m, _, _ = calculate_metrics(clam_df)
        print("\nCLAM Overall Metrics:")
        for k, v in clam_m.items():
            print(f"  {k}: {v:.4f}")
            
    if transmil_df is not None:
        model_results['TransMIL'] = transmil_df
        transmil_m, _, _ = calculate_metrics(transmil_df)
        print("\nTransMIL Overall Metrics:")
        for k, v in transmil_m.items():
            print(f"  {k}: {v:.4f}")
            
    if model_results:
        os.makedirs('evaluation_results', exist_ok=True)
        plot_roc_curves(model_results, save_path='evaluation_results/roc_comparison.png')
        plot_confusion_matrices(model_results, save_dir='evaluation_results')
        
        # Save combined metrics to CSV
        summary_rows = []
        for model_name, df in model_results.items():
            m, _, _ = calculate_metrics(df)
            m['Model'] = model_name
            summary_rows.append(m)
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv('evaluation_results/performance_summary.csv', index=False)
        print("\nPerformance summary saved to evaluation_results/performance_summary.csv")
    else:
        print("No results found to compare.")

if __name__ == "__main__":
    main()
