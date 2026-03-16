import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import ttest_rel
import pickle
import glob
from pathlib import Path

# Setup plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

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
            patient_results = pickle.load(f)
            
            slide_ids = []
            probs = []
            labels = []
            
            for slide_id, res in patient_results.items():
                slide_ids.append(slide_id)
                p = res['prob']
                if hasattr(p, 'shape') and len(p.shape) > 1:
                    probs.append(p[0, 1])
                else:
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
        csv_path = os.path.join(transmil_logs_dir, f'fold{fold}', 'test_predictions.csv')
        if not os.path.exists(csv_path):
            matches = list(Path(transmil_logs_dir).glob(f'**/fold{fold}/test_predictions.csv'))
            if matches:
                csv_path = str(matches[0])
            else:
                print(f"Warning: TransMIL result for fold {fold} not found at {csv_path}")
                continue
                
        fold_df = pd.read_csv(csv_path)
        fold_df['fold'] = fold
        fold_df = fold_df.rename(columns={'prob_1': 'prob'})
        all_fold_results.append(fold_df)
        
    if not all_fold_results:
        return None
    return pd.concat(all_fold_results)

def calculate_metrics(df):
    """
    Calculate performance metrics from a dataframe of results.
    Includes Specificity, which is crucial for medical applications.
    """
    y_true = df['label'].values
    y_prob = df['prob'].values
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Handle cases where a class is not present in a fold
    if len(np.unique(y_true)) < 2:
        return {
            'AUC': np.nan, 'Accuracy': np.nan, 'Precision': np.nan,
            'Recall': np.nan, 'F1-score': np.nan, 'Specificity': np.nan
        }, None, None

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'AUC': roc_auc,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-score': f1_score(y_true, y_pred, zero_division=0),
        'Specificity': specificity
    }
    
    return metrics, fpr, tpr

def plot_roc_curves(model_results, per_fold_metrics, save_path='roc_comparison.png'):
    """
    Plot ROC curves for multiple models, including mean AUC from folds.
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, df in model_results.items():
        # Overall ROC from all concatenated folds
        _, fpr, tpr = calculate_metrics(df)
        
        # Get mean AUC from per-fold calculations
        mean_auc = per_fold_metrics[model_name]['AUC']['mean']
        
        if fpr is not None and tpr is not None:
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (Mean AUC = {mean_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Comparison')
    plt.legend(loc="lower right")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"ROC comparison plot saved to {save_path}")

def plot_confusion_matrices(model_results, save_dir='.'):
    """
    Plot confusion matrices for each model based on aggregated results.
    """
    for model_name, df in model_results.items():
        y_true = df['label'].values
        y_pred = (df['prob'].values >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Aggregated Confusion Matrix: {model_name}')
        save_path = os.path.join(save_dir, f'confusion_matrix_{model_name.lower()}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix plot saved to {save_path}")

def main():
    # --- Configuration ---
    # Define which folds to load and compare for a fair "apples-to-apples" analysis.
    # This is crucial if you haven't completed all 5 folds for every model.
    # For the final thesis, this should be [0, 1, 2, 3, 4].
    FOLDS_TO_COMPARE = [0, 1, 2, 3, 4] 

    clam_results_dir = 'CLAM/results/brca_5yr_classification_clam_final_s1'
    transmil_logs_dir = 'logs/TransMIL/TCGA_BRCA'
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Running Comparison for Folds: {FOLDS_TO_COMPARE} ---")

    # --- Load Data ---
    print("Loading results...")
    clam_df = load_clam_results(clam_results_dir, folds=FOLDS_TO_COMPARE)
    transmil_df = load_transmil_results(transmil_logs_dir, folds=FOLDS_TO_COMPARE)
    
    model_dfs = {'CLAM': clam_df, 'TransMIL': transmil_df}
    
    # --- Per-Fold Analysis ---
    all_fold_metrics = []
    per_fold_summary = {} # For statistical tests
    
    for model_name, df in model_dfs.items():
        if df is None:
            print(f"Skipping {model_name}, no data found.")
            continue

        per_fold_summary[model_name] = {}
        
        for fold in sorted(df['fold'].unique()):
            fold_df = df[df['fold'] == fold]
            metrics, _, _ = calculate_metrics(fold_df)
            metrics['model'] = model_name
            metrics['fold'] = fold
            all_fold_metrics.append(metrics)
            
            # Store AUC for t-test
            if 'AUC' not in per_fold_summary[model_name]:
                per_fold_summary[model_name]['AUC'] = []
            per_fold_summary[model_name]['AUC'].append(metrics['AUC'])

    if not all_fold_metrics:
        print("No results found to compare.")
        return

    # Save per-fold results to a CSV
    fold_metrics_df = pd.DataFrame(all_fold_metrics).round(4)
    fold_metrics_df.to_csv(os.path.join(output_dir, 'performance_per_fold.csv'), index=False)
    print(f"\nPer-fold performance metrics saved to {os.path.join(output_dir, 'performance_per_fold.csv')}")

    # --- Summarize Metrics (Mean ± Std) ---
    summary_metrics = {}
    for model_name in model_dfs.keys():
        if model_dfs[model_name] is None: continue
        model_fold_df = fold_metrics_df[fold_metrics_df['model'] == model_name]
        
        summary_metrics[model_name] = {}
        for metric in ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Specificity']:
            mean_val = model_fold_df[metric].mean()
            std_val = model_fold_df[metric].std()
            summary_metrics[model_name][metric] = {'mean': mean_val, 'std': std_val}

    # Format for printing and saving
    summary_df_rows = []
    for metric in ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Specificity']:
        row = {'Metric': metric}
        for model_name in summary_metrics.keys():
            mean = summary_metrics[model_name][metric]['mean']
            std = summary_metrics[model_name][metric]['std']
            row[f'{model_name} (Mean ± Std)'] = f"{mean:.3f} ± {std:.3f}"
        summary_df_rows.append(row)
        
    summary_df = pd.DataFrame(summary_df_rows)
    summary_df.to_csv(os.path.join(output_dir, 'performance_summary_detailed.csv'), index=False)
    print(f"Detailed performance summary saved to {os.path.join(output_dir, 'performance_summary_detailed.csv')}")
    print("\n--- Performance Summary (Mean ± Std) ---")
    print(summary_df.to_string(index=False))

    # --- Statistical Significance (Paired T-Test on AUC) ---
    if 'CLAM' in per_fold_summary and 'TransMIL' in per_fold_summary:
        clam_aucs = per_fold_summary['CLAM']['AUC']
        transmil_aucs = per_fold_summary['TransMIL']['AUC']
        
        if len(clam_aucs) == len(transmil_aucs) and len(clam_aucs) > 1:
            stat, p_value = ttest_rel(clam_aucs, transmil_aucs)
            
            stats_summary = (
                f"Paired T-Test on AUC scores between CLAM and TransMIL:\n"
                f"---------------------------------------------------\n"
                f"CLAM AUCs:     {[round(x, 3) for x in clam_aucs]}\n"
                f"TransMIL AUCs: {[round(x, 3) for x in transmil_aucs]}\n"
                f"\nT-statistic: {stat:.4f}\n"
                f"P-value:     {p_value:.4f}\n\n"
                f"Conclusion: "
            )
            if p_value < 0.05:
                stats_summary += "The difference in performance IS statistically significant."
            else:
                stats_summary += "The difference in performance IS NOT statistically significant."
            
            print("\n--- Statistical Significance ---")
            print(stats_summary)
            with open(os.path.join(output_dir, 'statistical_summary.txt'), 'w') as f:
                f.write(stats_summary)
            print(f"Statistical analysis summary saved to {os.path.join(output_dir, 'statistical_summary.txt')}")

    # --- Generate Plots ---
    if model_dfs['CLAM'] is not None or model_dfs['TransMIL'] is not None:
        valid_model_dfs = {k:v for k,v in model_dfs.items() if v is not None}
        plot_roc_curves(valid_model_dfs, summary_metrics, save_path=os.path.join(output_dir, 'roc_comparison.png'))
        plot_confusion_matrices(valid_model_dfs, save_dir=output_dir)

if __name__ == "__main__":
    main()
