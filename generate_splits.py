import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import os
import argparse
from sklearn.model_selection import train_test_split

def main(args):
    df = pd.read_csv(args.data_dir)
    print(f"Loading data from {os.path.basename(args.data_dir)}...")

    if args.task == 'task_2_tumor_subtyping':
        # Create case_id if it doesn't exist
        if 'case_id' not in df.columns:
            df['case_id'] = df['slide_id'].apply(lambda x: '- '.join(x.split('-')[:3]))
        
        skf = StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
        print("Initializing StratifiedGroupKFold splitter...")
        
        if args.label_col not in df.columns:
            print(f"Error: Label column '{args.label_col}' not found in {args.data_dir}")
            return

        os.makedirs(args.split_dir, exist_ok=True)
        print(f"Created directory: {args.split_dir}")
        print(f"Generating {args.k} cross-validation splits...")

        for i, (train_ids, test_ids) in enumerate(skf.split(df, df[args.label_col], groups=df['case_id'])):
            fold_dir = os.path.join(args.split_dir, str(i))
            os.makedirs(fold_dir, exist_ok=True)

            # --- This block now ONLY creates train and test ---
            train_df = df.iloc[train_ids]
            test_df = df.iloc[test_ids]

            train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
            test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)
            
            print(f"Fold {i}: {len(train_df)} train, {len(test_df)} test")
            # --- END OF BLOCK ---

        print(f"\nSuccessfully created and saved {args.k} splits in the '{args.split_dir}' directory.")

    else:
        print(f"Task '{args.task}' not implemented in this script.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create cross-validation splits')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the master CSV file')
    parser.add_argument('--task', type=str, choices=['task_2_tumor_subtyping', 'survival'], required=True, help='Task type')
    parser.add_argument('--label_col', type=str, required=True, help='Name of the label column')
    parser.add_argument('--k', type=int, default=5, help='Number of folds')
    parser.add_argument('--val_frac', type=float, default=0.0, help='Fraction of training data to use for validation (NOT USED, but kept for compatibility)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--split_dir', type=str, default='splits', help='Directory to save the split files')

    args = parser.parse_args()
    main(args)
