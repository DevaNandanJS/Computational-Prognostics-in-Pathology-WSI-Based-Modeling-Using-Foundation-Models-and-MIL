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

            # --- This block creates train, val, and test sets ---
            train_df = df.iloc[train_ids]
            test_df = df.iloc[test_ids]

            # Split the training data into a new training set and a validation set
            val_frac = args.val_frac
            if val_frac > 0:
                print(f"Splitting {val_frac*100}% of training data for validation...")
                # Stratify the split to maintain label distribution in both train and val sets.
                train_df, val_df = train_test_split(train_df, test_size=val_frac, stratify=train_df[args.label_col], random_state=args.seed)
            else:
                val_df = pd.DataFrame() # Create empty dataframe if no validation set is needed

            train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
            val_df.to_csv(os.path.join(fold_dir, 'val.csv'), index=False)
            test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)
            
            print(f"Fold {i}: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
            print(f"Train distribution:\n{train_df[args.label_col].value_counts(normalize=True)}")
            if not val_df.empty:
                print(f"Validation distribution:\n{val_df[args.label_col].value_counts(normalize=True)}")
            print(f"Test distribution:\n{test_df[args.label_col].value_counts(normalize=True)}")

            # --- END OF BLOCK ---

        print(f"\nSuccessfully created and saved {args.k} splits in the '{args.split_dir}' directory.")

    else:
        print(f"Task '{args.task}' not implemented in this script.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create cross-validation splits')
    parser.add_argument('--data_dir', type=str, default='master.csv', help='Path to the master CSV file')
    parser.add_argument('--task', type=str, choices=['task_2_tumor_subtyping', 'survival'], default='task_2_tumor_subtyping', help='Task type')
    parser.add_argument('--label_col', type=str, default='label', help='Name of the label column')
    parser.add_argument('--k', type=int, default=5, help='Number of folds')
    parser.add_argument('--val_frac', type=float, default=0.2, help='Fraction of training data to use for validation.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--split_dir', type=str, default='splits', help='Directory to save the split files')

    args = parser.parse_args()
    main(args)
