import pandas as pd
import os

def main():
    split_dir = 'splits'
    for i in range(5):
        fold_dir = os.path.join(split_dir, str(i))
        train_file = os.path.join(fold_dir, 'train.csv')
        test_file = os.path.join(fold_dir, 'test.csv')

        if os.path.exists(train_file):
            df_train = pd.read_csv(train_file)
            if 'label' in df_train.columns:
                print(f"Fold {i} Train class distribution:\n{df_train['label'].value_counts()}")
            else:
                print(f"Fold {i} Train: 'label' column not found in {train_file}")
        else:
            print(f"Fold {i} Train: {train_file} not found")

        if os.path.exists(test_file):
            df_test = pd.read_csv(test_file)
            if 'label' in df_test.columns:
                print(f"Fold {i} Test class distribution:\n{df_test['label'].value_counts()}")
            else:
                print(f"Fold {i} Test: 'label' column not found in {test_file}")
        else:
            print(f"Fold {i} Test: {test_file} not found")

if __name__ == '__main__':
    main()