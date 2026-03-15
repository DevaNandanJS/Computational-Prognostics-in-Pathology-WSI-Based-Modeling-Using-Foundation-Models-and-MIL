import os
import pandas as pd
import h5py

data_dir = r'D:/thesis project/TCGA-BRCA_IDC'
master_csv = 'master.csv'
backup_csv = 'master_backup_v3.csv'

# 1. Identify corrupted files
corrupted_slides = []
h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]

print(f"Scanning {len(h5_files)} files with Deep Read Check...")

for i, f in enumerate(h5_files):
    if i % 100 == 0:
        print(f"Progress: {i}/{len(h5_files)} files checked...")
        
    path = os.path.join(data_dir, f)
    slide_id = f.replace('.h5', '')
    
    try:
        with h5py.File(path, 'r') as hf:
            if 'features' in hf:
                # DEEP CHECK: Try reading the first 10 rows of data
                # This is where the training script actually fails
                _ = hf['features'][0:10]
            else:
                # If 'features' isn't the key, check whatever is there
                k = list(hf.keys())[0]
                _ = hf[k][0:10]
    except Exception as e:
        print(f"\n[!] DEEP CORRUPTION FOUND in {f}: {e}")
        corrupted_slides.append(slide_id)

if not corrupted_slides:
    print("\nNo corrupted files found.")
else:
    print(f"\nFound {len(corrupted_slides)} corrupted slides.")

    # 2. Update master.csv
    df = pd.read_csv(master_csv)
    initial_len = len(df)
    
    # Remove the corrupted ones
    df = df[~df['slide_id'].isin(corrupted_slides)]
    
    final_len = len(df)
    
    if initial_len != final_len:
        # Create backup
        if os.path.exists(master_csv):
            if os.path.exists(backup_csv):
                os.remove(backup_csv)
            os.rename(master_csv, backup_csv)
        df.to_csv(master_csv, index=False)
        print(f"Removed {initial_len - final_len} rows from {master_csv}. Backup saved as {backup_csv}.")
    else:
        print("Corrupted slides were not in master.csv or already removed.")
