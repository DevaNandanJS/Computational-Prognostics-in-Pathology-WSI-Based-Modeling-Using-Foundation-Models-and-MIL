import pandas as pd
import os

# 1. Define Paths
clinical_file_path = r'C:\thesis project\TCGA-CDR-SupplementalTableS1.xlsx'
features_dir_path = r'C:\thesis project\TCGA-BRCA-features\TCGA'

# 2. Load Clinical Data
print("Loading clinical data...")
clinical_df = pd.read_excel(clinical_file_path, sheet_name='TCGA-CDR')

# 3. Process Clinical Data
clinical_df = clinical_df[['bcr_patient_barcode', 'OS', 'OS.time']]
clinical_df = clinical_df.rename(columns={
    'bcr_patient_barcode': 'case_id',
    'OS': 'OS_status',
    'OS.time': 'OS_time'
})

# 4. List Feature Files
print(f"Scanning for feature files in {features_dir_path}...")
h5_files = [f for f in os.listdir(features_dir_path) if f.endswith('.h5')]
print(f"Found {len(h5_files)} slide files.")

# 5. Process Feature Files
slide_ids = [os.path.splitext(f)[0] for f in h5_files]
case_ids = [s[:12] for s in slide_ids]

features_df = pd.DataFrame({
    'slide_id': slide_ids,
    'case_id': case_ids
})

# 6. Merge DataFrames
print("Merging data...")
merged_df = pd.merge(clinical_df, features_df, on='case_id', how='inner')

# 7. Finalize and Save
final_df = merged_df[['case_id', 'slide_id', 'OS_time', 'OS_status']]
final_df = final_df.dropna()

final_df = final_df.astype({
    'OS_time': int,
    'OS_status': int
})

output_path = 'master.csv'
final_df.to_csv(output_path, index=False)

print(f"Successfully created {output_path} with {len(final_df)} records.")
