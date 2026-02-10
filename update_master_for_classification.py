import pandas as pd

# 1. Configuration
MASTER_CSV_PATH = 'master.csv'

# 2. Load Data
print(f"Loading and updating {MASTER_CSV_PATH} for binary classification...")
df = pd.read_csv(MASTER_CSV_PATH)

# 3. Create Binary Label for 5-Year Survival
# Initialize the label column
df['survival_label_5yr'] = 0

# Set label to 1 for mortality events within 5 years (1825 days)
five_year_mortality_mask = (df['OS_status'] == 1) & (df['OS_time'] < 1825)
df.loc[five_year_mortality_mask, 'survival_label_5yr'] = 1

# 4. Rename for Compatibility
# The CLAM demo script expects a column named 'label' for classification tasks.
df = df.rename(columns={'survival_label_5yr': 'label'})

# 5. Save the File
df.to_csv(MASTER_CSV_PATH, index=False)

# 6. Add Logging
print("Successfully updated master.csv with binary labels.")
print("\nFinal label distribution:")
print(df['label'].value_counts())
