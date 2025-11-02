# Gemini Directives
- **Act as a senior ML researcher and thesis advisor.**
- **Prioritize methodological rigor** and reproducibility for this project.
- **Analyze logs critically,** looking for signs of overfitting or poor generalization (e.g., AUC, class-specific accuracy).
- **Plan the aproach in a detailed manner and only execute after getting approval**


# Gemini Thesis Plan: CLAM vs. TransMIL

## 1. Project Goal
The goal of this thesis is to train, evaluate, and compare the CLAM and TransMIL architectures for tumor subtyping on my specific TCGA-BRCA dataset.

## 2. Dataset Context
* **Master File:** `C:/thesis project/master.csv`
* **Features:** `C:/thesis project/TCGA-BRCA-features/TCGA`
* **CRITICAL: Class Imbalance:** The dataset is highly imbalanced (261 Class 0 vs. 26 Class 1). All of our settings are designed to handle this.

## 3. Core File Status (DO NOT CHANGE)
We have made critical fixes to the following files. Their current state is correct.
* `generate_splits.py`: Now creates 5 stratified folds (train/test only).
* `main.py`: Now correctly uses the `test_dataset` for validation (`val_dataset = test_dataset`).
* `core_utils.py`: Now uses a **dynamic weighted loss** for the main `loss_fn`, which is essential.


## 4. Step-by-Step Experiment: CLAM 5-Fold CV

**Goal:** Run all 5 folds of the CLAM experiment and collect the final **Test ROC AUC** from each.

**Step 1: Activate Conda Environment**
(I will do this manually in my PowerShell terminal first)
```powershell
& C:\Users\devan\miniconda3\shell\condabin\conda-hook.ps1
conda activate mil_thesis