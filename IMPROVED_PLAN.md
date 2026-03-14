# Improved Execution Plan: CLAM vs. TransMIL Benchmarking

This plan outlines the updated protocol for a comparative study between CLAM and TransMIL architectures, optimized for the current project state and hardware constraints (NVIDIA 1650, 4GB VRAM).

## 1. Research Objective
Benchmark the performance of **CLAM (Single-Branch)** and **TransMIL** on the task of **5-year survival classification** using Whole Slide Image (WSI) features from the TCGA-BRCA cohort.

## 2. Experimental Setup
- **Feature Extractor:** Fixed (UNI2-h embeddings).
- **Dataset:** TCGA-BRCA (Breast Invasive Carcinoma).
- **Target Variable:** Binary 5-year survival (`label`: 1 if deceased < 1825 days, 0 otherwise).
- **Validation Strategy:** 5-fold cross-validation (Stratified by Patient/Case ID).

## 3. Data Infrastructure
- **Feature Directory:** `TCGA-BRCA-features/TCGA/` (H5 format).
- **Master Manifest:** `master.csv` (contains `slide_id`, `case_id`, and `label`).
- **Splits:** `splits/` (Generated using `generate_splits.py`). Each fold directory (0-4) contains `train.csv`, `val.csv`, and `test.csv`.

## 4. Phase I: CLAM Implementation
CLAM serves as the attention-based MIL baseline.
- **Script:** `CLAM/main.py`
- **Configuration:**
  - `model_type`: `clam_sb`
  - `bag_loss`: `focal`
  - `bag_weight`: `0.9`
  - `lr`: `2e-5`
  - `reg`: `1e-3`
  - `early_stopping`: Enabled
- **Execution Command (Example Fold 0):**
  ```bash
  python CLAM/main.py --data_root_dir "TCGA-BRCA-features/TCGA" --split_dir "splits" --task task_2_tumor_subtyping --exp_code brca_fold_0_clam --model_type clam_sb --bag_loss focal --lr 2e-5 --k 5 --k_start 0 --k_end 1 --subtyping --bag_weight 0.9 --early_stopping
  ```

## 5. Phase II: TransMIL Implementation
TransMIL serves as the transformer-based challenger.
- **Script:** `TransMIL/train.py`
- **Configuration:** `TransMIL/TCGA_BRCA.yaml`
  - `optimizer.lr`: `2e-5`
  - `loss.base_loss`: `focal`
  - `epochs`: `200`
- **Execution Command (Example Fold 0):**
  ```bash
  python TransMIL/train.py --stage 'train' --config 'TransMIL/TCGA_BRCA.yaml' --gpus 0 --fold 0
  ```

## 6. Phase III: Evaluation & Statistical Analysis
Compare models based on:
- **Area Under the ROC Curve (AUC):** Primary metric for classification.
- **Concordance Index (C-Index):** Assessing risk ranking.
- **DeLong's Test:** Statistical significance of AUC differences.
- **Log-rank Test:** Comparing survival curves of high-risk vs. low-risk groups.

## 7. Phase IV: Interpretability
Generate attention heatmaps to visualize the regions of interest for both models using the scripts in `CLAM/create_heatmaps.py` (adapted for TransMIL attention weights).

---
*Last Updated: March 2026*
