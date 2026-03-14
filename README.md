# Benchmarking MIL Architectures: CLAM vs. TransMIL for 5-Year Survival Prediction

This repository contains the code and configuration for a comparative study between **CLAM (Clustering-constrained Attention Multiple Instance Learning)** and **TransMIL (Transformer-based Correlated Multiple Instance Learning)** in the context of computational pathology.

## 🚀 Overview
The goal of this project is to benchmark two state-of-the-art MIL architectures on the task of predicting 5-year survival from Whole Slide Images (WSIs). We use the **TCGA-BRCA** cohort and leverage pre-computed features from the **UNI2-h** foundation model.

### Key Features
- **Dataset:** TCGA-BRCA (Breast Invasive Carcinoma).
- **Features:** UNI2-h embeddings (1024-dim) stored in HDF5 format.
- **Task:** Binary 5-year survival classification.
- **Validation:** 5-fold cross-validation with Stratified Group K-Fold (stratified by patient).

## 📁 Project Structure
```text
D:/thesis project/
├── CLAM/               # CLAM repository and adaptations
├── TransMIL/           # TransMIL repository and adaptations
├── TCGA-BRCA-features/ # Pre-computed WSI features (HDF5)
├── splits/             # Generated cross-validation splits
├── master.csv          # Master manifest with clinical ground truth
├── generate_splits.py  # Script for generating CV splits
└── IMPROVED_PLAN.md    # Detailed research protocol
```

## 🛠️ Setup & Installation
1.  **Clone the Repository:**
    ```bash
    git clone [your-repo-url]
    cd thesis-project
    ```
2.  **Create Environment:**
    ```bash
    conda create -n mil_thesis python=3.8 -y
    conda activate mil_thesis
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install pandas scikit-learn h5py opencv-python-headless matplotlib seaborn pytorch-lightning lifelines scikit-survival
    ```

## 📊 Data Preparation
1.  **Generate Binary Labels:**
    Run `update_master_for_classification.py` to create the `label` column based on 5-year survival.
2.  **Generate Splits:**
    ```bash
    python generate_splits.py --data_dir master.csv --task task_2_tumor_subtyping --label_col label --k 5 --split_dir splits
    ```

## 🏋️ Training
### CLAM (Baseline)
```bash
python CLAM/main.py --data_root_dir "TCGA-BRCA-features/TCGA" --split_dir "splits" --task task_2_tumor_subtyping --exp_code brca_clam_fold_0 --model_type clam_sb --bag_loss focal --lr 2e-5 --k 5 --k_start 0 --k_end 1 --subtyping --bag_weight 0.9 --early_stopping
```

### TransMIL (Challenger)
```bash
python TransMIL/train.py --stage 'train' --config 'TransMIL/TCGA_BRCA.yaml' --gpus 0 --fold 0
```

## 📈 Evaluation
Evaluation scripts are located in `CLAM/eval.py` and TransMIL's inference mode. Results are compared using **AUC**, **C-Index**, and **DeLong's test** for statistical significance.

## 📜 Acknowledgments
- **CLAM:** [Mahmood Lab](https://github.com/mahmoodlab/CLAM)
- **TransMIL:** [TransMIL Repository](https://github.com/szc19990412/TransMIL)
- **Data:** TCGA-BRCA via GDC and Mahmood Lab UNI2-h features.
