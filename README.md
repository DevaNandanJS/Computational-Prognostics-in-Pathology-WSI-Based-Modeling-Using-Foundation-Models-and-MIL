Computational Prognostics in Pathology: WSI-Based Modeling Using Foundation Models and MIL
Author: Deva Nandan JS (VIT Bhopal University)
Domain: Computational Pathology (CPATH), Deep Learning, MLOps
Task: Patient Survival Prognosis (TCGA-BRCA Cohort)
📖 Executive Summary
This repository contains the codebase for a controlled, head-to-head comparative analysis of two dominant Multiple Instance Learning (MIL) architectures—CLAM (Clustering-constrained Attention MIL) and TransMIL (Transformer-based Correlated MIL)—for survival prognosis in breast cancer (TCGA-BRCA).
To eliminate feature representation variance and isolate the cognitive aggregation mechanism, this pipeline utilizes a standardized, frozen feature manifold extracted via the UNI2-h Pathology Foundation Model.
🏗️ Systems Architecture & Decoupled Pipeline
Processing gigapixel-scale Whole Slide Images (WSIs) end-to-end is computationally intractable due to severe GPU memory constraints. We mandate a two-stage decoupled pipeline:
Feature Extraction (Pre-processing): WSIs are tessellated into 256x256 patches. Each patch is forward-passed through the frozen UNI2-h ViT to generate a 1024-dimensional embedding. These are saved to disk as .h5 files.
Aggregation Modeling (MIL): The MIL networks ingest these lightweight .h5 sequence embeddings. This decoupling is a critical MLOps strategy that removes the multi-gigabyte memory bottleneck of the feature extractor from the iterative MIL training loop.
⚖️ Architectural Engineering Trade-offs
Choosing the optimal MIL architecture is a definitive engineering trade-off matrix: Contextual Accuracy vs. Computational Scalability.
1. TransMIL: Context-Aware Superiority (The Heavy Lifter)
TransMIL rejects the standard i.i.d. assumption, positing that survival prognosis relies on the spatial interrelationships of the Tumor Microenvironment (TME). It uses a Pyramid Position Encoding Generator (PPEG) and Multi-Head Self-Attention (MHSA).
Pros: Superior predictive engine. Higher C-Index and AUC. Captures diffuse, spatially correlated signals. Lower False Negative Rate (critical for clinical safety).
Cons: Quadratic-like memory scaling. Massive parameter bloat. Requires datacenter-grade hardware (e.g., A100/RTX 4090).
2. CLAM: i.i.d. Attention & Linear Scalability (The Scalable Workhorse)
CLAM operates under the i.i.d. assumption, using gated attention and an auxiliary instance-level clustering loss to linearly separate high/low-attention features.
Pros:  linear complexity. Extremely fast inference. Low VRAM footprint. Highly interpretable focal attention heatmaps. Perfect for resource-constrained edge deployments.
Cons: Lower baseline accuracy. Susceptible to high False Positive rates due to isolated noisy patches.
📊 Performance vs. Overhead Matrix
Metric / Parameter
CLAM (Attention + Clustering)
TransMIL (Self-Attention + PPEG)
C-Index (Survival)
0.655 ± 0.012
0.680 ± 0.011
AUC (5-Yr Binary)
0.615 ± 0.156
0.714 ± 0.140
Trainable Parameters
~2.8 Million
~12.5 Million
Peak VRAM (Training)
6.2 GB
18.4 GB
Inference Latency
45 ms / slide
180 ms / slide
Algorithmic Complexity

 (Nyström Approx)

Results based on 5-fold cross-validation on the TCGA-BRCA dataset using an NVIDIA RTX 4090 (24GB).
🚀 Production MLOps Deployment Blueprint
To maximize clinical yield while managing cloud billing, this architecture supports a Tiered Prognostic Pipeline:
Tier 1 (Edge Screening): CLAM is deployed locally on hospital IT infrastructure (e.g., NVIDIA T4). It processes slides with near-zero latency (45ms), acting as a rapid risk-stratification and triage tool.
Tier 2 (Cloud Compute): Complex or borderline cases are routed to AWS/GCP where TransMIL executes heavy contextual mapping for precise adjuvant therapy planning.
📂 Repository Structure
├── datasets/                   # Dataset loaders and split manifests (CSV)
│   ├── master.csv              # Unified metadata (patient ID, WSI paths, labels)
│   └── splits/                 # 5-fold cross-validation split manifests
├── models/                     # Network architectures
│   ├── clam.py                 # CLAM architecture implementation
│   ├── transmil.py             # TransMIL architecture (w/ Nyström approximation)
│   └── layers/                 # Custom attention and PPEG modules
├── utils/                      # Helper scripts
│   ├── core_utils.py           # Training loops, loss functions (Cox, CrossEntropy)
│   ├── eval_utils.py           # Lifelines C-Index and Scikit-Learn AUC calculations
│   └── file_utils.py           # .h5 feature loading mechanisms
├── train_cv.py                 # Main 5-fold cross-validation training script
├── evaluate.py                 # Standalone inference and evaluation script
├── extract_features.py         # (Optional) Pre-processing script for UNI2-h
├── requirements.txt            # Python dependencies
└── README.md                   # This file


⚙️ Installation & Setup
Clone the repository:
git clone [https://github.com/DevaNandanJS/Computational-Prognostics-in-Pathology-WSI-Based-Modeling-Using-Foundation-Models-and-MIL.git](https://github.com/DevaNandanJS/Computational-Prognostics-in-Pathology-WSI-Based-Modeling-Using-Foundation-Models-and-MIL.git)
cd Computational-Prognostics-in-Pathology-WSI-Based-Modeling-Using-Foundation-Models-and-MIL


Set up the virtual environment:
We recommend using conda for isolated dependencies to ensure PyTorch and CUDA bindings are correct.
conda create -n cpath_env python=3.10
conda activate cpath_env
pip install -r requirements.txt


Data Preparation:
Place the extracted .h5 UNI2-h feature files in a designated data directory.
Ensure master.csv (containing OS.time, OS event status, and survival_label_5yr) is mapped correctly to your .h5 paths.
💻 Running the Pipeline
1. Training (5-Fold Cross Validation)
To train CLAM:
python train_cv.py --model clam --data_root /path/to/h5_features --csv_path datasets/master.csv --task survival --epochs 50 --lr 2e-4


To train TransMIL:
python train_cv.py --model transmil --data_root /path/to/h5_features --csv_path datasets/master.csv --task survival --epochs 50 --lr 2e-4


Note: Ensure you have at least 20GB of VRAM available to comfortably train TransMIL with large sequence lengths.
2. Evaluation
To evaluate a trained model on a specific hold-out set and generate ROC curves / Confusion Matrices:
python evaluate.py --model transmil --weights checkpoints/transmil_fold0.pth --task survival_5yr --split test


🔬 Interpretability (XAI)
The codebase includes functionality to extract attention scores.
CLAM: Generates highly localized, focal heatmaps directly mapping to extreme nuclear pleomorphism.
TransMIL: Generates diffuse attention maps highlighting the macroscopic interface between tumor nests and stroma.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🤝 Acknowledgements
Mahmood Lab (Harvard) for providing gated access to the incredible UNI / UNI2-h Pathology Foundation Model.
TCGA for providing the comprehensive BRCA cohort dataset.
