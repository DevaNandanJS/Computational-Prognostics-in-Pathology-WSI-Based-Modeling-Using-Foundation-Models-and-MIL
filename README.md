# Computational Prognostics in Pathology: A WSI Analysis Using Foundation Models and Multiple Instance Learning

**Author:** Deva Nandan JS, Dr Velmani R
**Institution:** School of Computer Science and Artificial Intelligence, VIT Bhopal University  
**Supervisor:** Dr. Velmani R  

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900.svg)
![Task](https://img.shields.io/badge/Task-Survival_Analysis-success)

## 📌 Executive Summary
This repository contains the codebase for computational prognostics in pathology, addressing the "Gigapixel Challenge" where Whole Slide Images (WSIs) exceed 10 billion pixels at 40× magnification. Direct processing of such massive data structures is computationally prohibitive. To bypass the manual annotation bottleneck, this project employs a two-stage weakly-supervised Multiple Instance Learning (MIL) pipeline.

Our research strictly evaluates two aggregation paradigms—**CLAM** (Clustering-constrained Attention MIL) and **TransMIL** (Transformer-based Correlated MIL)—using a standardized foundational feature extractor (UNI2-h) to predict Overall Survival (OS) in the TCGA-BRCA cohort.

## 🏗️ Architectural Decisions & Trade-offs

As a core engineering principle, we decoupled feature extraction from aggregation to isolate performance metrics and ensure modular maintainability.

### 1. The Feature Extractor: UNI2-h Foundation Model
* **Decision:** Utilize pre-computed 1024-dimensional embeddings from the UNI2-h model.
* **Justification:** Eliminates the confounding variance introduced by weaker CNN extractors trained from scratch, ensuring that any downstream performance difference is strictly attributable to the MIL aggregation architecture.

### 2. Aggregation Modeling: CLAM vs. TransMIL
We evaluated two distinct paradigms for pooling instance-level embeddings into a slide-level diagnostic prediction:

* **CLAM (Focal Attention):** Treats diagnosis as a search problem, utilizing instance-level clustering loss to identify a small subset of highly informative, independent patches.
* **TransMIL (Diffuse Attention):** Treats diagnosis as a contextual problem, leveraging multi-head self-attention and PPEG to model complex spatial interrelationships within the Tumor Microenvironment (TME).

### 3. Systems Engineering Decision Matrix
The choice of model hinges on strict computational vs. accuracy trade-offs:

| Parameter | CLAM | TransMIL | Ratio / Trade-off |
| :--- | :--- | :--- | :--- |
| **Algorithmic Complexity** | O(N) Linear | O(N log N) Super-linear | CLAM scales effortlessly for massive slides. |
| **Inference Time / Slide** | 45 ms | 180 ms | TransMIL introduces a 4× latency penalty. |
| **Peak VRAM (Training)** | 6.2 GB | 18.4 GB | TransMIL requires 3× the memory footprint. |
| **Trainable Parameters** | ~2.8 Million | ~12.5 Million | TransMIL is 4.5× larger. |

## 🚀 Proposed Tiered Prognostic Deployment
Relying entirely on a Transformer-based architecture causes GPU exhaustion and massive cloud billing, while relying solely on CLAM sacrifices prognostic accuracy on complex cases. We propose a tiered pipeline to maximize scalability, maintainability, and clinical safety:

* **Tier 1 (Edge Screening):** Deploy CLAM on local hospital servers/mid-tier GPUs (e.g., NVIDIA T4) for high-throughput daily patient backlog clearing (45ms/slide).
* **Tier 2 (Cloud Analysis):** Deploy TransMIL on cloud instances (e.g., A100 GPUs) triggered only when CLAM flags a borderline case or an oncologist requests deeper spatial analysis for adjuvant therapy planning (+0.025 C-Index boost).

## 📊 Performance Results

All models were evaluated using strict 5-Fold Cross-Validation to ensure robust performance assessment against data partitioning variance.

### Primary Task: Overall Survival (Risk Ranking)
* **TransMIL:** 0.680 ± 0.011 C-Index
* **CLAM:** 0.655 ± 0.012 C-Index
* *TransMIL yields a statistically significant +0.025 advantage in accurately ranking patient risk pairs.*

### Secondary Task: 5-Year Binary Survival (Classification)

| Metric | CLAM | TransMIL | Δ |
| :--- | :--- | :--- | :--- |
| **AUC** | 0.615 | 0.714 | +0.099 |
| **Accuracy** | 0.573 | 0.690 | +0.117 |
| **F1-Score** | 0.189 | 0.239 | +0.050 |
| **Specificity** | 0.555 | 0.697 | +0.142 |

**Robustness Note:** CLAM exhibited high volatility across folds (AUC Standard Deviation: ± 0.156) and a False Negative Rate of 33.3%, largely due to the i.i.d. assumption limiting robustness against heterogeneous data. TransMIL maintained superior stability (AUC Standard Deviation: ± 0.140) by utilizing its self-attention mechanism to buffer localized noise against the global slide context.

## 💻 Hardware & Software Requirements

* **GPU:** NVIDIA GeForce RTX 4090 (24 GB GDDR6X VRAM ceiling stress-tests TransMIL's memory spikes while avoiding multi-GPU sharding complexity).
* **CPU/Memory:** Intel Core i9-13900K (24 cores), 128 GB DDR5 RAM.
* **Storage:** 2TB NVMe M.2 SSD (Gen 4) for high-throughput `.h5` embedding loading.
* **Environment:** PyTorch 2.0.1, Python 3.10, `lifelines`, `scikit-learn`.

## 📂 Expected Repository Structure
```text
.
├── data/
│   ├── tcga_brca/              # TCGA-BRCA cohort metadata
│   └── embeddings/             # Pre-extracted UNI2-h .h5 embeddings
├── models/
│   ├── clam.py                 # Clustering-constrained Attention MIL implementation
│   ├── transmil.py             # Transformer-based Correlated MIL implementation
│   └── modules/                # Reusable attention mechanisms and PPEG layers
├── core/
│   ├── engine.py               # Robust training/validation loops
│   ├── dataset.py              # Efficient PyTorch dataloaders for variable bag sizes
│   └── metrics.py              # C-Index, AUC, F1, and Confusion Matrix utilities
├── notebooks/
│   └── attention_vis.ipynb     # Visualization of focal (CLAM) vs. diffuse (TransMIL) attention
├── train.py                    # Entry point for model training
├── evaluate.py                 # 5-Fold Cross-Validation evaluation script
├── requirements.txt
└── README.md
