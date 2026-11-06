# LungXplain

**Explainable In-Hospital Mortality Prediction from Chest X-rays using Energy-Based Concept Bottleneck Models (ECBM)**

Graduation Thesis (KLTN) — Ho Chi Minh City University of Technology (HCMUT), VNU-HCM.

---

## 📌 Overview

LungXplain addresses the critical "black-box" dilemma in clinical AI by predicting patient in-hospital mortality (`hospital_expire_flag` from MIMIC-IV) from chest X-ray radiographs (MIMIC-CXR) through an interpretable intermediate layer of **14 standard CheXpert clinical concepts**.

### Key Architectures
1. **Black-box Classifier:** Direct image-to-mortality baseline using `EfficientNet-B0`.
2. **Soft CBM (`SoftJointCBM`):** Joint continuous probability concept bottleneck.
3. **Hard CBM (`HardJointCBM`):** Discrete categorical one-hot concept bottleneck with Straight-Through gradients.
4. **Energy-Based CBM (`CategoricalECBM`):** Unified joint energy function $E(X, C, Y) = E_{img}(X, C) + E_{concept}(C, Y) + E_{global}(X, C, Y)$ over $2^{14} = 16,384$ configurations with exact enumeration inference and test-time concept clamping/intervention.

---

## 📂 Project Structure

```text
├── src/
│   ├── dataset.py            # PyTorch Dataset for MIMIC-CXR + 14 CheXpert concepts + mortality
│   ├── models.py             # Unified model factories
│   ├── cbm.py                # Soft & Hard CBM implementations
│   ├── ecbm.py               # Categorical ECBM with exact enumeration & clamping
│   ├── concept_loss.py       # State-weighted concept loss
│   ├── intervention.py       # Sequential concept intervention (m = 0..14)
│   └── metrics.py            # Macro-F1, Balanced Accuracy, threshold tuning
├── experiments/
│   ├── prepare_mimic_mortality_cohort.py  # Cohort creation linking MIMIC-CXR & MIMIC-IV
│   ├── run_blackbox_mimic.py             # Train / evaluate Black-box
│   ├── run_cbm_mimic.py                  # Train / evaluate Soft & Hard CBM
│   ├── run_ecbm_mimic.py                 # Train / evaluate Categorical ECBM
│   └── run_intervention_mimic.py         # Evaluate test-time concept intervention curves
├── requirements.txt
└── PLAN.md                               # Detailed 24-week thesis plan
```

---

## 🚀 Running on Kaggle GPU

### 1. Kaggle Dataset Preparation
Create a Kaggle Dataset containing:
- `files/`: 5,000 `.jpg` chest radiographs.
- `manifest_mortality.csv`: Patient-disjoint split (Train: 4,000, Valid: 500, Test: 500).

### 2. In Kaggle Notebook (GPU T4 / P100)
```bash
# Clone the repository
!git clone https://github.com/LeNhatHuy219/LungXplain.git
%cd LungXplain

# Install requirements if needed
!pip install -r requirements.txt
```

### 3. Run Experiments

```bash
# Set path to your mounted Kaggle dataset
DATA_ROOT="/kaggle/input/mimic-cxr-mortality-5k"
MANIFEST="$DATA_ROOT/manifest_mortality.csv"

# 1. Train Soft CBM (30 epochs, Seed 42)
python experiments/run_cbm_mimic.py \
    --model-type soft \
    --manifest-path $MANIFEST \
    --data-root $DATA_ROOT \
    --epochs 30 \
    --seed 42 \
    --output-dir results/mimic_cxr/soft/seed42_e30

# 2. Train Hard CBM (30 epochs, Seed 42)
python experiments/run_cbm_mimic.py \
    --model-type hard \
    --manifest-path $MANIFEST \
    --data-root $DATA_ROOT \
    --epochs 30 \
    --seed 42 \
    --output-dir results/mimic_cxr/hard/seed42_e30

# 3. Train ECBM (30 epochs, Seed 42)
python experiments/run_ecbm_mimic.py \
    --manifest-path $MANIFEST \
    --data-root $DATA_ROOT \
    --epochs 30 \
    --seed 42 \
    --output-dir results/mimic_cxr/ecbm/seed42_e30

# 4. Evaluate Concept Intervention (m = 0..14)
python experiments/run_intervention_mimic.py \
    --checkpoint results/mimic_cxr/ecbm/seed42_e30/best.pt \
    --manifest-path $MANIFEST \
    --data-root $DATA_ROOT \
    --strategy uncertainty \
    --output-dir results/mimic_cxr/ecbm/seed42_e30/intervention
```

---

## 📊 Evaluation Protocols
- **Primary Metric:** Macro-F1 on In-Hospital Mortality.
- **Secondary Metrics:** Balanced Accuracy, ROC-AUC, PR-AUC.
- **Concept Quality:** Mean Concept Macro-F1 across 14 CheXpert labels, Exact-Match Ratio.
- **Intervention Quality:** $\text{AUC-IC}$ (Area Under Intervention Curve), $\Delta\text{Macro-F1}$.
- **Threshold Calibration:** Optimal decision threshold $T^* \in [0.05, 0.95]$ selected on Validation Balanced Accuracy.
