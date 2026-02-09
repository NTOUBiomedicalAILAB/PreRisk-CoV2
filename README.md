# PreRisk-CoV2

## Overview

PreRisk-CoV2 is a machine learning framework for pre-exposure risk assessment of SARS-CoV-2 susceptibility using plasma protein biomarkers. The main function is to predict infection risk **before exposure** based on a 5-protein panel identified through K-Nearest Neighbors (KNN) combined with Genetic Algorithm (GA) feature selection. The input consists of protein expression data (CSV format), and the output provides risk prediction results with comprehensive performance metrics.



If you have any trouble installing or using PreRisk-CoV2, you can post an issue or directly email us. We welcome any suggestions.

---

## Quick Install

*Note*: We suggest you install all packages using conda (both miniconda and [Anaconda](https://anaconda.org/) are ok).

### Prepare the Environment

#### 1. First-time Setup

```bash
# Create conda environment with required dependencies
conda create -n prerisk python=3.8 -y
conda activate prerisk

# Install core packages
conda install numpy pandas scikit-learn matplotlib openpyxl -c conda-forge -y
pip install keras tensorflow imbalanced-learn xgboost

# Download PreRisk-CoV2 scripts
git clone https://github.com/NTOUBiomedicalAILAB/PreRisk-CoV2.git
cd PreRisk-CoV2/

# Run an example (internal validation)
python Internal-validation.py

# Run external validation example
python External-validation.py
```

#### 2. Subsequent Usage

If the example runs without errors, you only need to activate your environment before using PreRisk-CoV2:

```bash
conda activate prerisk
```

---

## Usage

### Internal Validation (LOOCV)

```bash
python Internal-validation.py
```

**Key Parameters** (modifiable in script):

- `file`: Input CSV file path (default: `Discovery.csv`)
- `loop`: Number of cross-validation iterations (default: 100)
- `smote`: SMOTE oversampling (0: disabled, 1: enabled)
- `ID_index`: Protein indices for the 5-protein panel [3, 50, 40, 36, 83]
- `excel_location`: Output directory for results (default: `\輸出結果`)
- `save_new`: Enable/disable Excel output (0: disabled, 1: enabled)

**Configuration Options:**

- `train_process`: Display training process details (0/1)
- `loop_result`: Show each cross-validation iteration (0/1)
- `pr_roc`: Generate PR and ROC curves (0/1)
- `random_protein`: Number of proteins for random selection (default: 7)

### External Validation

```bash
python External-validation.py
```

**Key Parameters** (modifiable in script):

- `train_input`: Training dataset CSV (default: `Discovery.csv`)
- `test_input`: Independent validation dataset CSV (default: `Validation.csv`)
- `loop`: Number of validation iterations (default: 100)
- `smote`: SMOTE oversampling (1: enabled)
- `accuracy_indice`: Selected protein biomarker indices [3, 50, 40, 36, 83]
- `excel_location`: Output directory (default: `輸出結果\\`)
- `save`: Enable Excel output (0/1)

**KNN Hyperparameters:**

- `n_neighbors`: Number of neighbors (default: 5 when set to 0)
- `leaf_size`: Leaf size for tree algorithms
- `algorithm`: Algorithm type (1: auto, 2: ball_tree, 3: kd_tree, 4: brute)
- `weights`: Weight function (1: uniform, 2: distance)
- `p`: Power parameter for Minkowski metric

---

## Input Data Format

### CSV File Structure

- **Column 1**: `sample ID` - Unique sample identifier
- **Column 2**: `PCR result` - Ground truth label (`'Detected'` or `'Not'`)
- **Columns 3-94**: 92 protein expression levels (normalized values)

**Label Encoding:**
- `'Detected'` → 1 (SARS-CoV-2 positive)
- `'Not'` → 0 (SARS-CoV-2 negative)

**Preprocessing:**
- MinMax normalization (0-1 range)
- Missing value handling via `Missing_Counts()` function
- Optional SMOTE oversampling for class imbalance

---

## Outputs

### Internal Validation Output Structure

```
輸出結果/
└── 內部驗證_[timestamp].xlsx
    ├── Sheet: 編號 1
    │   ├── Per-iteration metrics (Accuracy, Sensitivity, Specificity, etc.)
    │   ├── Average performance (mean ± std)
    │   └── Detailed per-fold results
```

### External Validation Output Structure

```
輸出結果/
└── 外部驗證結果_[timestamp].xlsx
    ├── Sheet: 編號 1
    │   ├── Average metrics across 100 iterations
    │   ├── Standard deviations
    │   └── Per-iteration detailed results
```

### Performance Metrics

Both validation scripts output the following metrics:

- **Accuracy**: Overall classification accuracy
- **Sensitivity (Recall)**: True Positive Rate (TPR)
- **Specificity**: True Negative Rate (TNR)
- **Precision**: Positive Predictive Value (PPV)
- **F1-Score**: Harmonic mean of precision and recall
- **AUROC**: Area Under Receiver Operating Characteristic curve
- **AUPRC**: Area Under Precision-Recall Curve
- **MCC**: Matthews Correlation Coefficient

### Optional Visualizations

When `pr_roc = 1`:
- ROC curve with AUROC annotation
- Precision-Recall curve with AUPRC annotation
- Individual sample probability predictions

---

## Algorithm Details

### Feature Selection Strategy

The framework employs a **hybrid KNN-GA approach**:

1. **Genetic Algorithm (GA)**: Optimizes protein subset selection
2. **K-Nearest Neighbors (KNN)**: Classifier with distance-weighted voting
3. **5-Protein Panel**: Final biomarker signature (indices: 3, 50, 40, 36, 83)

### Validation Strategy

- **Internal Validation**: LOOCV (Leave-One-Out Cross-Validation)
  - Each sample serves as test set once
  - Remaining samples for training
  - 100 independent LOOCV runs for robustness

- **External Validation**: Independent cohort testing
  - Separate discovery and validation datasets
  - Model trained on `Discovery.csv`
  - Tested on `Validation.csv` (unseen data)
  - 100 iterations with consistent protein panel

### Class Imbalance Handling

- **SMOTE** (Synthetic Minority Over-sampling Technique)
- `k_neighbors=5` for synthetic sample generation
- Applied only on training folds to prevent data leakage

---


## Example Workflow

### Step 1: Prepare Data

Ensure your CSV files follow the required format:
```csv
sample ID,PCR result,Protein_1,Protein_2,...,Protein_92
Sample001,Detected,0.45,0.32,...,0.78
Sample002,Not,0.21,0.67,...,0.43
```

### Step 2: Run Internal Validation

```bash
conda activate prerisk
python Internal-validation.py
```



### Step 3: Run External Validation

```bash
python External-validation.py
```

Check output Excel file in `輸出結果/` directory.

### Step 4: Visualize Results (Optional)

Set `pr_roc = 1` in script to generate:
- ROC and PR curves
- Probability distributions
- Confusion matrices

---



