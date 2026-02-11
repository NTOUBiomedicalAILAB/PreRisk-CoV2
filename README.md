![PreRisk-CoV2](logo.png)

## Overview

PreRisk-CoV2 is a machine learning framework for pre-exposure risk assessment of SARS-CoV-2 susceptibility using Serum protein biomarkers. The main function is to predict infection risk **before exposure** based on a 5-protein panel identified through K-Nearest Neighbors (KNN) combined with Genetic Algorithm (GA) feature selection. The input consists of protein expression data (CSV format), and the output provides risk prediction results with comprehensive performance metrics.

📄 **Paper**: Predicting SARS-CoV-2 Susceptibility from Pre-Infection Serum Proteins: A Machine Learning Approach

If you have any trouble installing or using PreRisk-CoV2, you can post an issue or directly email us. We welcome any suggestions.

---

## Quick Install

*Note*: We suggest you install all packages using conda (both miniconda and [Anaconda](https://anaconda.org/) are ok).

### Prepare the Environment

#### 1. First-time Setup

```bash
# Create conda environment with required dependencies
conda create -n prerisk python=3.9 -y
conda activate prerisk

# Install core packages
conda install numpy pandas scikit-learn matplotlib openpyxl -c conda-forge -y
pip install imbalanced-learn

# Download PreRisk-CoV2 scripts
git clone https://github.com/NTOUBiomedicalAILAB/PreRisk-CoV2.git
cd PreRisk-CoV2/

# Quick test (internal validation, 10 iterations)
python prerisk_cov2.py --mode internal --input Discovery.csv --n-iterations 10 --verbose

```

#### 2. Subsequent Usage

If the runs without errors, you only need to activate your environment before using PreRisk-CoV2:

```bash
conda activate prerisk
```

---

## Usage

### Internal Validation (LOOCV)

```bash
python prerisk_cov2.py ^
    --mode internal ^
    --input Discovery.csv ^
    --n-iterations 100 ^
    --verbose
```

**Key Parameters** (command-line arguments):

- `--input`: Input CSV file path (required)
- `--n-iterations`: Number of cross-validation iterations (default: 100)
- `--use-smote`: Enable SMOTE oversampling for class imbalance
- `--protein-indices`: Protein indices for the 5-protein panel (default: [3, 50, 40, 36, 83])
- `--output-dir`: Output directory for results (default: ./results)
- `--verbose`: Display detailed progress

**Configuration Options**:

- `--n-neighbors`: Number of neighbors for KNN (default: 5)
- `--weights`: Weight function (uniform or distance, default: uniform)
- `--algorithm`: Algorithm type (auto, ball_tree, kd_tree, brute)
- `--plot-curves`: Generate PR and ROC curves
- `--p`: Power parameter for Minkowski metric (default: 2)

**Full Example**:

```cmd
python prerisk_cov2.py ^
    --mode internal ^
    --input Discovery.csv ^
    --protein-indices 3 50 40 36 83 ^
    --n-neighbors 5 ^
    --weights distance ^
    --algorithm kd_tree ^
    --use-smote ^
    --n-iterations 100 ^
    --plot-curves ^
    --verbose ^
    --output-dir ./results
```

**Or use single-line:**


```bash
python prerisk_cov2.py --mode internal --input Discovery.csv --protein-indices 3 50 40 36 83 --n-neighbors 5 --weights distance --algorithm kd_tree --use-smote --n-iterations 100 --plot-curves --verbose --output-dir ./results
```

---

### External Validation

```cmd
python prerisk_cov2.py ^
    --mode external ^
    --train-input Discovery.csv ^
    --test-input Validation.csv ^
    --protein-indices 3 50 40 36 83 ^
    --n-neighbors 5 ^
    --weights distance ^
    --algorithm kd_tree ^
    --use-smote ^
    --n-iterations 100 ^
    --plot-curves ^
    --verbose ^
    --output-dir ./results
```

**Or use single-line:**

```bash
python prerisk_cov2.py --mode external --train-input Discovery.csv --test-input Validation.csv --protein-indices 3 50 40 36 83 --n-neighbors 5 --weights distance --algorithm kd_tree --use-smote --n-iterations 100 --plot-curves --verbose --output-dir ./results
```



**Key Parameters** (command-line arguments):

- `--train-input`: Training dataset CSV (required for external mode)
- `--test-input`: Independent validation dataset CSV (required for external mode)
- `--n-iterations`: Number of validation iterations (default: 100)
- `--use-smote`: Enable SMOTE oversampling
- `--protein-indices`: Selected protein biomarker indices (default: [3, 50, 40, 36, 83])
- `--output-dir`: Output directory (default: ./results)

**KNN Hyperparameters**:

- `--n-neighbors`: Number of neighbors (default: 5)
- `--leaf-size`: Leaf size for tree algorithms (default: 30)
- `--algorithm`: Algorithm type (auto, ball_tree, kd_tree, brute)
- `--weights`: Weight function (uniform, distance)
- `--p`: Power parameter for Minkowski metric (default: 2)

**Full Example**:

```bash
python prerisk_cov2.py \
    --mode external \
    --train-input Discovery.csv \
    --test-input Validation.csv \
    --protein-indices 3 50 40 36 83 \
    --n-neighbors 5 \
    --weights distance \
    --use-smote \
    --n-iterations 100 \
    --plot-curves \
    --verbose \
    --output-dir ./results
```



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

**Example CSV:**

```csv
sample ID,PCR result,Protein_1,Protein_2,...,Protein_92
Sample001,Detected,0.45,0.32,...,0.78
Sample002,Not,0.21,0.67,...,0.43
Sample003,Detected,0.89,0.54,...,0.66
```

---

## Outputs

### Internal Validation Output Structure

```
results/
└── internal_validation_[timestamp].xlsx
    ├── Sheet: LOOCV_Results
    │   ├── Per-iteration metrics (Accuracy, Sensitivity, Specificity, etc.)
    │   ├── Average performance (mean ± std)
    │   └── Detailed per-fold results
```

### External Validation Output Structure

```
results/
└── external_validation_[timestamp].xlsx
    ├── Sheet: External_Results
    │   ├── Average metrics across 100 iterations
    │   ├── Standard deviations
    │   └── Per-iteration detailed results
```

### Performance Metrics

Both validation modes output the following metrics:

- **Accuracy**: Overall classification accuracy
- **Sensitivity (Recall)**: True Positive Rate (TPR)
- **Specificity**: True Negative Rate (TNR)
- **Precision**: Positive Predictive Value (PPV)
- **F1-Score**: Harmonic mean of precision and recall
- **AUROC**: Area Under Receiver Operating Characteristic curve
- **AUPRC**: Area Under Precision-Recall Curve
- **MCC**: Matthews Correlation Coefficient

### Optional Visualizations

When `--plot-curves` is enabled:

- ROC curve with AUROC annotation
- Precision-Recall curve with AUPRC annotation
- Individual sample probability predictions

**Output files:**
```
results/
├── internal_validation_[timestamp].xlsx
├── external_validation_[timestamp].xlsx
├── internal_roc_pr.png
└── external_roc_pr.png
```


---


## 📊 Data Availability

### Public Datasets

All data supporting this study are publicly available in the Gene Expression Omnibus (https://www.ncbi.nlm.nih.gov/geo) under accession numbers **GSE198449** and **GSE178967**.

### Data Privacy

🔒 **Note**: The actual patient-level dataset files are not included in this repository due to privacy regulations and ethical considerations. Researchers can access the data through GEO with appropriate institutional review.

### Accessing the Data

1. Visit NCBI GEO: https://www.ncbi.nlm.nih.gov/geo
2. Search for accession numbers: GSE198449, GSE178967
3. Download the series matrix files
4. Convert to the required CSV format (see Input Data Format section)

---



