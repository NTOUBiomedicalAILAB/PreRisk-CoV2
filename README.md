![PreRisk-CoV2](logo.png)

## Overview

PreRisk-CoV2 is a machine learning framework for pre-exposure risk assessment of SARS-CoV-2 susceptibility using Serum protein biomarkers. The main function is to predict infection risk **before exposure** based on a 7-protein panel identified through K-Nearest Neighbors (KNN) combined with Genetic Algorithm (GA) feature selection. The input consists of protein expression data (CSV format), and the output provides risk prediction results with comprehensive performance metrics.

📄 **Paper**: Development and External Validation of a Pre-Exposure Protein Biomarker Panel and Machine Learning Model for Predicting SARS-CoV-2 Infection Risk

If you have any trouble installing or using PreRisk-CoV2, you can post an issue or directly email us. We welcome any suggestions.

---

## Quick Install

*Note*: We suggest you install all packages using conda ([Anaconda](https://anaconda.org/)).

### Prepare the Environment

#### 1. First-time Setup

```bash
# Create conda environment with required dependencies
conda create -n PreRisk_CoV2 python=3.9 -y
conda activate PreRisk_CoV2

# Install core packages
pip install numpy pandas scikit-learn matplotlib openpyxl imbalanced-learn tenseal

# Download PreRisk-CoV2 scripts
git clone https://github.com/NTOUBiomedicalAILAB/PreRisk-CoV2.git
cd PreRisk-CoV2/

# Quick test 
python prerisk_cov2.py --train-input Discovery.csv --test-input Validation.csv --protein-indices 3 40 50 36 83 25 63 --n-neighbors 5 --weights distance --algorithm kd_tree --use-smote --n-iterations 100 --plot-curves --verbose --output-dir ./results

```

#### 2. Subsequent Usage

If the runs without errors, you only need to activate your environment before using PreRisk-CoV2:

```bash
conda activate PreRisk_CoV2
cd PreRisk-CoV2/
```

---

## Usage

### External Validation

**Full Example**:

```bash
python prerisk_cov2.py ^
    --train-input Discovery.csv ^
    --test-input Validation.csv ^
    --protein-indices 3 40 50 36 83 25 63 ^
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
python prerisk_cov2.py --train-input Discovery.csv --test-input Validation.csv --protein-indices 3 40 50 36 83 25 63 --n-neighbors 5 --weights distance --algorithm kd_tree --use-smote --n-iterations 100 --plot-curves --verbose --output-dir ./results
```



**Key Parameters** (command-line arguments):

- `--train-input`: Training dataset CSV 
- `--test-input`: Independent validation dataset CSV 
- `--n-iterations`: Number of validation iterations (default: 100)
- `--use-smote`: Enable SMOTE oversampling
- `--protein-indices`: Selected protein biomarker indices (default: [3, 50, 40, 36, 83])
- `--output-dir`: Output directory (default: ./results)
- `--plot-curves`: Generate and save ROC / PR curves
- `--verbose`: Print per-iteration metrics to console

**KNN Hyperparameters**:

- `--n-neighbors`: Number of neighbors (default: 5)
- `--leaf-size`: Leaf size for tree algorithms (default: 30)
- `--algorithm`: Algorithm type (auto, ball_tree, kd_tree, brute)
- `--weights`: Weight function (uniform, distance)
- `--p`: Power parameter for Minkowski metric (default: 2)





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

**Output files:**
```
results/
├── external_validation_[timestamp].xlsx
└── external_roc_pr.png
```


---


## 📊 Data Availability

### Public Datasets

De-identified individual participant data supporting the findings of this study are available in the Gene Expression Omnibus (GEO) (https://www.ncbi.nlm.nih.gov/geo) under accession num-bers **GSE198449 (CHARM cohort)** and **GSE178967 (CEIM cohort)**.

1. NPX data for the **CHARM cohort** are accessible via the supplementary material of Soares-Schanoski et al. https://pmc.ncbi.nlm.nih.gov/articles/PMC9037090
2. NPX data for the **CEIM cohort** are available at the following repository: [https://github.com/hzc363/COVID19_system_immunology/blob/master/OLINK\%20Proteomics/olink.csv](https://github.com/hzc363/COVID19_system_immunology/blob/master/OLINK%20Proteomics/olink.csv)






---



