# PreRisk-CoV2

## Overview

PreRisk-CoV2 is a machine learning framework for pre-exposure risk assessment of SARS-CoV-2 susceptibility using plasma protein biomarkers. The main function is to predict infection risk **before exposure** based on a 5-protein panel identified through K-Nearest Neighbors (KNN) combined with Genetic Algorithm (GA) feature selection. The input consists of protein expression data (CSV format), and the output provides risk prediction results with comprehensive performance metrics.

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

# Run an example (internal validation)
python prerisk_cov2.py --mode internal --input Discovery.csv --n-iterations 10

# Run external validation example
python prerisk_cov2.py --mode external --train-input Discovery.csv --test-input Validation.csv --n-iterations 10
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
python prerisk_cov2.py \
    --mode internal \
    --input Discovery.csv \
    --n-iterations 100 \
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

```bash
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

### External Validation

```bash
python prerisk_cov2.py \
    --mode external \
    --train-input Discovery.csv \
    --test-input Validation.csv \
    --n-iterations 100 \
    --verbose
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

### Quick Test

```bash
# Display help message
python prerisk_cov2.py --help

# Run system compatibility check
chmod +x quick_test.sh
./quick_test.sh

# Run example scripts (Linux/Mac)
chmod +x example_internal_validation.sh
./example_internal_validation.sh

# Run example scripts (Windows)
example_run_windows.bat
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

## Algorithm Details

### Feature Selection Strategy

The framework employs a **hybrid KNN-GA approach**:

1. **Genetic Algorithm (GA)**: Optimizes protein subset selection
2. **K-Nearest Neighbors (KNN)**: Classifier with distance-weighted voting
3. **5-Protein Panel**: Final biomarker signature (indices: 3, 50, 40, 36, 83)

### Validation Strategy

**Internal Validation: LOOCV (Leave-One-Out Cross-Validation)**
- Each sample serves as test set once
- Remaining samples for training
- 100 independent LOOCV runs for robustness

**External Validation: Independent cohort testing**
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
python prerisk_cov2.py --mode internal --input Discovery.csv --n-iterations 100 --verbose
```

**Monitor output:**
```
[INFO] Dataset shape: (77, 92)
[INFO] Class distribution: [20 57]
[INFO] Selected proteins: ['MCP-3', 'TRANCE', 'LIF-R', 'FGF-23', 'NT-3']
[INFO] Running 100 iterations of LOOCV...
  Iter 1/100: Acc=0.857, AUROC=0.881, AUPRC=0.941
  ...
----------------------------------------------------------------------
SUMMARY STATISTICS
----------------------------------------------------------------------
Accuracy       :  85.67 ±  2.34 %
Sensitivity    :  82.45 ±  3.12 %
AUROC          : 0.9145 ± 0.0234
```

### Step 3: Run External Validation

```bash
python prerisk_cov2.py --mode external --train-input Discovery.csv --test-input Validation.csv --n-iterations 100 --verbose
```

Check output Excel file in `results/` directory.

### Step 4: Visualize Results (Optional)

Add `--plot-curves` flag to generate:
- ROC and PR curves
- Probability distributions
- Confusion matrices

```bash
python prerisk_cov2.py --mode internal --input Discovery.csv --plot-curves --n-iterations 10
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

## Dependencies

### Core Requirements

- Python ≥ 3.8
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- OpenPyXL
- imbalanced-learn (SMOTE)

### Installation

```bash
# Using conda (recommended)
conda create -n prerisk python=3.9 -y
conda activate prerisk
conda install numpy pandas scikit-learn matplotlib openpyxl -c conda-forge -y
pip install imbalanced-learn

# Or using pip
pip install numpy pandas scikit-learn matplotlib openpyxl imbalanced-learn
```

---

## Troubleshooting

### Common Issues

**1. CSV Format Error**
```
Error: KeyError 'sample ID' or 'PCR result'
```
- Ensure column names match exactly (case-sensitive)
- First column must be `sample ID`
- Second column must be `PCR result`
- Values must be `'Detected'` or `'Not'`

**2. SMOTE Error**
```
ValueError: Expected n_neighbors <= n_samples
```
- Minority class has too few samples
- Remove `--use-smote` flag or reduce training set size

**3. ModuleNotFoundError**
```bash
pip install imbalanced-learn openpyxl
```

**4. Windows Command Line Issues**
- Use single-line commands or batch files (`.bat`)
- See `example_run_windows.bat` for Windows examples

**5. Getting Help**
```bash
python prerisk_cov2.py --help
```

---

## Citation

If you use PreRisk-CoV2 in your research, please cite:

**Paper**: Predicting SARS-CoV-2 Susceptibility from Pre-Infection Serum Proteins: A Machine Learning Approach

```bibtex
@article{prerisk_cov2_2026,
  title={Predicting SARS-CoV-2 Susceptibility from Pre-Infection Serum Proteins: A Machine Learning Approach},
  author={Your Name et al.},
  journal={Journal Name},
  year={2026},
  url={https://github.com/NTOUBiomedicalAILAB/PreRisk-CoV2}
}
```

**GitHub Repository**:
```
https://github.com/NTOUBiomedicalAILAB/PreRisk-CoV2
```

**Data Repository**:
- GEO Accession: GSE198449
- GEO Accession: GSE178967

---

## Contact

For questions, bug reports, or feature requests:
- **GitHub Issues**: https://github.com/NTOUBiomedicalAILAB/PreRisk-CoV2/issues
- **Email**: [Your contact email]

---

## License

[Specify your license here, e.g., MIT, GPL-3.0, etc.]

---

## Acknowledgments

This work was supported by [funding sources if applicable]. We thank all contributors and the research community for valuable feedback.

---

## Version History

### v1.0.0 (2026-02)
- ✅ Unified command-line interface with argparse
- ✅ Internal validation (LOOCV) and External validation modes
- ✅ 5-protein biomarker panel (KNN-GA framework)
- ✅ SMOTE support for class imbalance
- ✅ Comprehensive performance metrics (8 indicators)
- ✅ ROC/PR curve visualization
- ✅ Excel export with detailed statistics
- ✅ Example scripts for quick start

---

## Project Structure

```
PreRisk-CoV2/
├── prerisk_cov2.py                      # Main unified program
├── README.md                             # This file
├── USAGE_GUIDE.md                        # Detailed usage documentation
├── example_internal_validation.sh        # Internal validation example (Linux/Mac)
├── example_external_validation.sh        # External validation example (Linux/Mac)
├── example_run_windows.bat               # Windows batch file example
├── quick_test.sh                         # System check script
├── Discovery.csv                         # Discovery cohort data (download from GEO)
├── Validation.csv                        # Validation cohort data (download from GEO)
└── results/                              # Output directory (auto-created)
    ├── internal_validation_*.xlsx
    ├── external_validation_*.xlsx
    ├── internal_roc_pr.png
    └── external_roc_pr.png
```

---

**Enjoy using PreRisk-CoV2! 🧬🔬**

For detailed usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md).
