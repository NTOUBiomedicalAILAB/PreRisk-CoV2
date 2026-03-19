## 📊 Data Availability

### Public Datasets

De-identified individual participant data supporting the findings of this study are available in the Gene Expression Omnibus (GEO) (https://www.ncbi.nlm.nih.gov/geo) under accession num-bers **GSE198449 (CHARM cohort)** and **GSE178967 (CEIM cohort)**.

1. NPX data for the CHARM cohort are accessible via the supplementary material of Soares-Schanoski et al. https://pmc.ncbi.nlm.nih.gov/articles/PMC9037090
2. NPX data for the CEIM cohort are available at the following repository: [https://github.com/hzc363/COVID19_system_immunology/blob/master/OLINK\%20Proteomics/olink.csv](https://github.com/hzc363/COVID19_system_immunology/blob/master/OLINK%20Proteomics/olink.csv)

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
