# PreRisk-CoV2
Pre-exposure risk assessment for SARS-CoV-2 using plasma protein  biomarkers. Machine learning framework (KNN+GA) with 5-protein panel.  Dual validation: LOOCV internal + independent external cohort.  Python implementation.

**📄 Paper:** *Predicting SARS-CoV-2 Susceptibility from Pre-Infection Serum Proteins: A Machine Learning Approach*


## 📋 Repository Scope

This repository contains:

✅ **Internal Validation** - Leave-One-Out Cross-Validation (LOOCV) on discovery cohort  
✅ **External Validation** - Independent validation on external cohort  
✅ **KNN Classifier** - K-Nearest Neighbors with optimized parameters  
✅ **SMOTE Oversampling** - Class imbalance handling


<br>


### What's Included

✅ **Internal Validation** (`Internal-validation.py`)
   - Leave-One-Out Cross-Validation (LOOCV)
   - 100 iterations for robust performance estimation
   - Discovery cohort 

✅ **External Validation** (`External-validation.py`)
   - Independent external cohort testing
   - 100 iterations for validation
   - Validation cohort 

✅ **Machine Learning Pipeline**
   - K-Nearest Neighbors (KNN) classifier
   - SMOTE oversampling for class imbalance
   - 5-protein biomarker panel (pre-selected features)

✅ **Performance Metrics**
   - AUROC, AUPRC
   - Accuracy, Sensitivity, Specificity
   - MCC (Matthews Correlation Coefficient)
   - F1-score, Precision

✅ **Visualization**
   - ROC curves
   - Precision-Recall curves

<br>

---

## 📋 Overview

Two complementary approaches for selecting optimal protein biomarkers from high-dimensional data:

### 1. Genetic Algorithm with OAX
- **Purpose**: Intelligent search in large feature spaces (50+ candidates)
- **Innovation**: Orthogonal Array Crossover + Simulated Annealing
- **Fitness**: Matthews Correlation Coefficient (MCC)
- **Best for**: Exploring 92 protein candidates

### 2. Permutation Selection
- **Purpose**: Exhaustive search for guaranteed optimal solutions
- **Method**: Test all combinations C(n, k)
- **Best for**: Confirming results with small candidate sets (≤10 proteins)

Both methods use:
- K-Nearest Neighbors (KNN) classifier
- Leave-One-Out Cross-Validation (LOOCV)
- SMOTE for handling class imbalance

---

<br>

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/NTOUBiomedicalAILAB/PreRisk-CoV2.git
cd PreRisk-CoV2

# Install dependencies
pip install -r requirements.txt
```
Usage

```bash
# Internal validation - LOOCV on discovery cohort with 100 iterations
python "Internal validation.py"

# External validation - Independent cohort testing with 100 iterations
python "External validation.py"
```

<br>

## 📊 Data Availability

### Public Datasets

All data supporting this study are publicly available in the **Gene Expression Omnibus** (https://www.ncbi.nlm.nih.gov/geo) under accession numbers **GSE198449** and **GSE178967**.



### Data Privacy

🔒 **Note**: The actual patient-level dataset files are not included in this repository due to privacy regulations and ethical considerations. Researchers can access the data through GEO with appropriate institutional review.



<br>

## 🔬 Complete Validation Pipeline

This repository provides to ensure your selected biomarkers are reliable and generalizable, you must complete the following validation steps:

---


### 📊 Step 1: Feature Selection 



**Purpose**: Identify the optimal subset of protein biomarkers from 92 candidates.

**Output**: A list of selected feature indices

**Methods**:
- **Genetic Algorithm with OAX**: Evolutionary search for large feature spaces
- **Permutation Search**: Exhaustive evaluation for small candidate sets

---


### 🔄 Step 2: Internal Validation 

**Purpose**: Evaluate the stability and reliability of selected features using the training dataset.

**Why It's Important**:
- Detects overfitting to the training data
- Assesses model stability across different data splits
- Provides confidence intervals for performance metrics

**Methodology**:

**Algorithm**: Leave-One-Out Cross-Validation (LOOCV)

For each sample in the training set:
1. Hold out one sample as test
2. Train on remaining samples
3. Apply SMOTE to balance training data
4. Train KNN classifier (n_neighbors=5)
5. Predict the held-out sample
6. Record prediction and probability

Repeat entire process 100 times with different random seeds
Calculate mean ± standard deviation for all metrics



