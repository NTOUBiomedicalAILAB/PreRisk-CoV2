# PreRisk-CoV2
Pre-exposure risk assessment for SARS-CoV-2 using plasma protein  biomarkers. Machine learning framework (KNN+GA) with 5-protein panel.  Dual validation: LOOCV internal + independent external cohort.  Python implementation.

**📄 Paper:** *Predicting SARS-CoV-2 Susceptibility from Pre-Infection Plasma Proteins: A Machine Learning Approach*

# protein-biomarker-feature-selection
🧬 Feature selection algorithms for discovering optimal protein biomarkers using GA with OAX and Permutation methods
<br>

##  Repository Scope

This repository contains feature selection methods :
- ✅ Genetic Algorithm with Orthogonal Array Crossover (OAX)
- ✅ Permutation-based exhaustive search
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

Usage
python "External validation.py"


python "Internal validation.py"
```

Data Format
Your input CSV file should contain:

| Column                 | Type   | Description                  | Example             |
| ---------------------- | ------ | ---------------------------- | ------------------- |
| sample ID              | string | Anonymized sample identifier | S001, S002, ...     |
| PCR result             | string | Detection result             | "Detected" or "Not" |
| Protein_1 ~ Protein_92 | float  | Protein concentration values | 0.123, 4.567, ...   |

All data supporting this study are publicly available in the Gene Expression Omnibus (https://www.ncbi.nlm.nih.gov/geo) under accession numbers GSE198449 and GSE178967.

🔒 Data Privacy: The actual dataset is not provided due to patient privacy concerns.



<br>

## 🔬 Complete Validation Pipeline

This repository provides **feature selection methods**. To ensure your selected biomarkers are reliable and generalizable, you must complete the following validation steps:

---


### 📊 Step 1: Feature Selection (This Repository)

**Files Provided**: 
- `permutation_selection.py`
- `genetic_algorithm_selection.py`

**Purpose**: Identify the optimal subset of protein biomarkers from 92 candidates.

**Output**: A list of selected feature indices (e.g., [3, 50, 40, 36, 83])

**Methods**:
- **Genetic Algorithm with OAX**: Evolutionary search for large feature spaces
- **Permutation Search**: Exhaustive evaluation for small candidate sets

---


### 🔄 Step 2: Internal Validation (Required)

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



