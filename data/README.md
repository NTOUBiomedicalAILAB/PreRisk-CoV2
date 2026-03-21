## 📊 Data Availability

### Public Datasets

De-identified individual participant data supporting the findings of this study are available in the Gene Expression Omnibus (GEO) (https://www.ncbi.nlm.nih.gov/geo) under accession num-bers **GSE198449 (CHARM cohort)** and **GSE178967 (CEIM cohort)**.

1. NPX data for the CHARM cohort are accessible via the supplementary material of Soares-Schanoski et al. https://pmc.ncbi.nlm.nih.gov/articles/PMC9037090
2. NPX data for the CEIM cohort are available at the following repository: [https://github.com/hzc363/COVID19_system_immunology/blob/master/OLINK\%20Proteomics/olink.csv](https://github.com/hzc363/COVID19_system_immunology/blob/master/OLINK%20Proteomics/olink.csv)

---

## 📊 Input Data Format
The input consists of protein expression data (CSV format), and the output provides infection risk prediction.
To ensure compatibility with the FHE encryption and prediction pipeline, please format your input CSV as follows:

### CSV File Structure

- **Column 0**: `sample ID` - Unique identifier for each patient/sample.
- **Column 1**: `PCR result` - Ground truth labels (can be `Detected`/`Not` or `1`/`0`).
  - *Note: If using `--no-labels` for pure prediction, this column can contain placeholders.*
- **Column 2 ~ N**: Protein expression levels (e.g., Olink NPX values).

### 🧬 The 7-Protein Panel (Default)
By default, the system automatically extracts and encrypts the following 7 biomarkers using  name matching:
> **MCP-3, LIF-R, TRANCE, FGF-23, NT-3, CXCL1, CXCL6**

<br>

