# Multi-Omic Biomarker Discovery: MOFA+ & Random Forest Workflow

This repository demonstrates a robust **biomarker strategy** using Multi-Omics Factor Analysis (MOFA+) for non-linear dimensionality reduction, followed by a Random Forest classifier to predict patient outcomes.

## 🚀 Workflow Overview
1.  **Data Generation:** Simulating matched mRNA and Proteomics data for 100 patients.
2.  **Unsupervised Integration:** Using [MOFA+] to extract latent factors that capture shared variance across omic layers.
3.  **Supervised Learning:** Training a Random Forest model on the extracted factors to predict "Responders" vs "Non-Responders."
4.  **Biological Interpretation:** Identifying which molecular features contribute most to the predictive latent factors.

## 📁 Repository Structure
*   `analysis.R`: The complete script for data simulation, MOFA training, and ML modeling.
*   `model.hdf5`: The trained MOFA+ object (generated after running the script).
*   `README.md`: Project documentation.

## 🛠️ Requirements
You will need the following R packages:
```R
install.packages(c("dplyr", "ggplot2", "randomForest", "caret"))
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("MOFA2")
