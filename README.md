# Overview

This repository provides the base code for the Bachelor thesis project on the topic **Comparison of Feature Selection Algorithms in Gene Analysis**.

This study presents a comparative evaluation of feature selection algorithms in gene expression datasets on their ability to select the most relevant features for classification and how the selected genes aligns biologically with curated cancer pathways. 

Experiments are conducted on five publicly available Gene Expression Omnibus (GEO) datasets covering leukemia, prostate cancer, and lung cancer. Seven feature selection algorithms are evaluated, comprising two univariate filter-based methods including random selection and Pearson Correlation, and five multivariate wrapper-based methods, namely Support Vector Machine Recursive Feature Elimination (SVM RFE), Genetic Algorithm (GA), Population-Based Incremental Learning (PBIL), Simulated Annealing (SA), and Black Widow Optimization (BWO). 

A 10 × 10 cross-validation experiment setup is implemented across all algorithms and datasets. Results are analyzed in terms of accuracy, stability, and biological interpretation by comparing the top-50 selected features from the algorithms with genes existing in Kyoto Encyclopedia of Genes and Genomes (KEGG) cancer-specific pathways.


<br />


# Repository Structure

- `algorithms` directory contains the _py_ files for the implementation of each feature selection algorithm
- `dataset` directory contains the Jupyter Notebook for data pre-processing and the processed _csv_ files for each dataset.
- `result` directory contains the _json_ output for the summary results of the 10x10 CV feature selection experiments for each dataset, which is implemented in `outer_cv.py`
- `result_baseline` directory contains the _json_ output for the summary results of the baseline accuracies for each dataset, which is implemented in `baseline.py`
- `res_analysis` directory contains the Jupyter Notebook for result analysis of the feature selection algorithms of each dataset

<br />

# How to run
1. Install requirements 
   ```bash
   pip install -r requirements.txt
   ```

2. To execute the 10x10 CV experiments for all datasets, do
   ```bash
   python main.py
   ```

3. To get the baseline accuracies for all datasets, do
   ```bash
   python main_baseline.py
   ```
