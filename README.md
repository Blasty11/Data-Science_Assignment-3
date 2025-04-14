# Disease Feature Encoding & KNN Hyperparameter Tuning

This repository contains a Python script that:

1. **Loads** a medical dataset of diseases with associated risk factors, symptoms, and signs.
2. **Extracts** features using:
   - **TF‑IDF** (with tuned parameters: unigrams + bigrams, limited vocabulary, minimum document frequency)
   - **One‑Hot Encoding** of the original list fields.
3. **Combines** TF‑IDF and one‑hot features into a single matrix.
4. **Selects** the top 300 features by χ² score.
5. **Performs** a comprehensive **GridSearchCV** over KNN hyperparameters:
   - **Normalization**: none, StandardScaler, MinMaxScaler  
   - **Number of neighbors**: 3, 5, 7, 9, 11  
   - **Distance metric**: cosine, euclidean, manhattan  
   - **Weighting scheme**: uniform, distance  
6. **Reports** best parameters and compares accuracy by normalization method for both TF‑IDF and one‑hot feature sets.

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Installation](#installation)  
3. [Dataset](#dataset)  
4. [Usage](#usage)  
5. [Script Overview](#script-overview)  
   - [1. Load & Parse Data](#1-load--parse-data)  
   - [2. Feature Extraction & Combination](#2-feature-extraction--combination)  
   - [3. Feature Selection](#3-feature-selection)  
   - [4. KNN Grid Search](#4-knn-grid-search)  
   - [5. Results](#5-results)  
6. [Interpreting the Output](#interpreting-the-output)  
7. [Future Improvements](#future-improvements)  
8. [License](#license)

---

## Prerequisites

- Python 3.7 or higher  
- The following Python packages:
  - `pandas`
  - `numpy`
  - `scipy`
  - `scikit-learn`

You can install them via `pip`:

```bash
pip install pandas numpy scipy scikit-learn


Dataset
Place your disease_features.csv file in the root of this directory.
The CSV is expected to have at least these columns:

Disease — the disease name (used as the target label)

Risk Factors, Symptoms, Signs, Subtypes — each a string‑encoded Python list, e.g. "[\"hypertension\",\"diabetes\"]"

Script Overview
1. Load & Parse Data
Read disease_features.csv into a DataFrame.

Convert string‑encoded lists into real Python lists using ast.literal_eval.

2. Feature Extraction & Combination
TF‑IDF:

max_features=1000

min_df=2

ngram_range=(1,2)

One‑Hot:

MultiLabelBinarizer on each of Risk Factors, Symptoms, Signs.

Combine into one sparse matrix via scipy.sparse.hstack.

3. Feature Selection
Use SelectKBest(chi2, k=300) to reduce the combined feature matrix to 300 most relevant features.

4. KNN Grid Search
Define a Pipeline with a placeholder scaler step and a KNeighborsClassifier.

Parameter grid:
{
  'scaler':      ['passthrough', StandardScaler(), MinMaxScaler()],
  'knn__n_neighbors': [3,5,7,9,11],
  'knn__metric':      ['cosine','euclidean','manhattan'],
  'knn__weights':     ['uniform','distance']
}
5‑fold Stratified CV (StratifiedKFold with shuffle=True, random_state=42).

Run GridSearchCV separately on:

Selected TF‑IDF features

Selected One‑Hot features

5. Results
Best hyperparameters for each feature set.

Mean CV accuracy for the best model.

Pivot tables showing accuracy broken down by normalization method.

Interpreting the Output
Best Params:
Shows the combination of scaler, n_neighbors, metric, and weights that yielded the highest CV accuracy.

Accuracy by Normalization:
A small table listing mean CV accuracy for each normalization option (passthrough, StandardScaler, MinMaxScaler).

The highest value indicates which preprocessing step works best for KNN on that feature set.

Future Improvements
Expand grid: include more values of k, try different p values for Minkowski distance.

Feature engineering: experiment with different max_features, min_df, or additional n‑gram ranges.

Dimensionality reduction: apply TruncatedSVD or PCA before KNN to see if lower‑dimensional data helps.

Ensemble methods: compare KNN to Random Forest, Gradient Boosting, or SVM on the same selected features.

