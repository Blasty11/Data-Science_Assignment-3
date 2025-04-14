# Import necessary libraries
import pandas as pd
import numpy as np
import ast
import streamlit as st
from scipy.sparse import hstack
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# ----------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------
# Read the CSV file into a DataFrame
df = pd.read_csv('disease_features.csv')

# ----------------------------------------------------------------
# 2. TASK 1: FEATURE EXTRACTION
#    - Convert list-like strings into actual lists
#    - Create string versions for TF-IDF
#    - Compute TF-IDF and combine
#    - Compute one-hot encoding and compare
# ----------------------------------------------------------------

# 2.1 Parse the columns that contain Python-list strings into real lists
columns_to_parse = ['Risk Factors', 'Symptoms', 'Signs', 'Subtypes']
for col in columns_to_parse:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

# 2.2 Create space-separated string columns for TF-IDF
df['Risk Factors_str'] = df['Risk Factors'].apply(lambda x: ' '.join(x))
df['Symptoms_str']    = df['Symptoms'].apply(lambda x: ' '.join(x))
df['Signs_str']       = df['Signs'].apply(lambda x: ' '.join(x))

# 2.3 Apply TF-IDF vectorization to each string column
vectorizers  = {}
tfidf_matrices = {}
for col in ['Risk Factors_str', 'Symptoms_str', 'Signs_str']:
    vec = TfidfVectorizer()
    mat = vec.fit_transform(df[col])
    vectorizers[col] = vec
    tfidf_matrices[col] = mat

# 2.4 Combine the three TF-IDF matrices into one sparse matrix
combined_tfidf = hstack([
    tfidf_matrices['Risk Factors_str'],
    tfidf_matrices['Symptoms_str'],
    tfidf_matrices['Signs_str']
])
dense_tfidf = combined_tfidf.toarray()  # Convert to dense for analysis

# 2.5 One-hot encode the original list columns
mlb = MultiLabelBinarizer()
onehot_matrices = {}
for col in ['Risk Factors', 'Symptoms', 'Signs']:
    mat = mlb.fit_transform(df[col])
    onehot_matrices[col] = mat

# Combine the one-hot matrices into one dense array
combined_onehot = np.hstack([
    onehot_matrices['Risk Factors'],
    onehot_matrices['Symptoms'],
    onehot_matrices['Signs']
])

# 2.6 Compare sparsity, dimensionality, and basic statistics
tfidf_sparsity  = 1.0 - (np.count_nonzero(dense_tfidf) / dense_tfidf.size)
onehot_sparsity = 1.0 - (np.count_nonzero(combined_onehot) / combined_onehot.size)

print("=== Sparsity Comparison ===")
print(f"TF-IDF Sparsity:  {tfidf_sparsity:.2%}")
print(f"One-hot Sparsity: {onehot_sparsity:.2%}\n")

print("=== Dimensionality Comparison ===")
print(f"TF-IDF features:  {combined_tfidf.shape[1]}")
print(f"One-hot features: {combined_onehot.shape[1]}\n")

print("=== Matrix Statistics ===")
print("TF-IDF:")
print(f"- Non-zero elements: {combined_tfidf.nnz}")
print(f"- Mean value:        {combined_tfidf.mean():.4f}\n")
print("One-hot:")
print(f"- Non-zero elements: {np.count_nonzero(combined_onehot)}")
print(f"- Mean value:        {combined_onehot.mean():.4f}\n")

# 2.7 Detailed TF-IDF value distribution
tfidf_values = combined_tfidf.data
print("=== TF-IDF Value Distribution ===")
print(f"- Min:    {tfidf_values.min():.4f}")
print(f"- Max:    {tfidf_values.max():.4f}")
print(f"- Mean:   {tfidf_values.mean():.4f}")
print(f"- Median: {np.median(tfidf_values):.4f}")
print(f"- Std Dev:{tfidf_values.std():.4f}\n")

# 2.8 One-hot value distribution
onehot_values = combined_onehot.flatten()
print("=== One-hot Value Distribution ===")
print(f"- Unique values: {np.unique(onehot_values)}")
print(f"- Min:           {onehot_values.min():.4f}")
print(f"- Max:           {onehot_values.max():.4f}\n")

# 2.9 Memory usage comparison
print("=== Memory Usage (KB) ===")
print(f"TF-IDF:  {combined_tfidf.data.nbytes  / 1024:.2f} KB")
print(f"One-hot: {combined_onehot.nbytes     / 1024:.2f} KB\n")

# 2.10 Information density (avg non-zero features per sample)
print("=== Information Density ===")
print(f"TF-IDF avg features per sample:  {combined_tfidf.nnz / combined_tfidf.shape[0]:.2f}")
print(f"One-hot avg features per sample: {np.count_nonzero(combined_onehot) / combined_onehot.shape[0]:.2f}\n")

# ----------------------------------------------------------------
# 3. TASK 2: DIMENSIONALITY REDUCTION
#    - Use TruncatedSVD on TF-IDF (sparse)
#    - Use PCA on one-hot (dense)
#    - Print explained variance
#    - Plot 2D scatter for comparison
# ----------------------------------------------------------------

# 3.1 Apply TruncatedSVD and PCA
n_components = 3
svd = TruncatedSVD(n_components=n_components)
tfidf_reduced  = svd.fit_transform(combined_tfidf)

pca = PCA(n_components=n_components)
onehot_reduced = pca.fit_transform(combined_onehot)

# 3.2 Print explained variance ratios
print("=== Explained Variance Ratios ===\n")
print("TF-IDF (TruncatedSVD):")
print(f"- Total variance explained: {svd.explained_variance_ratio_.sum():.4f}")
for i, ratio in enumerate(svd.explained_variance_ratio_, 1):
    print(f"  Component {i}: {ratio:.4f}")

print("\nOne-hot (PCA):")
print(f"- Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")
for i, ratio in enumerate(pca.explained_variance_ratio_, 1):
    print(f"  Component {i}: {ratio:.4f}")
print()

# 3.3 Prepare categories for plotting
category_mapping = {
    "Acute Coronary Syndrome": "Cardiovascular",
    "Adrenal Insufficiency":    "Endocrine",
    "Alzheimer":                "Neurological",
    "Aortic Dissection":        "Cardiovascular",
    "Asthma":                   "Respiratory",
    "Atrial Fibrillation":      "Cardiovascular",
    "Cardiomyopathy":           "Cardiovascular",
    "COPD":                     "Respiratory",
    "Diabetes":                 "Endocrine",
    "Epilepsy":                 "Neurological",
    "Gastritis":                "Gastrointestinal",
    "Gastro-oesophageal Reflux Disease": "Gastrointestinal",
    "Heart Failure":            "Cardiovascular",
    "Hyperlipidemia":           "Cardiovascular",
    "Hypertension":             "Cardiovascular",
    "Migraine":                 "Neurological",
    "Multiple Sclerosis":       "Neurological",
    "Peptic Ulcer Disease":     "Gastrointestinal",
    "Pituitary Disease":        "Endocrine",
    "Pneumonia":                "Respiratory",
    "Pulmonary Embolism":       "Cardiovascular",
    "Stroke":                   "Neurological",
    "Thyroid Disease":          "Endocrine",
    "Tuberculosis":             "Infectious",
    "Upper Gastrointestinal Bleeding": "Gastrointestinal"
}
df['Category'] = df['Disease'].map(category_mapping)
unique_categories = df['Category'].unique()
category_to_num   = {cat: idx for idx, cat in enumerate(unique_categories)}
category_nums     = df['Category'].map(category_to_num)

# 3.4 Plot 2D projections side by side
plt.figure(figsize=(16, 7))

# TF-IDF plot
plt.subplot(1, 2, 1)
plt.scatter(tfidf_reduced[:, 0], tfidf_reduced[:, 1], c=category_nums, cmap='viridis', alpha=0.8)
plt.title('TF-IDF Vectorization (2D)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
legend_elems = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=plt.cm.viridis(category_to_num[cat]/len(category_to_num)),
           markersize=8, label=cat)
    for cat in unique_categories
]
plt.legend(handles=legend_elems, title="Disease Categories")
plt.grid(True, linestyle='--', alpha=0.6)

# One-hot plot
plt.subplot(1, 2, 2)
plt.scatter(onehot_reduced[:, 0], onehot_reduced[:, 1], c=category_nums, cmap='viridis', alpha=0.8)
plt.title('One-hot Encoding (2D)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend(handles=legend_elems, title="Disease Categories")
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# Discussion (as comments):
# - TF-IDF shows clearer separation because it weights unique terms more heavily.
# - One-hot clusters overlap more, since all features are treated equally.

# ----------------------------------------------------------------
# 4. TASK 3: MODEL TRAINING & EVALUATION
#    - KNN with various k and metrics
#    - Logistic Regression
#    - Cross-validated scoring
#    - Summarize results in tables
# ----------------------------------------------------------------

# 4.1 Prepare target and cross-validation
target = df['Category']
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Convert sparse TF-IDF to dense array for modeling
tfidf_array  = combined_tfidf.toarray()
onehot_array = combined_onehot

# Define hyperparameters and scoring
k_values = [3, 5, 7]
metrics_list = ['euclidean', 'manhattan', 'cosine']
scoring = {
    'accuracy':  make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall':    make_scorer(recall_score,    average='weighted', zero_division=0),
    'f1':        make_scorer(f1_score,        average='weighted', zero_division=0)
}
normalizers = {
    'None':           None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler':   MinMaxScaler()
}

# DataFrame to collect all results
results_df = pd.DataFrame(columns=[
    'Model', 'Feature', 'Normalization', 'k', 'Metric',
    'Accuracy', 'Precision', 'Recall', 'F1-Score'
])

# 4.2 KNN evaluation function
def evaluate_knn(X, feature_name):
    for norm_name, normalizer in normalizers.items():
        for k in k_values:
            for metric in metrics_list:
                try:
                    if normalizer is None:
                        model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights='distance')
                    else:
                        model = Pipeline([
                            ('scaler', normalizer),
                            ('knn', KNeighborsClassifier(n_neighbors=k, metric=metric, weights='distance'))
                        ])
                    cv_res = cross_validate(model, X, target, cv=cv, scoring=scoring)
                    results_df.loc[len(results_df)] = [
                        'KNN', feature_name, norm_name, k, metric,
                        cv_res['test_accuracy'].mean(),
                        cv_res['test_precision'].mean(),
                        cv_res['test_recall'].mean(),
                        cv_res['test_f1'].mean()
                    ]
                except:
                    results_df.loc[len(results_df)] = [
                        'KNN', feature_name, norm_name, k, metric,
                        np.nan, np.nan, np.nan, np.nan
                    ]

# Evaluate KNN on both feature sets
evaluate_knn(tfidf_array, 'TF-IDF')
evaluate_knn(onehot_array, 'One-hot')

# 4.3 Logistic Regression evaluation
for feature_name, X in [('TF-IDF', tfidf_array), ('One-hot', onehot_array)]:
    for norm_name, normalizer in normalizers.items():
        try:
            if normalizer is None:
                model = LogisticRegression(max_iter=2000, solver='saga',
                                           multi_class='auto', class_weight='balanced')
            else:
                model = Pipeline([
                    ('scaler', normalizer),
                    ('lr', LogisticRegression(max_iter=2000, solver='saga',
                                              multi_class='auto', class_weight='balanced'))
                ])
            cv_res = cross_validate(model, X, target, cv=cv, scoring=scoring)
            results_df.loc[len(results_df)] = [
                'Logistic Regression', feature_name, norm_name, 'N/A', 'N/A',
                cv_res['test_accuracy'].mean(),
                cv_res['test_precision'].mean(),
                cv_res['test_recall'].mean(),
                cv_res['test_f1'].mean()
            ]
        except:
            results_df.loc[len(results_df)] = [
                'Logistic Regression', feature_name, norm_name, 'N/A', 'N/A',
                np.nan, np.nan, np.nan, np.nan
            ]

# 4.4 Format numeric columns
for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
    results_df[col] = results_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

# 4.5 Display summary tables

# 4.5.1 KNN comparison by normalization
print("\n=== 1. KNN Comparison by Normalization ===")
for norm_name in results_df['Normalization'].unique():
    print(f"\n-- Normalization: {norm_name} --")
    subset = results_df[(results_df['Model']=='KNN') & (results_df['Normalization']==norm_name)]
    pivot = pd.pivot_table(
        subset,
        values=['Accuracy','F1-Score'],
        index=['k','Metric'],
        columns='Feature',
        aggfunc='first'
    )
    print(pivot)

# 4.5.2 Logistic Regression comparison
print("\n=== 2. Logistic Regression Comparison ===")
lr_subset = results_df[results_df['Model']=='Logistic Regression']
lr_pivot = pd.pivot_table(
    lr_subset,
    values=['Accuracy','Precision','Recall','F1-Score'],
    index='Normalization',
    columns='Feature',
    aggfunc='first'
)
print(lr_pivot)

# 4.5.3 Best models by feature type
results_df['F1_float'] = results_df['F1-Score'].replace('N/A', np.nan).astype(float)
best_tfidf  = results_df[results_df['Feature']=='TF-IDF'].loc[results_df[results_df['Feature']=='TF-IDF']['F1_float'].idxmax()]
best_onehot = results_df[results_df['Feature']=='One-hot'].loc[results_df[results_df['Feature']=='One-hot']['F1_float'].idxmax()]
best_models = pd.DataFrame([best_tfidf, best_onehot]).drop('F1_float', axis=1)
best_models.index = ['Best TF-IDF Model', 'Best One-hot Model']
print("\n=== 3. Best Models by Feature Type ===")
print(best_models[['Model','Normalization','k','Metric','Accuracy','Precision','Recall','F1-Score']])

# 4.5.4 Top 5 models overall
top5 = results_df.sort_values(by='F1_float', ascending=False).head(5).drop('F1_float', axis=1)
print("\n=== 4. Top 5 Models Overall ===")
print(top5[['Model','Feature','Normalization','k','Metric','Accuracy','Precision','Recall','F1-Score']])

# 4.5.5 Effect of k on KNN performance
if 'KNN' in top5['Model'].values:
    best_norm   = top5[top5['Model']=='KNN']['Normalization'].iloc[0]
    best_metric = top5[top5['Model']=='KNN']['Metric'].iloc[0]
else:
    best_norm   = normalizers.keys().__iter__().__next__()
    best_metric = metrics_list[0]

k_effect = results_df[
    (results_df['Model']=='KNN') &
    (results_df['Normalization']==best_norm) &
    (results_df['Metric']==best_metric)
]
k_pivot = pd.pivot_table(
    k_effect,
    values=['Accuracy','F1-Score'],
    index='k',
    columns='Feature',
    aggfunc='first'
)
print(f"\n=== 5. Effect of k (Normalization={best_norm}, Metric={best_metric}) ===")
print(k_pivot)
