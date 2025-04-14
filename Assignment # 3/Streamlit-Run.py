import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    make_scorer
)

import matplotlib.pyplot as plt
import seaborn as sns

# ── Streamlit page setup ─────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("Disease Classification: TF-IDF vs One-Hot Encoding")

# ── File upload widget ────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a preprocessed CSV file", type="csv")

if not uploaded_file:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────────────
df = pd.read_csv(uploaded_file)
st.success("File uploaded successfully!")
st.write("### Data Preview")
st.dataframe(df.head())

# ── Sidebar controls ─────────────────────────────────────────────────────────────────
st.sidebar.subheader("🔧 Model Settings")
selected_k = st.sidebar.selectbox("Select k for KNN", [3, 5, 7])
selected_metric = st.sidebar.selectbox(
    "Select distance metric for KNN",
    ['euclidean', 'manhattan', 'cosine']
)

# ── Prepare combined text for TF‑IDF ─────────────────────────────────────────────────
# We merge Risk Factors, Symptoms, Signs into one string per row
df['combined_text'] = (
    df['Risk Factors'].fillna('') + ' '
    + df['Symptoms'].fillna('') + ' '
    + df['Signs'].fillna('')
)

# ── 1) TF‑IDF vectorization ──────────────────────────────────────────────────────────
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['combined_text']).toarray()

# ── 2) One‑hot encoding ─────────────────────────────────────────────────────────────
# Each column is assumed to contain a Python list (as a string) of tokens.
mlb = MultiLabelBinarizer()
onehot_parts = []
for col in ['Risk Factors', 'Symptoms', 'Signs']:
    # Convert string repr of list into an actual list, then fit_transform
    tokens = df[col].apply(lambda x: eval(x) if isinstance(x, str) else [])
    onehot_parts.append(mlb.fit_transform(tokens))

# Concatenate the three one‑hot matrices side by side
onehot_matrix = np.concatenate(onehot_parts, axis=1)

# ── 3) Map diseases to high‑level categories ─────────────────────────────────────────
category_mapping = {
    "Acute Coronary Syndrome": "Cardiovascular",
    "Adrenal Insufficiency": "Endocrine",
    "Alzheimer": "Neurological",
    "Aortic Dissection": "Cardiovascular",
    "Asthma": "Respiratory",
    "Atrial Fibrillation": "Cardiovascular",
    "Cardiomyopathy": "Cardiovascular",
    "COPD": "Respiratory",
    "Diabetes": "Endocrine",
    "Epilepsy": "Neurological",
    "Gastritis": "Gastrointestinal",
    "Gastro-oesophageal Reflux Disease": "Gastrointestinal",
    "Heart Failure": "Cardiovascular",
    "Hyperlipidemia": "Cardiovascular",
    "Hypertension": "Cardiovascular",
    "Migraine": "Neurological",
    "Multiple Sclerosis": "Neurological",
    "Peptic Ulcer Disease": "Gastrointestinal",
    "Pituitary Disease": "Endocrine",
    "Pneumonia": "Respiratory",
    "Pulmonary Embolism": "Cardiovascular",
    "Stroke": "Neurological",
    "Thyroid Disease": "Endocrine",
    "Tuberculosis": "Infectious",
    "Upper Gastrointestinal Bleeding": "Gastrointestinal"
}
df['Category'] = df['Disease'].map(category_mapping)
target = df['Category']

# ── 4) Model evaluation function ────────────────────────────────────────────────────
def evaluate_models(tfidf_matrix, onehot_matrix, target):
    """
    Trains and cross-validates:
      - KNN with user-selected k & metric
      - Logistic Regression (balanced classes)
    on both TF-IDF and one-hot features.
    Returns a DataFrame of mean CV scores.
    """
    # Define the four scorers we care about
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1': make_scorer(f1_score, average='weighted', zero_division=0)
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = []

    for feature_name, X in [('TF-IDF', tfidf_matrix), ('One-Hot', onehot_matrix)]:
        # ── KNN ───────────────────────────────────────────
        knn = KNeighborsClassifier(
            n_neighbors=selected_k,
            metric=selected_metric,
            weights='distance'
        )
        knn_scores = cross_validate(knn, X, target, cv=cv, scoring=scoring)
        results.append({
            'Model': 'KNN',
            'Feature': feature_name,
            'Accuracy': np.mean(knn_scores['test_accuracy']),
            'Precision': np.mean(knn_scores['test_precision']),
            'Recall': np.mean(knn_scores['test_recall']),
            'F1-Score': np.mean(knn_scores['test_f1'])
        })

        # ── Logistic Regression ─────────────────────────
        lr = LogisticRegression(
            max_iter=2000,
            class_weight='balanced'
        )
        lr_scores = cross_validate(lr, X, target, cv=cv, scoring=scoring)
        results.append({
            'Model': 'Logistic Regression',
            'Feature': feature_name,
            'Accuracy': np.mean(lr_scores['test_accuracy']),
            'Precision': np.mean(lr_scores['test_precision']),
            'Recall': np.mean(lr_scores['test_recall']),
            'F1-Score': np.mean(lr_scores['test_f1'])
        })

    return pd.DataFrame(results)

# ── 5) Run evaluation & show results ────────────────────────────────────────────────
with st.spinner("Evaluating models..."):
    results_df = evaluate_models(tfidf_matrix, onehot_matrix, target)

st.write("### Model Results")
st.dataframe(results_df)

# ── 6) Plot F1-Score comparison ─────────────────────────────────────────────────────
st.write("### Model Performance by F1-Score")
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Feature',
    y='F1-Score',
    hue='Model',
    data=results_df
)
plt.title("KNN vs Logistic Regression (F1-Score)")
st.pyplot(plt)
