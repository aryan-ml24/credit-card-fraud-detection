import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score,
    classification_report, precision_recall_fscore_support
)
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
st.title("💳 Credit Card Fraud Detection")
st.caption("Anomaly detection and machine-learning dashboard | Python • Scikit-learn • Pandas • Streamlit")

@st.cache_data
def make_demo_data(n=5000, seed=42):
    """Synthetic data only for testing the UI; not a replacement for the Kaggle dataset."""
    rng = np.random.default_rng(seed)
    fraud_n = int(n * 0.025)
    normal_n = n - fraud_n
    normal = rng.normal(0, 1, size=(normal_n, 8))
    fraud = rng.normal(0, 2.2, size=(fraud_n, 8))
    X = np.vstack([normal, fraud])
    y = np.r_[np.zeros(normal_n, dtype=int), np.ones(fraud_n, dtype=int)]
    order = rng.permutation(n)
    X, y = X[order], y[order]
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 9)])
    df.insert(0, "Amount", np.round(np.exp(rng.normal(3.0, 1.0, size=n)), 2))
    df.insert(0, "Time", np.arange(n))
    df["Class"] = y
    return df

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload Kaggle creditcard.csv", type=["csv"])
    use_demo = st.checkbox("Use built-in demo data", value=True)
    st.markdown("---")
    st.caption("Expected target column: Class (0 = legitimate, 1 = fraud).")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success(f"Loaded {len(df):,} rows")
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
elif use_demo:
    df = make_demo_data()
    st.info("Demo data is active. Upload Kaggle's creditcard.csv for meaningful evaluation.")
else:
    st.warning("Upload a CSV or enable demo data in the sidebar.")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]
target_candidates = [c for c in df.columns if c.lower() in ("class", "fraud", "is_fraud", "label", "target")]
if not target_candidates:
    st.error("Target column not found. Upload a labelled CSV containing a Class column (0/1).")
    st.stop()
target = target_candidates[0]
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()

raw_y = df[target]
if raw_y.dtype == "object":
    raw_y = raw_y.astype(str).str.strip().str.lower().map({
        "0": 0, "1": 1, "normal": 0, "legitimate": 0, "fraud": 1,
        "true": 1, "false": 0
    })
if raw_y.isna().any():
    st.error("Target must contain 0/1 or recognizable labels such as normal/fraud.")
    st.stop()
y = raw_y.astype(int)
if set(y.unique()) - {0, 1} or len(y.unique()) < 2:
    st.error("Target must contain both 0 (legitimate) and 1 (fraud).")
    st.stop()

X = df.drop(columns=[target]).select_dtypes(include=[np.number]).copy()
if X.empty:
    st.error("No numeric feature columns were found.")
    st.stop()
X = X.fillna(X.median(numeric_only=True)).replace([np.inf, -np.inf], 0)

# Limit very large files so the app remains responsive.
if len(X) > 30000:
    sampled = df.sample(30000, random_state=42)
    X = sampled.drop(columns=[target]).select_dtypes(include=[np.number]).copy()
    X = X.fillna(X.median(numeric_only=True)).replace([np.inf, -np.inf], 0)
    y = sampled[target].astype(int)
    st.warning("For speed, a reproducible sample of 30,000 rows is used.")

st.subheader("Dataset overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Transactions", f"{len(X):,}")
m2.metric("Numeric features", X.shape[1])
m3.metric("Fraud cases", f"{int(y.sum()):,}")
m4.metric("Fraud rate", f"{100*y.mean():.2f}%")
with st.expander("Preview data"):
    st.dataframe(df.head(10), use_container_width=True)

with st.sidebar:
    test_pct = st.slider("Test set (%)", 10, 40, 25, 5)
    contamination = st.slider("Anomaly detector contamination", 0.001, 0.10,
                              float(max(0.001, min(0.10, y.mean()))), 0.001)
    threshold = st.slider("Fraud probability threshold", 0.05, 0.95, 0.50, 0.05)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_pct / 100, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

with st.spinner("Training models..."):
    iso = IsolationForest(n_estimators=150, contamination=contamination,
                          random_state=42, n_jobs=-1)
    iso.fit(X_train_s)
    iso_pred = (iso.predict(X_test_s) == -1).astype(int)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination,
                             novelty=True, n_jobs=-1)
    lof.fit(X_train_s)
    lof_pred = (lof.predict(X_test_s) == -1).astype(int)

    # Balanced class weights help with highly imbalanced fraud datasets.
    model = RandomForestClassifier(
        n_estimators=180,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)
    probabilities = model.predict_proba(X_test_s)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

st.subheader("Model comparison")
comparison = []
for name, yhat in [
    ("Isolation Forest", iso_pred),
    ("Local Outlier Factor", lof_pred),
    (f"Random Forest (threshold {threshold:.2f})", predictions)
]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, yhat, average="binary", zero_division=0
    )
    comparison.append({"Model": name, "Precision": precision, "Recall": recall, "F1-score": f1})
st.dataframe(pd.DataFrame(comparison).set_index("Model").style.format("{:.3f}"),
             use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Random Forest confusion matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=["Legitimate", "Fraud"],
        ax=ax, colorbar=False
    )
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("ROC curve")
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"Random Forest (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)
    st.metric("ROC-AUC", f"{auc:.4f}")

with st.expander("Classification report"):
    st.code(classification_report(
        y_test, predictions, labels=[0, 1],
        target_names=["Legitimate", "Fraud"], zero_division=0
    ))

st.subheader("Try a transaction")
st.write("Adjust feature values, then ask the model for a prediction.")
medians = X.median(numeric_only=True)
values = {}
feature_cols = st.columns(3)
for i, name in enumerate(X.columns):
    with feature_cols[i % 3]:
        default_value = float(medians[name]) if pd.notna(medians[name]) else 0.0
        values[name] = st.number_input(name, value=default_value, format="%.5f", key=f"input_{i}")

if st.button("Predict transaction", type="primary"):
    one = pd.DataFrame([values], columns=X.columns)
    score = float(model.predict_proba(scaler.transform(one))[0, 1])
    if score >= threshold:
        st.error(f"⚠️ Potential fraud pattern detected. Model score: {score:.1%}")
    else:
        st.success(f"✅ Model predicts likely legitimate. Model score: {score:.1%}")
    st.caption("Educational prototype only; not suitable for real banking decisions.")

export_df = pd.DataFrame({
    "actual_class": y_test.to_numpy(),
    "predicted_class": predictions,
    "fraud_probability": probabilities
})
st.download_button(
    "Download test predictions CSV",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="fraud_predictions.csv",
    mime="text/csv"
)
st.markdown("---")
st.caption("Real-world fraud datasets are highly imbalanced. Evaluate precision, recall, F1 and ROC-AUC, not accuracy alone.")
