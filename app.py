from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_lightgbm_regressor.pkl"
SCALER_PATH = BASE_DIR / "scaler_advanced.pkl"

# Order must match what the scaler/model were trained on
FEATURES = ["year_numeric", "source_page", "univ_target_enc", "page_target_enc"]
TARGET_LABEL = "Predicted value"  # rename to your actual target

st.set_page_config(page_title="LightGBM Regressor", page_icon="📈", layout="centered")


# ----------------------------------------------------------------------------
# Load artifacts once
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict(df: pd.DataFrame, model, scaler, inverse_log: bool) -> np.ndarray:
    X = df[FEATURES].astype(float)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    if inverse_log:
        preds = np.expm1(preds)
    return preds


try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Could not load model/scaler: {e}")
    st.info(
        "Make sure `best_lightgbm_regressor.pkl` and `scaler_advanced.pkl` are in "
        "the same folder as `app.py`, and that the installed library versions "
        "match `requirements.txt`."
    )
    st.stop()

# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("📈 LightGBM Regression App")
st.caption("Enter raw (unscaled) feature values. The saved scaler is applied automatically.")

with st.sidebar:
    st.header("Options")
    inverse_log = st.checkbox(
        "Apply expm1 to output",
        value=False,
        help="Tick this only if the target was log1p-transformed during training.",
    )
    st.markdown("**Model features**")
    st.write(FEATURES)

tab_single, tab_batch = st.tabs(["Single prediction", "Batch (CSV)"])

# --- Single prediction ------------------------------------------------------
with tab_single:
    c1, c2 = st.columns(2)
    with c1:
        year_numeric = st.number_input(
            "year_numeric", min_value=2000.0, max_value=2100.0, value=2025.0, step=1.0
        )
        univ_target_enc = st.number_input(
            "univ_target_enc", value=392.12, step=1.0, format="%.2f"
        )
    with c2:
        source_page = st.number_input(
            "source_page", min_value=0.0, value=38.0, step=1.0
        )
        page_target_enc = st.number_input(
            "page_target_enc", value=392.12, step=1.0, format="%.2f"
        )

    if st.button("Predict", type="primary"):
        row = pd.DataFrame(
            [[year_numeric, source_page, univ_target_enc, page_target_enc]],
            columns=FEATURES,
        )
        pred = predict(row, model, scaler, inverse_log)[0]
        st.metric(TARGET_LABEL, f"{pred:,.2f}")

# --- Batch prediction -------------------------------------------------------
with tab_batch:
    st.write("Upload a CSV containing these columns:")
    st.code(", ".join(FEATURES))

    template = pd.DataFrame([[2025, 38, 392.12, 392.12]], columns=FEATURES)
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            data[TARGET_LABEL] = predict(data, model, scaler, inverse_log)
            st.dataframe(data, use_container_width=True)
            st.download_button(
                "Download predictions",
                data.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
