import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

from app_utils import load_model, load_data, predict
from constants import FEATURES, TARGET, CONFIG_FAI
from .static_fai import get_aif_metric, compute_fairness_measures, plot_hist, alg_fai


def print_model_perf(y_val, y_pred):
    text = ""
    text += "Model accuracy = {:.4f}\n".format(metrics.accuracy_score(y_val, y_pred))
    text += "Weighted Average Precision = {:.4f}\n".format(
        metrics.precision_score(y_val, y_pred, average="weighted"))
    text += "Weighted Average Recall = {:.4f}\n\n".format(
        metrics.recall_score(y_val, y_pred, average="weighted"))
    text += metrics.classification_report(y_val, y_pred, digits=4)
    return text


def fai():
    st.title("Fairness AI Dashboard")

    st.sidebar.title("Instructions")
    st.sidebar.info("- See `Global explainability` page for instructions on model and data.\n"
                    "- Also set `CONFIG_FAI` in `constants.py`.")

    protected_attribute = st.selectbox("Select protected column.", list(CONFIG_FAI.keys()))

    # Load sample, data
    clf = load_model("output/lgb.pkl")
    valid = load_data("output/valid.csv")  # Fairness does not allow NaNs
    x_valid = valid[FEATURES]
    y_valid = valid[TARGET].values

    # Predict on val data
    y_prob = predict(clf, x_valid)

    st.header("Prediction Distributions")
    cutoff = st.slider("Set probability cutoff", 0., 1., 0.5, 0.01, key="proba")
    y_pred = (y_prob > cutoff).astype(int)

    source = pd.DataFrame({
        protected_attribute: x_valid[protected_attribute].values,
        "Prediction": y_prob,
    })
    st.altair_chart(plot_hist(source, cutoff), use_container_width=True)

    st.header("Model Performance")
    st.text(print_model_perf(y_valid, y_pred))

    st.header("Algorithmic Fairness Metrics")    
    fthresh = st.slider("Set fairness threshold", 0., 1., 0.2, 0.05, key="fairness")

    # Compute fairness measures
    privi_info = CONFIG_FAI[protected_attribute]
    aif_metric = get_aif_metric(
        valid,
        y_valid,
        y_pred,
        protected_attribute,
        privi_info["privileged_attribute_values"],
        privi_info["unprivileged_attribute_values"],
    )
    fmeasures = compute_fairness_measures(aif_metric)
    fmeasures["Fair?"] = fmeasures["Ratio"].apply(
        lambda x: "Yes" if np.abs(x - 1) < fthresh else "No")
    alg_fai(fmeasures, aif_metric, fthresh)

    st.subheader("Notes")
    st.write("**Equal opportunity**:")
    st.latex(r"\frac{\text{FNR}(D=\text{unprivileged})}{\text{FNR}(D=\text{privileged})}")
    st.write("**Predictive parity**:")
    st.latex(r"\frac{\text{PPV}(D=\text{unprivileged})}{\text{PPV}(D=\text{privileged})}")
    st.write("**Statistical parity**:")
    st.latex(r"\frac{\text{Selection Rate}(D=\text{unprivileged})}{\text{Selection Rate}(D=\text{privileged})}")


if __name__ == "__main__":
    fai()
