"""
App for FAI.
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

from app_utils import load_model, load_data, predict
from constants import FEATURES, TARGET, TARGET_CLASSES, CONFIG_FAI
from xai_fairness.static_fai import (
    binarize,
    get_aif_metric,
    compute_fairness_measures,
    plot_hist,
    alg_fai,
)


def print_model_perf(y_val, y_pred):
    text = ""
    text += "Model accuracy = {:.4f}\n".format(metrics.accuracy_score(y_val, y_pred))
    text += "Weighted Average Precision = {:.4f}\n".format(
        metrics.precision_score(y_val, y_pred, average="weighted"))
    text += "Weighted Average Recall = {:.4f}\n\n".format(
        metrics.recall_score(y_val, y_pred, average="weighted"))
    text += metrics.classification_report(y_val, y_pred, digits=4)
    return text


def fai(version=1):
    st.title("Fairness AI Dashboard")

    protected_attribute = st.selectbox("Select protected column", list(CONFIG_FAI.keys()))

    # Load sample, data
    clf = load_model("models/lgb_clf.pkl")
    valid = load_data("data/valid.csv")
    y_valid = valid[TARGET].values
    valid_fai = valid[list(CONFIG_FAI.keys())]

    # Predict on val data
    y_score = predict(clf, valid[FEATURES])

    if version == 1:
        y_pred = (y_score[:, 1] > 0.5).astype(int)

        select_class = st.selectbox("Select class", TARGET_CLASSES, 1)
        true_class = binarize(y_valid, select_class)
        pred_class = binarize(y_pred, select_class)
    else:
        st.header("Prediction Distributions")
        y_prob = y_score[:, 1]
        cutoff = st.slider("Set probability cutoff", 0., 1., 0.5, 0.01, key="proba")
        y_pred = (y_prob > cutoff).astype(int)

        source = pd.DataFrame({
            protected_attribute: valid[protected_attribute].values,
            "Prediction": y_prob,
        })
        st.altair_chart(plot_hist(source, cutoff), use_container_width=True)

        st.header("Model Performance")
        st.text(print_model_perf(y_valid, y_pred))

        true_class = y_valid
        pred_class = y_pred

    st.header("Algorithmic Fairness Metrics")    
    fthresh = st.slider("Set fairness threshold", 0., 1., 0.2, 0.05, key="fairness")

    # Compute fairness measures
    privi_info = CONFIG_FAI[protected_attribute]
    aif_metric = get_aif_metric(
        valid_fai,
        true_class,
        pred_class,
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
