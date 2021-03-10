"""
App for fairness comparison.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from xai_fairness.toolkit_fai import (
    prepare_dataset, 
    get_aif_metric,
    get_perf_measure_by_group,
)
from xai_fairness.static_fai import (
    custom_fmeasures,
    alg_fai,
    fairness_notes,
    confusion_matrix_chart,
    fmeasures_chart,
)

from data.constants import FEATURES, TARGET, PROTECTED_FEATURES
from data.utils import load_model, load_data, predict, print_model_perf

CONFIG = yaml.load(open("config.yaml", "r"), Loader=yaml.SafeLoader)
METRICS_TO_USE = CONFIG["metrics_to_use"]


@st.cache
def prepare_pred(x_valid, y_valid, debias=False):
    # Load model
    clf = load_model("models/lgb_clf.pkl")

    # Predict on val data
    y_prob = predict(clf, x_valid)[:, 1]

    # st.header("Prediction Distributions")
    cutoff = 0.5  # st.slider("Set probability cutoff", 0., 1., 0.5, 0.01, key="proba")
    y_pred = (y_prob > cutoff).astype(int)

    if debias:
        model = load_model("models/eq_odds_sex.pkl")
        attr = "Sex=Male"
        predicted_val = prepare_dataset(
            x_valid,
            y_pred,
            attr,
            PROTECTED_FEATURES[attr]["privileged_attribute_values"],
            PROTECTED_FEATURES[attr]["unprivileged_attribute_values"],
        )

        adj_pred_val = model.predict(predicted_val)
        y_pred = adj_pred_val.labels

    # Model performance
    text_model_perf = print_model_perf(y_valid, y_pred)

    return y_pred, text_model_perf


def fai(debias=False):
    st.subheader("User Write-up")
    if debias:
        st.write(CONFIG["after_mitigation"])
    else:
        st.write(CONFIG["before_mitigation"])

    protected_attribute = st.selectbox("Select protected feature.", list(PROTECTED_FEATURES.keys()))

    # Load data
    valid = load_data("data/valid.csv")
    x_valid = valid[FEATURES]
    y_valid = valid[TARGET].values

    # Get predictions
    y_pred, text_model_perf = prepare_pred(x_valid, y_valid, debias=debias)
    
    st.header("Model Performance")
    st.text(text_model_perf)

    st.header("Algorithmic Fairness Metrics")
    fthresh = st.slider("Set fairness deviation threshold", 0., 1., 0.2, 0.05)

    # Compute fairness measures
    privi_info = PROTECTED_FEATURES[protected_attribute]
    aif_metric = get_aif_metric(
        valid,
        y_valid,
        y_pred,
        protected_attribute,
        privi_info["privileged_attribute_values"],
        privi_info["unprivileged_attribute_values"],
    )
    fmeasures = custom_fmeasures(aif_metric, threshold=fthresh, fairness_metrics=METRICS_TO_USE)
    alg_fai(fmeasures, aif_metric, fthresh)

    st.subheader("Notes")
    fairness_notes()


def chart_cm_comparison(orig_clf_metric, clf_metric, privileged, title):
    cm1 = orig_clf_metric.binary_confusion_matrix(privileged=privileged)
    cm2 = clf_metric.binary_confusion_matrix(privileged=privileged)
    c1 = confusion_matrix_chart(cm1, f"{title}: Before Mitigation")
    c2 = confusion_matrix_chart(cm2, f"{title}: After Mitigation")
    return c1 | c2


def compare():
    protected_attribute = st.selectbox("Select protected column.", list(PROTECTED_FEATURES.keys()))

    # Load data
    valid = load_data("data/valid.csv")
    x_valid = valid[FEATURES]
    y_valid = valid[TARGET].values

    # Get predictions
    orig_y_pred, orig_text_model_perf = prepare_pred(x_valid, y_valid, debias=False)
    y_pred, text_model_perf = prepare_pred(x_valid, y_valid, debias=True)

    st.header("Model Performance")
    st.subheader("Before Mitigation")
    st.text(orig_text_model_perf)
    st.subheader("After Mitigation")
    st.text(text_model_perf)

    st.header("Algorithmic Fairness Metrics")
    fthresh = st.slider("Set fairness deviation threshold", 0., 1., 0.2, 0.05)
    st.write(f"Fairness is when **ratio is between {1 - fthresh:.2f} and {1 + fthresh:.2f}**.")

    # Compute fairness measures
    privi_info = PROTECTED_FEATURES[protected_attribute]
    orig_aif_metric = get_aif_metric(
        valid,
        y_valid,
        orig_y_pred,
        protected_attribute,
        privi_info["privileged_attribute_values"],
        privi_info["unprivileged_attribute_values"],
    )
    orig_fmeasures = custom_fmeasures(orig_aif_metric, threshold=fthresh)

    aif_metric = get_aif_metric(
        valid,
        y_valid,
        y_pred,
        protected_attribute,
        privi_info["privileged_attribute_values"],
        privi_info["unprivileged_attribute_values"],
    )
    fmeasures = custom_fmeasures(aif_metric, threshold=fthresh)

    for m in METRICS_TO_USE:
        source = pd.concat([orig_fmeasures.query(f"Metric == '{m}'"),
                            fmeasures.query(f"Metric == '{m}'")])
        source["Metric"] = ["1-Before Mitigation", "2-After Mitigation"]

        st.write(m)
        st.altair_chart(fmeasures_chart(source, fthresh), use_container_width=True)

#     st.altair_chart(chart_cm_comparison(orig_clf_metric, clf_metric, None, "All"),
#                     use_container_width=False)
#     st.altair_chart(chart_cm_comparison(orig_clf_metric, clf_metric, True, "Privileged"),
#                     use_container_width=False)
#     st.altair_chart(chart_cm_comparison(orig_clf_metric, clf_metric, False, "Unprivileged"),
#                     use_container_width=False)

    
if __name__ == "__main__":
    fai()
