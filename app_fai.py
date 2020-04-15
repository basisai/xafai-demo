import json
import pickle

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import shap
import matplotlib.pyplot as plt
from aif360.metrics.classification_metric import ClassificationMetric
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

from toolkit import (
    prepare_dataset,
    compute_fairness_metrics,
    get_perf_measure_by_group,
    plot_confusion_matrix_by_group,
)
from constants import *


@st.cache(allow_output_mutation=True)
def load_model(filename):
    return pickle.load(open(filename, "rb"))


@st.cache
def load_data(filename, sample_size=None, random_state=0):
    df = pd.read_csv(filename)
    if sample_size is None:
        return df
    return df.sample(sample_size, random_state=random_state)


@st.cache
def compute_preds(clf, x_val):
    y_prob = clf.predict_proba(x_val)[:, 1]
    return (y_prob > 0.5).astype(int)


@st.cache(allow_output_mutation=True)
def get_clf_metric(grdtruth, predicted, unprivileged_groups, privileged_groups):
    return ClassificationMetric(grdtruth, predicted, unprivileged_groups, privileged_groups)


def color_red(val, threshold=0.2):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if abs(val) > threshold else 'black'
    return 'color: %s' % color
    

def fai():
    max_width = st.sidebar.slider("Set page width", min_value=700, max_value=1500, value=1000, step=20)
    st.markdown(
        f"""
        <style>
        .reportview-container .main .block-container{{
            max-width: {max_width}px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title("Fairness AI Dashboard")
    
    st.sidebar.title("Model and Data Instructions")
    st.sidebar.info("Write your own `load_model`, `load_data` functions.")

    # Load sample, data
    clf = load_model("output/lgb.pkl")
    val = load_data("output/val.csv")
    x_val = val[FEATURES]
    y_val = val[TARGET]

    # Predict on val data
    y_pred = compute_preds(clf, x_val)
    
    # Prepare val dataset
    grdtruth_val = prepare_dataset(x_val, y_val, **BIAS_INFO, **PRIVILEGED_INFO)
    predicted_val = prepare_dataset(x_val, y_pred, **BIAS_INFO, **PRIVILEGED_INFO)
    
    st.header("Model performance")
    st.text(f"Accuracy = {accuracy_score(y_val, y_pred):.4f}")
    st.text(f"Precision = {precision_score(y_val, y_pred):.4f}")
    st.text(f"Recall = {recall_score(y_val, y_pred):.4f}")
    st.code(classification_report(y_val, y_pred))
    
    
    clf_metric = get_clf_metric(grdtruth_val, predicted_val, **PRIVILEGED_INFO)
    
    st.header("Algorithmic fairness metrics")
    st.sidebar.title("Fairness Instructions")
    st.sidebar.info("[placeholder]")
    
    threshold = st.slider("Set fairness threshold", 0., 1., 0.2, 0.05)
    st.write(f"Fairness is when deviation < {threshold}")
    metrics, metrics_others = compute_fairness_metrics(clf_metric, threshold)
    st.dataframe(
        metrics.drop(columns=["Definition"])
        .style.applymap(color_red, subset=["Deviation"])
    )
    st.markdown("Definitions:\n"
                "- Statistical parity: equal proportion of predicted positives\n"
                "- Equal opportunity: equal FNR\n"
                "- Predictive parity: equal PPV")

    
    st.subheader("Performance metrics")
    all_perfs = []
    for metric_name in [
            'TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV', 'FDR', 'FOR', 'ACC',
            'selection_rate', 'precision', 'recall', 'sensitivity',
            'specificity', 'power', 'error_rate']:
        df = get_perf_measure_by_group(clf_metric, metric_name)
        c = alt.Chart(df).mark_bar().encode(
            x=f"{metric_name}:Q",
            y="Group:O",
            tooltip=["Group", metric_name],
        )
        all_perfs.append(c)
    
    all_charts = alt.concat(*all_perfs, columns=2)
    st.altair_chart(all_charts, use_container_width=False)
    
    st.subheader("Confusion matrices")
    st.pyplot(plot_confusion_matrix_by_group(clf_metric))

#     cm = clf_metric.binary_confusion_matrix(privileged=None)
#     source = pd.DataFrame([[0, 0, cm['TN']],
#                            [0, 1, cm['FP']],
#                            [1, 0, cm['FN']],
#                            [1, 1, cm['TP']],
#                           ], columns=["actual values", "predicted values", "count"])
    
#     rects = alt.Chart(source).mark_rect().encode(
#         y='actual values:O',
#         x='predicted values:O',
#         color='count:Q',
#     )
#     text = rects.mark_text(
#         align='center',
#         baseline='middle',
#         dx=0,  # Nudges text to right so it doesn't appear on top of the bar
#     ).encode(
#         text='count:Q'
#     )
#     st.altair_chart(rects + text, use_container_width=False)

    
if __name__ == "__main__":
    fai()
