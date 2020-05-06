import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
from aif360.metrics.classification_metric import ClassificationMetric
from sklearn import metrics

from toolkit import (
    prepare_dataset,
    compute_fairness_metrics,
    get_perf_measure_by_group,
    plot_confusion_matrix_by_group,
)
from constants import *
from app_xai import load_model, load_data


@st.cache
def compute_proba(clf, x_val):
    return clf.predict_proba(x_val)[:, 1]
     

@st.cache(allow_output_mutation=True)
def get_clf_metric(grdtruth, predicted, unprivileged_groups, privileged_groups):
    return ClassificationMetric(grdtruth, predicted, unprivileged_groups, privileged_groups)


def plot_hist(source):
    base = alt.Chart(source)
    chart = base.mark_area(
        opacity=0.5, interpolate="step",
    ).encode(
        alt.X("Predicted Probability:Q", bin=alt.Bin(maxbins=20), title="Predicted Probability"),
        alt.Y("count()", stack=None),
        alt.Color("Actual Target:N"),
    )
    rule = base.mark_rule(color="red").encode(
        alt.X("Cutoff:Q"),
        size=alt.value(2),
    )
    mean = base.mark_rule().encode(
        alt.X("mean(Predicted Probability):Q"),
        alt.Color("Actual Target:N"),
        size=alt.value(2),
    )
    return chart + rule + mean


def plot_fmeasures_bar(df0, threshold):
    df = df0.copy()
    df["min_val"] = -threshold
    df["max_val"] = threshold
    
    base = alt.Chart(df)
    bar = base.mark_bar().encode(
        alt.X("Deviation:Q", scale=alt.Scale(domain=[-1., 1.])),
        color=alt.condition(f"abs(datum.Deviation) > {threshold}",
                            alt.value("#FF0D57"), alt.value("#1E88E5")),
        y="Metric:O",
        tooltip=["Metric", "Ratio", "Deviation"],
    )
    rule1 = base.mark_rule(color="red").encode(
        alt.X("min_val:Q"),
        size=alt.value(2),
    )
    rule2 = base.mark_rule(color="red").encode(
        alt.X("max_val:Q", title="Deviation"),
        size=alt.value(2),
    )
    return bar + rule1 + rule2


def color_red(val, threshold=0.2):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if abs(val) > threshold else 'black'
    return 'color: %s' % color
    

def fai():
    max_width = 1000  #st.sidebar.slider("Set page width", min_value=700, max_value=1500, value=1000, step=20)
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
    
    st.sidebar.title("Instructions")
    st.sidebar.info("- See `Global explainability` page for instructions on model and data.\n"
                    "- Also set `BIAS_INFO` and `PRIVILEGED_INFO` in `constants.py`.\n"
                    "- Algorithmic fairness is only for binary classification.")
    
    st.header("Bias Information")
    st.subheader("Favourable and Unfavourable labels")
    st.text("Favourable label: {}".format(BIAS_INFO["favorable_label"]))
    st.text("Unfavourable label: {}".format(BIAS_INFO["unfavorable_label"]))
    
    st.subheader("Protected Feature Columns")
    st.text(BIAS_INFO["protected_columns"])
    
    st.subheader("Privileged and Unprivileged groups")
    for k, v in PRIVILEGED_INFO.items():
        st.text(f"{k}: {v}")

    # Load sample, data
    clf = load_model("output/lgb.pkl")
    val = load_data("output/val.csv")
    x_val = val[FEATURES]
    y_val = val[TARGET].values

    # Predict on val data
    y_prob = compute_proba(clf, x_val)
    
    st.header("Prediction Distributions")
    cutoff = st.slider("Set probability cutoff", 0., 1., 0.5, 0.01, key="proba")
    y_pred = (y_prob > cutoff).astype(int)
    
    source = pd.DataFrame({
        "Actual Target": y_val,
        "Predicted Probability": y_prob,
    })
    source["Cutoff"] = cutoff
    hist = plot_hist(source)
    st.altair_chart(hist, use_container_width=True)
    
    st.header("Model Performance")
    st.text(f"Model accuracy = {metrics.accuracy_score(y_val, y_pred):.4f}")
    st.text("." + metrics.classification_report(y_val, y_pred, digits=4))
    
    # Prepare val dataset
    grdtruth_val = prepare_dataset(x_val, y_val, **BIAS_INFO, **PRIVILEGED_INFO)
    predicted_val = prepare_dataset(x_val, y_pred, **BIAS_INFO, **PRIVILEGED_INFO)
    
    clf_metric = get_clf_metric(grdtruth_val, predicted_val, **PRIVILEGED_INFO)
    
    st.header("Algorithmic Fairness Metrics")
    
    fthresh = st.slider("Set fairness threshold", 0., 1., 0.2, 0.05, key="fairness")
    st.write(f"Fairness is when **absolute deviation < {fthresh}**.")
    
    fmeasures = compute_fairness_metrics(clf_metric)
    
    chart = plot_fmeasures_bar(fmeasures.iloc[:3], fthresh)
    st.altair_chart(chart, use_container_width=True)
    
    st.dataframe(
        fmeasures.iloc[:3]
        .style.applymap(lambda x: color_red(x, fthresh), subset=['Deviation'])
    )
    
    st.subheader("Performance Metrics")
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
    
    st.subheader("Confusion Matrices")
    st.pyplot(plot_confusion_matrix_by_group(clf_metric), figsize=(8, 6))

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
    
    st.header("Appendix")
    
    st.write("**Statistical parity**:")
    st.latex(r"\frac{\text{Selection Rate}(D=\text{unprivileged})}{\text{Selection Rate}(D=\text{privileged})}")
    
    st.write("**Equal opportunity**:")
    st.latex(r"\frac{\text{FNR}(D=\text{unprivileged})}{\text{FNR}(D=\text{privileged})}")
    
    st.write("**Equal opportunity**:")
    st.latex(r"\frac{\text{PPV}(D=\text{unprivileged})}{\text{PPV}(D=\text{privileged})}")

    
if __name__ == "__main__":
    fai()
