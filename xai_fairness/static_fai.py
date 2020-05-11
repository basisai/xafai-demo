import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
from aif360.metrics.classification_metric import ClassificationMetric
from sklearn import metrics

from app_utils import load_model, load_data
from constants import *
from .toolkit import (
    prepare_dataset,
    compute_fairness_metrics,
    get_perf_measure_by_group,
    plot_confusion_matrix_by_group,
)


@st.cache(allow_output_mutation=True)
def get_fmeasures(x_val, y_val, y_pred, bias_info, privileged_info, fthresh=0.2):
    grdtruth_val = prepare_dataset(x_val, y_val, **bias_info, **privileged_info)
    predicted_val = prepare_dataset(x_val, y_pred, **bias_info, **privileged_info)

    clf_metric = ClassificationMetric(grdtruth_val, predicted_val, **privileged_info)
    fmeasures = compute_fairness_metrics(clf_metric)
    fmeasures["Fair?"] = fmeasures["Ratio"].apply(lambda x: "Yes" if np.abs(x - 1) < fthresh else "No")
    return fmeasures, clf_metric


def plot_hist(source):
    var = source.columns[0]
    base = alt.Chart(source)
    chart = base.mark_area(
        opacity=0.5, interpolate="step",
    ).encode(
        alt.X("Predicted Probability:Q", bin=alt.Bin(maxbins=20), title="Predicted Probability"),
        alt.Y("count()", stack=None),
        alt.Color(f"{var}:N"),
    )
    rule = base.mark_rule(color="red").encode(
        alt.X("Cutoff:Q"),
        size=alt.value(2),
    )
    mean = base.mark_rule().encode(
        alt.X("mean(Predicted Probability):Q"),
        alt.Color(f"{var}:N"),
        size=alt.value(2),
    )
    return chart + rule + mean


def plot_fmeasures_bar(df0, threshold, mode=None):
    df = df0.copy()
    if mode == "deviation":
        df["Deviation"] = df["Ratio"] - 1
        df["min_val"] = -threshold
        df["max_val"] = threshold
    else:
        df["min_val"] = 1 - threshold
        df["max_val"] = 1 + threshold
    
    base = alt.Chart(df)
    if mode == "deviation":
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
    else:
        bar = base.mark_bar().encode(
            alt.X("Ratio:Q", scale=alt.Scale(domain=[0., 2.])),
            color=alt.condition(f"abs(datum.Ratio) - 1 > {threshold}",
                                alt.value("#FF0D57"), alt.value("#1E88E5")),
            y="Metric:O",
            tooltip=["Metric", "Ratio"],
        )
        rule1 = base.mark_rule(color="black").encode(
            alt.X("min_val:Q"),
            size=alt.value(2),
        )
        rule2 = base.mark_rule(color="black").encode(
            alt.X("max_val:Q", title="Ratio"),
            size=alt.value(2),
        )
    return bar + rule1 + rule2


def color_red(x):
    return "color: red" if x == "No" else "color: black"


def static_fai(fmeasures, clf_metric, x_val, y_val, y_pred, y_prob, bias_info, privileged_info, cutoff, fthresh):
    st.subheader("Bias Information")
    st.write("Favourable and Unfavourable Labels")
    st.text("Favourable label: {}".format(bias_info["favorable_label"]))
    st.text("Unfavourable label: {}".format(bias_info["unfavorable_label"]))
    
    st.write("Privileged and Unprivileged Groups")
    for k, v in privileged_info.items():
        st.text(f"{k}: {v}")    
    
    st.subheader("Prediction Distributions")
    select_protected = bias_info["protected_columns"][0]
    
    source = pd.DataFrame({
        select_protected: x_val[select_protected].values,
        "Predicted Probability": y_prob,
    })
    source["Cutoff"] = cutoff
    hist = plot_hist(source)
    st.altair_chart(hist, use_container_width=True)
    
    st.subheader("Algorithmic Fairness Metrics")
    st.write(f"Fairness is when **ratio is between {1-fthresh:.2f} and {1+fthresh:.2f}**.")
    
    chart = plot_fmeasures_bar(fmeasures.iloc[:3], fthresh)
    st.altair_chart(chart, use_container_width=True)
    
    st.dataframe(
        fmeasures[["Metric", "Unprivileged", "Privileged", "Ratio", "Fair?"]].iloc[:3]
        .style.applymap(color_red, subset=["Fair?"])
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
    
    all_charts = alt.concat(*all_perfs, columns=1)
    st.altair_chart(all_charts, use_container_width=False)
    
    st.subheader("Confusion Matrices")
    st.pyplot(plot_confusion_matrix_by_group(clf_metric), figsize=(8, 6))
    