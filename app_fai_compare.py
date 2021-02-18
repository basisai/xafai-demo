"""
App for fairness comparison.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from xai_fairness.static_fai import (
    get_aif_metric,
    custom_fmeasures,
    plot_confusion_matrix,
    plot_fmeasures_bar,
    color_red,
    fairness_notes,
)
from xai_fairness.toolkit_fai import prepare_dataset, get_perf_measure_by_group

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
    st.write("Absolute fairness is 1. The model is considered fair "
             f"if **ratio is between {1 - fthresh:.2f} and {1 + fthresh:.2f}**.")

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

    st.altair_chart(plot_fmeasures_bar(fmeasures, fthresh), use_container_width=True)
    
    st.dataframe(
        fmeasures[["Metric", "Unprivileged", "Privileged", "Ratio", "Fair?"]]
        .style.applymap(color_red, subset=["Fair?"])
        .format({"Unprivileged": "{:.3f}", "Privileged": "{:.3f}", "Ratio": "{:.3f}"})
    )

    st.subheader("Confusion Matrices")
    cm1 = aif_metric.binary_confusion_matrix(privileged=None)
    c1 = plot_confusion_matrix(cm1, "All")
    st.altair_chart(alt.concat(c1, columns=2), use_container_width=False)
    cm2 = aif_metric.binary_confusion_matrix(privileged=True)
    c2 = plot_confusion_matrix(cm2, "Privileged")
    cm3 = aif_metric.binary_confusion_matrix(privileged=False)
    c3 = plot_confusion_matrix(cm3, "Unprivileged")
    st.altair_chart(c2 | c3, use_container_width=False)

    # if debias:
    #     # Compute original model confusion matrix
    #     orig_y_pred = (y_prob > cutoff).astype(int)
    #     orig_fmeasures, orig_clf_metric = get_fmeasures(
    #         x_val, y_val, orig_y_pred, protected_attribute,
    #         privileged_attribute_values, unprivileged_attribute_values,
    #         fthresh, METRICS_TO_USE,
    #     )
    #
    #     st.header("Comparison before and after mitigation")
    #     for m in METRICS_TO_USE:
    #         source = pd.concat([orig_fmeasures.query(f"Metric == '{m}'"),
    #                             fmeasures.query(f"Metric == '{m}'")])
    #         source["Metric"] = ["1-Before Mitigation", "2-After Mitigation"]
    #
    #         st.write(m)
    #         st.altair_chart(plot_fmeasures_bar(source, fthresh), use_container_width=True)
    #
    #     cm4 = orig_clf_metric.binary_confusion_matrix(privileged=None)
    #     c4a = get_confusion_matrix_chart(cm4, "All: Before Mitigation")
    #     c4b = get_confusion_matrix_chart(cm1, "All: After Mitigation")
    #     st.altair_chart(c4a | c4b, use_container_width=False)
    #
    #     cm5 = orig_clf_metric.binary_confusion_matrix(privileged=True)
    #     c5a = get_confusion_matrix_chart(cm5, "Privileged: Before Mitigation")
    #     c5b = get_confusion_matrix_chart(cm2, "Privileged: After Mitigation")
    #     st.altair_chart(c5a | c5b, use_container_width=False)
    #
    #     cm6 = orig_clf_metric.binary_confusion_matrix(privileged=False)
    #     c6a = get_confusion_matrix_chart(cm6, "Unprivileged: Before Mitigation")
    #     c6b = get_confusion_matrix_chart(cm3, "Unprivileged: After Mitigation")
    #     st.altair_chart(c6a | c6b, use_container_width=False)

    st.header("Annex")
    st.subheader("Performance Metrics")
    all_perfs = []
    for metric_name in [
            'TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV', 'FDR', 'FOR', 'ACC',
            'selection_rate', 'precision', 'recall', 'sensitivity',
            'specificity', 'power', 'error_rate']:
        df = get_perf_measure_by_group(aif_metric, metric_name)
        c = alt.Chart(df).mark_bar().encode(
            x=f"{metric_name}:Q",
            y="Group:O",
            tooltip=["Group", metric_name],
        )
        all_perfs.append(c)
    
    all_charts = alt.concat(*all_perfs, columns=1)
    st.altair_chart(all_charts, use_container_width=False)

    st.subheader("Notes")
    fairness_notes()


def chart_cm_comparison(orig_clf_metric, clf_metric, privileged, title):
    cm1 = orig_clf_metric.binary_confusion_matrix(privileged=privileged)
    cm2 = clf_metric.binary_confusion_matrix(privileged=privileged)
    c1 = get_confusion_matrix_chart(cm1, f"{title}: Before Mitigation")
    c2 = get_confusion_matrix_chart(cm2, f"{title}: After Mitigation")
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
    st.write("Absolute fairness is 1. The model is considered fair "
             f"if **ratio is between {1 - fthresh:.2f} and {1 + fthresh:.2f}**.")

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
    orig_fmeasures = compute_fairness_measures(orig_aif_metric)
    orig_fmeasures["Fair?"] = orig_fmeasures["Ratio"].apply(
        lambda x: "Yes" if np.abs(x - 1) < fthresh else "No")

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

    for m in METRICS_TO_USE:
        source = pd.concat([orig_fmeasures.query(f"Metric == '{m}'"),
                            fmeasures.query(f"Metric == '{m}'")])
        source["Metric"] = ["1-Before Mitigation", "2-After Mitigation"]

        st.write(m)
        st.altair_chart(plot_fmeasures_bar(source, fthresh), use_container_width=True)

#     st.altair_chart(chart_cm_comparison(orig_clf_metric, clf_metric, None, "All"),
#                     use_container_width=False)
#     st.altair_chart(chart_cm_comparison(orig_clf_metric, clf_metric, True, "Privileged"),
#                     use_container_width=False)
#     st.altair_chart(chart_cm_comparison(orig_clf_metric, clf_metric, False, "Unprivileged"),
#                     use_container_width=False)

    
if __name__ == "__main__":
    fai()
