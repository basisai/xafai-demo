import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics

from app_utils import load_model, load_data, predict
from constants import *
from .static_fai import get_fmeasures, plot_hist, plot_fmeasures_bar, color_red, static_fai
from .toolkit import get_perf_measure_by_group, plot_confusion_matrix_by_group
    

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
                    "- Also set `BIAS_INFO` and `PRIVILEGED_INFO` in `constants.py`.")
    
    select_protected = st.selectbox("Select protected column.", list(CONFIG_FAI.keys()))
    
    bias_info = CONFIG_FAI[select_protected]["bias_info"]
    privileged_info = CONFIG_FAI[select_protected]["privileged_info"]
    
    st.header("Bias Information")
    st.subheader("Favourable and Unfavourable Labels")
    st.text("Favourable label: {}".format(bias_info["favorable_label"]))
    st.text("Unfavourable label: {}".format(bias_info["unfavorable_label"]))
    
    st.subheader("Privileged and Unprivileged Groups")
    for k, v in privileged_info.items():
        st.text(f"{k}: {v}")

    # Load sample, data
    clf = load_model("output/lgb.pkl")
    val = load_data("output/val.csv").fillna(0)  # Fairness does not allow NaNs
    x_val = val[FEATURES]
    y_val = val[TARGET].values

    # Predict on val data
    y_prob = predict(clf, x_val)
    
    st.header("Prediction Distributions")
    cutoff = st.slider("Set probability cutoff", 0., 1., 0.5, 0.01, key="proba")
    y_pred = (y_prob > cutoff).astype(int)
    
    source = pd.DataFrame({
        select_protected: x_val[select_protected].values,
        "Predicted Probability": y_prob,
    })
    source["Cutoff"] = cutoff
    hist = plot_hist(source)
    st.altair_chart(hist, use_container_width=True)
    
    
    st.header("Model Performance")
    st.text(f"Model accuracy = {metrics.accuracy_score(y_val, y_pred):.4f}")
    st.text("Weighted Average Precision = {:.4f}".format(metrics.precision_score(y_val, y_pred, average="weighted")))
    st.text("Weighted Average Recall = {:.4f}".format(metrics.recall_score(y_val, y_pred, average="weighted")))
    st.text("." + metrics.classification_report(y_val, y_pred, digits=4))

    
    st.header("Algorithmic Fairness Metrics")
    
    fthresh = st.slider("Set fairness threshold", 0., 1., 0.2, 0.05, key="fairness")
    st.write(f"Fairness is when **ratio is between {1-fthresh:.2f} and {1+fthresh:.2f}**.")
    
    # Compute fairness metrics
    fmeasures, clf_metric = get_fmeasures(x_val, y_val, y_pred, bias_info, privileged_info, fthresh)
    
    chart = plot_fmeasures_bar(fmeasures, fthresh, mode="deviation")
    st.altair_chart(chart, use_container_width=True)
    
    st.dataframe(
        fmeasures[["Metric", "Unprivileged", "Privileged", "Ratio", "Fair?"]]
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
    
    st.header("Notes")
    
    st.write("**Statistical parity**:")
    st.latex(r"\frac{\text{Selection Rate}(D=\text{unprivileged})}{\text{Selection Rate}(D=\text{privileged})}")
    
    st.write("**Equal opportunity**:")
    st.latex(r"\frac{\text{FNR}(D=\text{unprivileged})}{\text{FNR}(D=\text{privileged})}")
    
    st.write("**Equal opportunity**:")
    st.latex(r"\frac{\text{PPV}(D=\text{unprivileged})}{\text{PPV}(D=\text{privileged})}")

    
if __name__ == "__main__":
    fai()
