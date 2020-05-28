import pandas as pd
import altair as alt
import streamlit as st
from sklearn import metrics

from app_utils import load_model, load_data, predict
from constants import FEATURES, TARGET, CONFIG_FAI
from .static_fai import get_fmeasures, plot_hist, plot_fmeasures_bar, color_red, alg_fai
from .toolkit import prepare_dataset, get_perf_measure_by_group


def print_model_perf(y_val, y_pred):
    text = ""
    text += "Model accuracy = {:.4f}\n".format(metrics.accuracy_score(y_val, y_pred))
    text += "Weighted Average Precision = {:.4f}\n".format(metrics.precision_score(y_val, y_pred, average="weighted"))
    text += "Weighted Average Recall = {:.4f}\n\n".format(metrics.recall_score(y_val, y_pred, average="weighted"))
    text += metrics.classification_report(y_val, y_pred, digits=4)
    return text


def fai():
    max_width = 1000
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
                    "- Also set `CONFIG_FAI` in `constants.py`.")
    
    protected_attribute = st.selectbox("Select protected column.", list(CONFIG_FAI.keys()))
    privileged_attribute_values = CONFIG_FAI[protected_attribute]["privileged_attribute_values"]
    unprivileged_attribute_values = CONFIG_FAI[protected_attribute]["unprivileged_attribute_values"]

    # Load sample, data
    clf = load_model("output/lgb.pkl")
    val = load_data("output/valid.csv").fillna(0)  # Fairness does not allow NaNs
    x_val = val[FEATURES]
    y_val = val[TARGET].values

    # Predict on val data
    y_prob = predict(clf, x_val)
    
    st.header("Prediction Distributions")
    cutoff = st.slider("Set probability cutoff", 0., 1., 0.5, 0.01, key="proba")
    y_pred = (y_prob > cutoff).astype(int)
    
    source = pd.DataFrame({
        protected_attribute: x_val[protected_attribute].values,
        "Prediction": y_prob,
    })
    st.altair_chart(plot_hist(source, cutoff), use_container_width=True)
    
    st.header("Model Performance")
    st.text(print_model_perf(y_val, y_pred))

    st.header("Algorithmic Fairness Metrics")    
    fthresh = st.slider("Set fairness threshold", 0., 1., 0.2, 0.05, key="fairness")
    
    # Compute fairness metrics
    fmeasures, clf_metric = get_fmeasures(
        x_val, y_val, y_pred, protected_attribute,
        privileged_attribute_values, unprivileged_attribute_values,
        fthresh=fthresh,
    )
    alg_fai(fmeasures, clf_metric, fthresh)
        
    st.subheader("Notes")
    st.write("**Equal opportunity**:")
    st.latex(r"\frac{\text{FNR}(D=\text{unprivileged})}{\text{FNR}(D=\text{privileged})}")
    st.write("**Predictive parity**:")
    st.latex(r"\frac{\text{PPV}(D=\text{unprivileged})}{\text{PPV}(D=\text{privileged})}")
    st.write("**Statistical parity**:")
    st.latex(r"\frac{\text{Selection Rate}(D=\text{unprivileged})}{\text{Selection Rate}(D=\text{privileged})}")

    
if __name__ == "__main__":
    fai()
