import json
import pickle

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import shap
import matplotlib.pyplot as plt
from aif360.metrics.classification_metric import ClassificationMetric

from toolkit import pdp_plot, pdp_interact_plot
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
def compute_shap_values(clf, x_sample):
    # Use the relevant explainer
    explainer = shap.TreeExplainer(clf)
    return explainer.shap_values(x_sample)[1]
    

def main():
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
    
    st.title("Explanability AI Dashboard")

    st.sidebar.title("Model and Data Instructions")
    st.sidebar.info("Write your own `load_model`, `load_data` functions.")
    
    # Load model, sample data
    clf = load_model("output/lgb.pkl")
    sample = load_data("output/train.csv", sample_size=3000)
    x_sample = sample[FEATURES]
    
    st.header("SHAP")
    st.sidebar.title("SHAP Instructions")
    st.sidebar.info(
        "Set the relevant explainer in `compute_shap_values` for your model.\n"
        "shap.TreeExplainer works with tree models.\n"
        "shap.DeepExplainer works with Deep Learning models.\n"
        "shap.KernelExplainer works with all models, though it is slower than other Explainers and it offers an approximation rather than exact Shap values."
        "See Explainers[https://shap.readthedocs.io/en/latest/#explainers] for more details"
    )
    
    # Compute SHAP values
    shap_values = compute_shap_values(clf, x_sample)
    
    # summarize the effects of all features
    st.subheader("SHAP summary plot")
    max_display = st.slider("Select number of top features to show", 10, 30, 10)
    
    shap.summary_plot(shap_values, plot_type="bar", feature_names=FEATURES,
                      max_display=max_display, plot_size=0.25, show=False)
    plt.gcf().tight_layout()
    st.pyplot()
    
    shap.summary_plot(shap_values, x_sample, feature_names=FEATURES,
                      max_display=max_display, plot_size=0.25, show=False)
    plt.gcf().tight_layout()
    st.pyplot()
    
    st.subheader("SHAP dependence contribution plots")
    features = st.multiselect("Select two features", FEATURES, key="shap")
    if len(features) > 1:
        feat1, feat2 = features[:2]
        shap.dependence_plot(feat1, shap_values, x_sample, interaction_index=feat2, show=False)
        plt.gcf().tight_layout()
        st.pyplot()
    
    st.header("Partial dependence plots")
    st.sidebar.title("PDPbox Instructions")
    st.sidebar.info("[placeholder]")
    
    st.subheader("Partial dependence plots")
    feature_name = st.selectbox("Select feature", NUMERIC_FEATS + CATEGORICAL_FEATS)
    if feature_name in CATEGORICAL_FEATS:
        feature = CATEGORY_MAP[feature_name]
        st.pyplot(pdp_plot(clf, x_sample, FEATURES, feature, feature_name).tight_layout())
    else:
        feature = feature_name
        st.pyplot(pdp_plot(clf, x_sample, FEATURES, feature, feature_name,
                           num_grid_points=12, show_percentile=True))
    
    st.subheader("Partial dependence interaction plots")
    features = st.multiselect("Select two features", CATEGORICAL_FEATS + NUMERIC_FEATS, key="pdp")
    if len(features) > 1:
        feat1, feat2 = features[:2]
        if feat1 in CATEGORICAL_FEATS:
            feat1 = CATEGORY_MAP[feat1]
        if feat2 in CATEGORICAL_FEATS:
            feat2 = CATEGORY_MAP[feat2]
        st.pyplot(pdp_interact_plot(clf, x_sample, FEATURES, feat1, feat2))


if __name__ == "__main__":
    main()
