import numpy as np
import pandas as pd
import shap
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

from app_utils import load_model, load_data, compute_shap_values
from constants import *
from .static_xai import get_top_features, compute_pdp_isolate, pdp_chart, compute_pdp_interact, pdp_heatmap


def xai():
    max_width = 1000  # st.sidebar.slider("Set page width", min_value=700, max_value=1500, value=1000, step=20)
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
    
    st.title("Explainability AI Dashboard")

    st.sidebar.title("Model and Data Instructions")
    st.sidebar.info(
        "- Write your own `load_model`, `load_data` functions.\n"
        "- Model must be a fitted `sklearn` model.\n"
        "- Sample data must be a `pandas.DataFrame`.\n"
        "- Feature names and a category map for one-hot encoded features must be "
        "furnished in `constants.py`."
    )
    
    # Load model, sample data
    clf = load_model("output/lgb.pkl")
    sample = load_data("output/valid.csv", sample_size=3000)
    x_sample = sample[FEATURES]
    
    st.header("SHAP")
    st.sidebar.title("SHAP Instructions")
    st.sidebar.info(
        "Set the relevant explainer in `compute_shap_values` for your model.\n"
        "- `shap.TreeExplainer` works with tree models.\n"
        "- `shap.DeepExplainer` works with Deep Learning models.\n"
        "- `shap.KernelExplainer` works with all models, though it is slower than "
        "other Explainers and it offers an approximation rather than exact "
        "Shap values.\n\n"
        "See [Explainers](https://shap.readthedocs.io/en/latest/#explainers) for more details."
    )
    
    # Compute SHAP values
    shap_values = compute_shap_values(clf, x_sample)
    
    # summarize the effects of all features
    max_display = 15
    st.write("**SHAP Summary Plots of Top Features**")

    source = get_top_features([shap_values], FEATURES, max_display)
    chart = alt.Chart(source).mark_bar().encode(
        x=alt.X("value", title="mean(|SHAP value|) (average impact on model output magnitude)"),
        y=alt.Y("feature", title="", sort="-x"),
        tooltip=["feature", "value"],
    )
    st.altair_chart(chart, use_container_width=True)

    shap.summary_plot(shap_values,
                      x_sample,
                      feature_names=FEATURES,
                      max_display=max_display,
                      plot_size=[12, 6],
                      show=False)
    plt.gcf().tight_layout()
    st.pyplot()
    
    st.subheader("SHAP Dependence Contribution Plots")
    feat1 = st.selectbox("Select feature", FEATURES)
    feat2 = st.selectbox("Select interacting feature", FEATURES)
    fig, ax = plt.subplots(figsize=(12, 6))
    shap.dependence_plot(feat1, shap_values, x_sample, interaction_index=feat2,
                         ax=ax, show=False)
    st.pyplot()
    
    
    st.header("Partial Dependence Plots")
    # PDPbox does not allow NaNs
    _x_sample = x_sample.fillna(0)
    
    st.subheader("Partial Dependence Plots")
    feature_name = st.selectbox("Select feature", NUMERIC_FEATS + CATEGORICAL_FEATS)
    
    feature = CATEGORY_MAP.get(feature_name) or feature_name
    pdp_isolate_out = compute_pdp_isolate(clf, _x_sample, FEATURES, feature)
    st.altair_chart(pdp_chart(pdp_isolate_out, feature_name), use_container_width=True)
    
    st.subheader("Partial Dependence Interaction Plots")
    feature_names = st.multiselect("Select two features", NUMERIC_FEATS + CATEGORICAL_FEATS, key="pdp")
    if len(feature_names) > 1:
        feature_name1, feature_name2 = feature_names[:2]
        feature1 = CATEGORY_MAP.get(feature_name1) or feature_name1
        feature2 = CATEGORY_MAP.get(feature_name2) or feature_name2
        pdp_interact_out = compute_pdp_interact(clf, _x_sample, FEATURES, [feature1, feature2])
        st.altair_chart(pdp_heatmap(pdp_interact_out, feature_names[:2]), use_container_width=True)


if __name__ == "__main__":
    xai()
