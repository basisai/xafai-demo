import shap
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

from app_utils import load_model, load_data, compute_shap_values
from constants import *
from xai_fairness.static_xai import (
    get_top_features,
    compute_pdp_isolate,
    pdp_chart,
    pdp_heatmap,
)


def xai():
    st.title("Explainability AI Dashboard")
    
    # Load model, sample data
    clf = load_model("output/lgb_clf.pkl")
    sample = load_data("output/valid.csv", sample_size=3000)
    x_sample = sample[FEATURES]
    
    st.header("SHAP")
    # Compute SHAP values
    shap_values = compute_shap_values(clf, x_sample)
    
    # summarize the effects of all features
    max_display = 15
    st.write("**SHAP Summary Plots of Top Features**")

    source = get_top_features([shap_values], FEATURES, max_display)
    chart = alt.Chart(source).mark_bar().encode(
        alt.X("value", title="mean(|SHAP value|) (average impact on model output magnitude)"),
        alt.Y("feature", title="", sort="-x"),
        alt.Tooltip(["feature", "value"]),
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
    select_feats = st.multiselect("Select two features", NUMERIC_FEATS + CATEGORICAL_FEATS)
    if len(select_feats) > 1:
        feat1, feat2 = select_feats[:2]
        feature1 = CATEGORY_MAP.get(feat1) or feat1
        feature2 = CATEGORY_MAP.get(feat2) or feat2
        pdp_interact_out = compute_pdp_interact(clf, _x_sample, FEATURES, [feature1, feature2])
        st.altair_chart(pdp_heatmap(pdp_interact_out, select_feats[:2]),
                        use_container_width=True)


if __name__ == "__main__":
    xai()
