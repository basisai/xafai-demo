import shap
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

from app_utils import load_model, load_data, compute_shap_values
from constants import *
from xai_fairness.static_xai import (
    get_top_features,
    make_source_dp,
    dependence_chart,
    compute_pdp_isolate,
    compute_pdp_interact,
    pdp_chart,
    pdp_heatmap,
)


def xai():
    st.title("Explainability AI Dashboard")
    
    # Load model, valid data
    clf = load_model("models/lgb_clf.pkl")
    valid = load_data("data/valid.csv", sample_size=3000)
    x_valid = valid[FEATURES]
    
    st.header("SHAP")
    # Compute SHAP values
    shap_values = compute_shap_values(clf, x_valid)
    
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
                      x_valid,
                      feature_names=FEATURES,
                      max_display=max_display,
                      plot_size=[12, 6],
                      show=False)
    plt.gcf().tight_layout()
    st.pyplot()

    st.subheader("SHAP Dependence Contribution Plot")
    shap_feat = st.selectbox("Select feature", FEATURES[-4:] + FEATURES[:-4])
    source = make_source_dp(shap_values, x_valid.values, FEATURES, shap_feat)
    st.altair_chart(dependence_chart(source, shap_feat), use_container_width=False)

    st.subheader("SHAP Dependence Contribution Interaction Plot")
    shap_feats = st.multiselect("Select two features", FEATURES[-4:] + FEATURES[:-4])
    if len(shap_feats) > 1:
        shap_feat1, shap_feat2 = shap_feats[:2]
        fig, ax = plt.subplots(figsize=(12, 6))
        shap.dependence_plot(shap_feat1, shap_values, x_valid,
                             interaction_index=shap_feat2,
                             ax=ax, show=False)
        st.pyplot()

    st.header("Partial Dependence Plot")
    # PDPbox does not allow NaNs
    _x_valid = x_valid.fillna(0)

    st.subheader("Partial Dependence Plot")
    pdp_feat = st.selectbox("Select feature", NUMERIC_FEATS + CATEGORICAL_FEATS)
    feature = CATEGORY_MAP.get(pdp_feat) or pdp_feat
    pdp_isolate_out = compute_pdp_isolate(clf, _x_valid, FEATURES, feature)
    st.altair_chart(pdp_chart(pdp_isolate_out, pdp_feat), use_container_width=False)

    st.subheader("Partial Dependence Interaction Plot")
    pdp_feats = st.multiselect("Select two features", NUMERIC_FEATS + CATEGORICAL_FEATS)
    if len(pdp_feats) > 1:
        pdp_feat1, pdp_feat2 = pdp_feats[:2]
        feature1 = CATEGORY_MAP.get(pdp_feat1) or pdp_feat1
        feature2 = CATEGORY_MAP.get(pdp_feat2) or pdp_feat2
        pdp_interact_out = compute_pdp_interact(
            clf, _x_valid, FEATURES, [feature1, feature2])
        st.altair_chart(pdp_heatmap(pdp_interact_out, pdp_feats[:2]),
                        use_container_width=False)


if __name__ == "__main__":
    xai()
