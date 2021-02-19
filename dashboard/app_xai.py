"""
App for global XAI.
"""
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from xai_fairness.toolkit_xai import (
    shap_summary_plot,
    shap_dependence_plot,
)
from xai_fairness.static_xai import (
    make_source_dp,
    dependence_chart,
    compute_pdp_isolate,
    compute_pdp_interact,
    pdp_chart,
    pdp_heatmap,
)

from data.constants import (
    FEATURES, TARGET_CLASSES, NUMERIC_FEATS, CATEGORICAL_FEATS, CATEGORY_MAP)
from data.utils import xai_data

MAX_DISPLAY = 15


def _rank_features(shap_values, corrs, feature_names):
    shap_summary_df = pd.DataFrame({
        "feature": feature_names,
        "mas_value": np.abs(shap_values).mean(axis=0),
        "corrcoef": corrs,
    })
    shap_summary_df = shap_summary_df.sort_values(
        "mas_value", ascending=False, ignore_index=True)
    return shap_summary_df


def xai():
    st.title("Global XAI")

    # Load model, valid data, SHAP values
    clf, x_valid, all_shap_values, all_corrs = xai_data()

    idx = 0
    if TARGET_CLASSES is not None and len(TARGET_CLASSES) > 2:
        idx = st.selectbox(
            "Select class", list(range(len(TARGET_CLASSES))), 1,
            format_func=lambda i: TARGET_CLASSES[i])

    st.subheader("SHAP Summary Plots of Top Features")
    source = (
        _rank_features(all_shap_values[idx], all_corrs[idx], FEATURES)
        .iloc[:MAX_DISPLAY]
    )
    source["corr"] = source["corrcoef"].apply(lambda x: "positive" if x > 0 else "negative")
    chart = alt.Chart(source).mark_bar().encode(
        alt.X("mas_value:Q", title="mean(|SHAP value|) (average impact on model output magnitude)"),
        alt.Y("feature:N", title="", sort="-x"),
        alt.Color("corr:N", scale=alt.Scale(
            domain=["positive", "negative"], range=["#FF0D57", "#1E88E5"])),
        alt.Tooltip(["feature", "mas_value"]),
    )
    st.altair_chart(chart, use_container_width=True)

    fig = shap_summary_plot(
        all_shap_values[idx],
        x_valid,
        feature_names=FEATURES,
        max_display=MAX_DISPLAY,
        plot_size=[12, 6],
        show=False,
    )
    st.pyplot(fig)

    st.subheader("SHAP Dependence Contribution Plot")
    _feat_options = FEATURES[-4:] + FEATURES[:-4]
    shap_feat = st.selectbox("Select feature", _feat_options)
    source = make_source_dp(all_shap_values[idx], x_valid.values, FEATURES, shap_feat)
    st.altair_chart(dependence_chart(source, shap_feat), use_container_width=False)

    st.subheader("SHAP Dependence Contribution Interaction Plot")
    shap_feats = st.multiselect("Select two features", _feat_options)
    if len(shap_feats) > 1:
        shap_feat1, shap_feat2 = shap_feats[:2]
        fig = shap_dependence_plot(
            shap_feat1,
            all_shap_values[idx],
            x_valid,
            interaction_index=shap_feat2,
        )
        st.pyplot(fig)

    _x_valid = x_valid.fillna(0)  # PDPbox does not allow NaNs

    st.subheader("PDPbox Partial Dependence Plot")
    pdp_feat = st.selectbox("Select feature", NUMERIC_FEATS + CATEGORICAL_FEATS)
    feature = CATEGORY_MAP.get(pdp_feat) or pdp_feat
    pdp_isolate_out = compute_pdp_isolate(clf, _x_valid, FEATURES, feature)
    st.altair_chart(pdp_chart(pdp_isolate_out, pdp_feat), use_container_width=False)

    st.subheader("PDPbox Partial Dependence Interaction Plot")
    pdp_feats = st.multiselect("Select two features", NUMERIC_FEATS + CATEGORICAL_FEATS)
    if len(pdp_feats) > 1:
        pdp_feat1, pdp_feat2 = pdp_feats[:2]
        feature1 = CATEGORY_MAP.get(pdp_feat1) or pdp_feat1
        feature2 = CATEGORY_MAP.get(pdp_feat2) or pdp_feat2
        pdp_interact_out = compute_pdp_interact(
            clf, _x_valid, FEATURES, [feature1, feature2])
        st.altair_chart(
            pdp_heatmap(pdp_interact_out, pdp_feats[:2]), use_container_width=False)


if __name__ == "__main__":
    xai()
