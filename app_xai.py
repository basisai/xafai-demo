import shap
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

from app_utils import load_model, load_data, compute_shap_values
from constants import FEATURES, TARGET_CLASSES, NUMERIC_FEATS, CATEGORICAL_FEATS, CATEGORY_MAP
from xai_fairness.toolkit import compute_corrcoef
from xai_fairness.static_xai import (
    get_top_features,
    make_source_dp,
    dependence_chart,
    compute_pdp_isolate,
    compute_pdp_interact,
    pdp_chart,
    pdp_heatmap,
)

MAX_DISPLAY = 15


def xai():
    st.title("Explainability AI Dashboard")

    # Load model, valid data
    clf = load_model("models/lgb_clf.pkl")
    valid = load_data("data/valid.csv", sample_size=3000)
    x_valid = valid[FEATURES]

    # Compute SHAP values
    all_shap_values = compute_shap_values(clf, x_valid)
    all_corrs = compute_corrcoef(x_valid, all_shap_values)

    select_class = st.selectbox("Select class", TARGET_CLASSES, 1)
    idx = TARGET_CLASSES.index(select_class)

    st.header("SHAP")
    # summarize the effects of all features
    st.write("**SHAP Summary Plots of Top Features**")
    source = (
        get_top_features(all_shap_values[idx], all_corrs[idx], FEATURES)
        .iloc[:MAX_DISPLAY]
    )
    source["corr"] = source["corrcoef"].apply(lambda x: "positive" if x > 0 else "negative")
    chart = alt.Chart(source).mark_bar().encode(
        alt.X("value:Q", title="mean(|SHAP value|) (average impact on model output magnitude)"),
        alt.Y("feature:N", title="", sort="-x"),
        alt.Color("corr:N", scale=alt.Scale(
            domain=["positive", "negative"], range=["#FF0D57", "#1E88E5"])),
        alt.Tooltip(["feature", "value"]),
    )
    st.altair_chart(chart, use_container_width=True)

    shap.summary_plot(all_shap_values[idx],
                      x_valid,
                      feature_names=FEATURES,
                      max_display=MAX_DISPLAY,
                      plot_size=[12, 6],
                      show=False)
    plt.gcf().tight_layout()
    st.pyplot()

    st.subheader("SHAP Dependence Contribution Plot")
    shap_feat = st.selectbox("Select feature", FEATURES[-4:] + FEATURES[:-4])
    source = make_source_dp(all_shap_values[idx], x_valid.values, FEATURES, shap_feat)
    st.altair_chart(dependence_chart(source, shap_feat), use_container_width=False)

    st.subheader("SHAP Dependence Contribution Interaction Plot")
    shap_feats = st.multiselect("Select two features", FEATURES[-4:] + FEATURES[:-4])
    if len(shap_feats) > 1:
        shap_feat1, shap_feat2 = shap_feats[:2]
        fig, ax = plt.subplots(figsize=(12, 6))
        shap.dependence_plot(shap_feat1,
                             all_shap_values[idx],
                             x_valid,
                             interaction_index=shap_feat2,
                             ax=ax,
                             show=False)
        st.pyplot()

    st.header("Partial Dependence Plot")
    _x_valid = x_valid.fillna(0)  # PDPbox does not allow NaNs

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
