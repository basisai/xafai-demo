import numpy as np
import pandas as pd
import shap
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
from pdpbox import pdp

from app_utils import load_model, load_data, compute_shap_values
from constants import *


@st.cache(allow_output_mutation=True)
def compute_pdp_isolate(model, dataset, feature):
    pdp_isolate_out = pdp.pdp_isolate(
        model=model,
        dataset=dataset,
        model_features=FEATURES,
        feature=feature,
        num_grid_points=15,
    )
    return pdp_isolate_out


def pdp_chart(pdp_isolate_out, feature_name):
    """Plot pdp charts."""
    source = pd.DataFrame({
        "feature": pdp_isolate_out.feature_grids,
        "value": pdp_isolate_out.pdp,
    })
    
    if pdp_isolate_out.feature_type == "numeric":
        chart = alt.Chart(source).mark_line().encode(
            x=alt.X("feature", title=feature_name),
            y=alt.Y("value", title=""),
            tooltip=["feature", "value"],
        )
    else:
        source["feature"] = source["feature"].astype(str)
        chart = alt.Chart(source).mark_bar().encode(
            x=alt.X("value", title=""),
            y=alt.Y("feature", title=feature_name, sort="-x"),
            tooltip=["feature", "value"],
        )
        
    return chart


@st.cache(allow_output_mutation=True)
def compute_pdp_interact(model, dataset, features):
    pdp_interact_out = pdp.pdp_interact(
        model=model,
        dataset=dataset,
        model_features=FEATURES,
        features=features,
    )
    return pdp_interact_out


def pdp_heatmap(pdp_interact_out, feature_names):
    source = pdp_interact_out.pdp

    for i in [0, 1]:
        if pdp_interact_out.feature_types[i] == "onehot":
            value_vars = pdp_interact_out.feature_grids[i]
            id_vars = list(set(source.columns) - set(value_vars))
            source = pd.melt(source, value_vars=value_vars,
                             id_vars=id_vars, var_name=feature_names[i])
            source = source[source["value"] == 1].drop(columns=["value"])

        elif pdp_interact_out.feature_types[i] == "binary":
            source[feature_names[i]] = source[feature_names[i]].astype(str)

    chart = alt.Chart(source).mark_rect().encode(
        x=feature_names[0],
        y=feature_names[1],
        color='preds',
        tooltip=feature_names + ["preds"]
    )
    return chart


def xai_summary(x_sample, shap_values, max_display):
    st.subheader("SHAP Summary Plots of Top Features")
    shap.summary_plot(shap_values, plot_type="bar", feature_names=FEATURES,
                      max_display=max_display, plot_size=[12, 6], show=False)
    plt.gcf().tight_layout()
    st.pyplot()
    
    shap.summary_plot(shap_values, x_sample, feature_names=FEATURES,
                      max_display=max_display, plot_size=[12, 6], show=False)
    plt.gcf().tight_layout()
    st.pyplot()
    
    
def static_xai(clf, x_sample, shap_values, top_features):
    st.subheader("SHAP Dependence Contribution Plots")
    
    for i in range(5):
        for j in range(i + 1, 5):
            fig, ax = plt.subplots(figsize=(12, 6))
            shap.dependence_plot(top_features[i], shap_values, x_sample, interaction_index=top_features[j],
                                 ax=ax, show=False)
            st.pyplot()
    
    st.subheader("Partial Dependence Plots of Top Features")
    # PDPbox does not allow NaNs
    _x_sample = x_sample.fillna(0)
    rev_category_map = {e: k for k, v in CATEGORY_MAP.items() for e in v}
    _top_features = []
    for feature_name in top_features:
        if feature_name in rev_category_map.keys() and rev_category_map[feature_name] not in _top_features:
            _top_features.append(rev_category_map[feature_name])
        else:
            _top_features.append(feature_name)

    for feature_name in _top_features:
        feature = CATEGORY_MAP.get(feature_name) or feature_name
        pdp_isolate_out = compute_pdp_isolate(clf, _x_sample, feature)
        st.altair_chart(pdp_chart(pdp_isolate_out, feature_name), use_container_width=True)
        