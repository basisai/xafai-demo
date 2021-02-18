"""
App for FAI.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from xai_fairness.static_fai import (
    binarize,
    get_aif_metric,
    custom_fmeasures,
    alg_fai,
    fairness_notes,
    plot_hist,
)

from data.constants import FEATURES, TARGET, TARGET_CLASSES, PROTECTED_FEATURES
from data.utils import fai_data, print_model_perf


def fai(version=1):
    st.title("Fairness")

    # Load sample, data.
    x_fai, y_valid, y_score = fai_data()

    protected_attribute = st.selectbox("Select protected feature.", list(PROTECTED_FEATURES.keys()))

    if version == 1:
        y_pred = (y_score[:, 1] > 0.5).astype(int)

        select_class = st.selectbox(
            "Select target class to be the positive class.", TARGET_CLASSES, 1)
        true_class = binarize(y_valid, select_class)
        pred_class = binarize(y_pred, select_class)
    else:
        st.header("Prediction Distributions")
        y_prob = y_score[:, 1]
        cutoff = st.slider("Set probability cutoff.", 0., 1., 0.5, 0.01, key="proba")
        y_pred = (y_prob > cutoff).astype(int)

        source = pd.DataFrame({
            protected_attribute: x_fai[protected_attribute].values,
            "Prediction": y_prob,
        })
        st.altair_chart(plot_hist(source, cutoff), use_container_width=True)

        st.header("Model Performance")
        st.text(print_model_perf(y_valid, y_pred))

        true_class = y_valid
        pred_class = y_pred

    st.header("Algorithmic Fairness Metrics")    
    fthresh = st.slider("Set fairness threshold.", 0., 1., 0.2, 0.05, key="fairness")

    # Compute fairness measures
    privi_info = PROTECTED_FEATURES[protected_attribute]
    aif_metric = get_aif_metric(
        x_fai,
        true_class,
        pred_class,
        protected_attribute,
        privi_info["privileged_attribute_values"],
        privi_info["unprivileged_attribute_values"],
    )
    fmeasures = custom_fmeasures(aif_metric, threshold=fthresh)
    alg_fai(fmeasures, aif_metric, fthresh)

    st.subheader("Notes")
    fairness_notes()


if __name__ == "__main__":
    select_ver = st.sidebar.selectbox("Select version", ["Version 1", "Version 2"])
    if select_ver == "Version 1":
        fai(version=1)
    else:
        fai(version=2)
