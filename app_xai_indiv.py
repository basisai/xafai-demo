import numpy as np
import pandas as pd
import shap
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
from pdpbox import pdp

from constants import *
from app_xai import load_model, load_data


@st.cache(allow_output_mutation=True)
def load_explainer(clf):
    # Use the relevant explainer
    return shap.TreeExplainer(clf)


def _make_source_waterfall(instance, base_value, shap_val, max_display=20):
    df = pd.melt(instance)
    df.columns = ["feature", "feature_value"]
    df["shap_value"] = shap_val

    df["val_"] = df["shap_value"].abs()
    df = df.sort_values("val_", ascending=False)
    
    df["val_"] = df["shap_value"]
    remaining = df["shap_value"].iloc[max_display:].sum()
    output_value = df["shap_value"].sum() + base_value
    
    df0 = pd.DataFrame({"feature": ["base_value"],
                        "shap_value": [base_value],
                        "val_": [base_value]})
    df1 = pd.DataFrame({"feature": ["others", "output_value"],
                        "shap_value": [remaining, output_value],
                        "val_": [remaining, 0]})
    source = pd.concat([df0, df.iloc[:max_display], df1], axis=0, ignore_index=True)

    source["close"] = source["val_"].cumsum()
    source["open"] = source["close"].shift(1)
    source["open"].iloc[-1] = 0
    source["open"].fillna(0, inplace=True)

    source["feature_value"] = source["feature_value"].round(6).astype(str)
    source["shap_value"] = source["shap_value"].round(6).astype(str)
    return source


def plot_waterfall(source):
    chart = alt.Chart(source).mark_bar().encode(
        alt.X("feature:O", sort=source["feature"].tolist()),
        alt.Y("open:Q", title="log odds", scale=alt.Scale(zero=False)),
        alt.Y2("close:Q"),
        color=alt.condition(
            "datum.open >= datum.close", alt.value("#FF0D57"), alt.value("#1E88E5")),
        tooltip=["feature", "feature_value", "shap_value"],
    )
    return chart


def xai_indiv():
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
    
    st.title("Individual Instance Explainability")
    
    clf = load_model("output/lgb.pkl")
    sample = load_data("output/train.csv")
    x_sample = sample[FEATURES]
    y_sample = sample[TARGET].values
    
    # Load explainer
    explainer = load_explainer(clf)
    
    # Select instance
    row = st.slider("Select instance", 0, sample.shape[0], 0)
    instance = x_sample.iloc[row: row + 1]
    
    st.subheader("Feature values")
    st.dataframe(instance.T)
    
    st.subheader("Actual label")
    st.write(y_sample[row])
    
    st.subheader("Prediction")
    st.text(clf.predict_proba(instance)[0])
    
    # Compute SHAP values
    st.subheader("SHAP values")
    shap_value = explainer.shap_values(instance)[1][0]
    base_value = explainer.expected_value[1]
    
    source = _make_source_waterfall(instance, base_value, shap_value)
    st.altair_chart(plot_waterfall(source), use_container_width=True)
    

if __name__ == "__main__":
    xai_indiv()
