import pandas as pd
import shap
import streamlit as st

from app_utils import load_model, load_data
from xai_utils import *
from constants import *
    

def xai_indiv():
    max_width = 1000  #st.sidebar.slider("Set page width", min_value=700, max_value=1500, value=1000, step=20)
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
    explainer = shap.TreeExplainer(clf)
    
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
    plot_shap_waterfall(explainer, instance, max_display=20)
    

if __name__ == "__main__":
    xai_indiv()
