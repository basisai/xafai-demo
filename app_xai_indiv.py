import shap
import streamlit as st

from app_utils import load_model, load_data
from constants import FEATURES, TARGET
from xai_fairness.static_xai import make_source_waterfall, waterfall_chart

    
def xai_indiv():
    st.title("Individual Instance Explainability")
    
    clf = load_model("models/lgb_clf.pkl")
    sample = load_data("data/valid.csv")
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
    shap_values = explainer.shap_values(instance)[1][0]
    base_value = explainer.expected_value[1]
    source = make_source_waterfall(instance, base_value, shap_values, max_display=20)
    st.altair_chart(waterfall_chart(source), use_container_width=True)
    

if __name__ == "__main__":
    xai_indiv()
