"""
Streamlit app.
"""
import streamlit as st

from dashboard.app_xai import xai
from dashboard.app_fai import fai
from dashboard.app_xai_indiv import xai_indiv
from dashboard import app_fai_compare


def main():
    select_ex = st.sidebar.selectbox("Select example", [
        "Binary classification",
    ])

    select_db = st.sidebar.selectbox("Select dashboard", [
        "Global explainability",
        "Individual explainability",
        "Fairness",
        "Fairness before and after mitigation",
    ])
    
    if select_db == "Global explainability":
        xai()
    elif select_db == "Individual explainability":
        xai_indiv()
    elif select_db == "Fairness":
        select_ver = st.sidebar.selectbox("Select version", ["Version 1", "Version 2"])
        if select_ver == "Version 1":
            st.sidebar.info(
                "- Applicable for both binary and multiclass.\n"
                "- Prediction threshold is fixed a priori.\n"
                "- Allows user to toggle classes.")
            fai(version=1)
        else:
            st.sidebar.info(
                "- Applicable for binary only.\n"
                "- No selection of classes.\n"
                "-  Allows user to toggle prediction threshold.")
            fai(version=2)
    elif select_db == "Fairness before and after mitigation":
        select = st.selectbox(
            "", ["Fairness Before Mitigation", "Fairness After Mitigation", "Comparison"])
        st.title(select)
        if select == "Fairness Before Mitigation":
            app_fai_compare.fai(debias=False)
        elif select == "Fairness After Mitigation":
            app_fai_compare.fai(debias=True)
        elif select == "Comparison":
            app_fai_compare.compare()
    
    
if __name__ == "__main__":
    main()
