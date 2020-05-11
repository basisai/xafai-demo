import streamlit as st

from xai_fairness.app_xai import xai
from xai_fairness.app_fai import fai
from xai_fairness.app_xai_indiv import xai_indiv


def main():
    select = st.sidebar.selectbox(
        "Select dashboard", ["Global explainability", "Fairness", "Individual explainability"])
    
    if select == "Global explainability":
        xai()
    elif select == "Fairness":
        fai()
    elif select == "Individual explainability":
        xai_indiv()
    
    
if __name__ == "__main__":
    main()
