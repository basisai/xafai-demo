import streamlit as st

from app_xai import xai
from app_fai import fai
from app_xai_indiv import xai_indiv


def main():
    select = st.sidebar.selectbox("Select dashboard", ["Global explainability", "Fairness", "Individual explainability"])
    if select == "Global explainability":
        xai()
    elif select == "Fairness":
        fai()
    else:
        xai_indiv()
    
    
if __name__ == "__main__":
    main()
