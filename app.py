import streamlit as st

from app_xai import xai
from app_fai import fai


def main():
    select = st.sidebar.selectbox("Select dashboard", ["Explainability", "Fairness"])
    if select == "Explainability":
        xai()
    else:
        fai()
    
    
if __name__ == "__main__":
    main()
    