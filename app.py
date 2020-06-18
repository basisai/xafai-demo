import streamlit as st

from app_xai import xai
from app_fai import fai
from app_xai_indiv import xai_indiv


def main():
    max_width = 1000  # st.sidebar.slider("Set page width", min_value=700, max_value=1500, value=1000, step=20)
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

    select = st.sidebar.selectbox(
        "Select dashboard", ["Global explainability", "Individual explainability",
                             "Fairness v1", "Fairness v2"])
    
    if select == "Global explainability":
        xai()
    elif select == "Individual explainability":
        xai_indiv()
    elif select == "Fairness v1":
        fai(version=1)
    elif select == "Fairness v2":
        fai(version=2)
    
    
if __name__ == "__main__":
    main()
