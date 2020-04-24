import numpy as np
import pandas as pd
import shap
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
from pdpbox import pdp

from constants import *


@st.cache(allow_output_mutation=True)
def load_model(filename):
    import pickle
    return pickle.load(open(filename, "rb"))


@st.cache
def load_data(filename, sample_size=None, random_state=0):
    df = pd.read_csv(filename)
    if sample_size is None:
        return df
    return df.sample(sample_size, random_state=random_state)


@st.cache
def compute_shap_values(clf, x_sample):
    # Use the relevant explainer
    explainer = shap.TreeExplainer(clf)
    return explainer.shap_values(x_sample)[1]


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


def xai():
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
    
    st.title("Explainability AI Dashboard")

    st.sidebar.title("Model and Data Instructions")
    st.sidebar.info(
        "Write your own `load_model`, `load_data` functions.\n"
        "Model must be a fitted `sklearn` model.\n"
        "Sample data must be a pandas.DataFrame"
        "Feature names and a category map for one-hot encoded features must be "
        "furnished in `constants.py`."
    )
    
    # Load model, sample data
    clf = load_model("output/lgb.pkl")
    sample = load_data("output/train.csv", sample_size=3000)
    x_sample = sample[FEATURES]
    
    st.header("SHAP")
    st.sidebar.title("SHAP Instructions")
    st.sidebar.info(
        "Set the relevant explainer in `compute_shap_values` for your model.\n"
        "shap.TreeExplainer works with tree models.\n"
        "shap.DeepExplainer works with Deep Learning models.\n"
        "shap.KernelExplainer works with all models, though it is slower than "
        "other Explainers and it offers an approximation rather than exact "
        "Shap values."
        "See Explainers[https://shap.readthedocs.io/en/latest/#explainers] for more details"
    )
    
    # Compute SHAP values
    shap_values = compute_shap_values(clf, x_sample)
    
    # summarize the effects of all features
    st.subheader("SHAP summary plot")
    max_display = st.slider("Select number of top features to show", 10, 30, 10)
    
    shap.summary_plot(shap_values, plot_type="bar", feature_names=FEATURES,
                      max_display=max_display, plot_size=0.2, show=False)
    plt.gcf().tight_layout()
    st.pyplot()
    
    shap.summary_plot(shap_values, x_sample, feature_names=FEATURES,
                      max_display=max_display, plot_size=0.2, show=False)
    plt.gcf().tight_layout()
    st.pyplot()
    
    st.subheader("SHAP dependence contribution plots")
    features = st.multiselect("Select two features", FEATURES, key="shap")
    if len(features) > 1:
        feat1, feat2 = features[:2]
        shap.dependence_plot(feat1, shap_values, x_sample, interaction_index=feat2, show=False)
        plt.gcf()#.tight_layout()
        st.pyplot()
    
    st.header("Partial dependence plots")
    st.sidebar.title("PDPbox Instructions")
    st.sidebar.info("[placeholder]")
    
    st.subheader("Partial dependence plots")
    feature_name = st.selectbox("Select feature", NUMERIC_FEATS + CATEGORICAL_FEATS)
#     if feature_name in CATEGORICAL_FEATS:
#         feature = CATEGORY_MAP[feature_name]
#         st.pyplot(pdp_plot(clf, x_sample, FEATURES, feature, feature_name).tight_layout())
#     else:
#         feature = feature_name
#         st.pyplot(pdp_plot(clf, x_sample, FEATURES, feature, feature_name,
#                            num_grid_points=12, show_percentile=True))
    
    feature = CATEGORY_MAP.get(feature_name) or feature_name
    pdp_isolate_out = compute_pdp_isolate(clf, x_sample, feature)
    st.altair_chart(pdp_chart(pdp_isolate_out, feature_name), use_container_width=True)
    
    st.subheader("Partial dependence interaction plots")
    feature_names = st.multiselect("Select two features", CATEGORICAL_FEATS + NUMERIC_FEATS, key="pdp")
    if len(feature_names) > 1:
        feature_name1, feature_name2 = feature_names[:2]
        feature1 = CATEGORY_MAP.get(feature_name1) or feature_name1
        feature2 = CATEGORY_MAP.get(feature_name2) or feature_name2
#         st.pyplot(pdp_interact_plot(clf, x_sample, FEATURES, feature1, feature2))
        pdp_interact_out = compute_pdp_interact(clf, x_sample, [feature1, feature2])
        st.altair_chart(pdp_heatmap(pdp_interact_out, feature_names[:2]), use_container_width=True)


if __name__ == "__main__":
    xai()
