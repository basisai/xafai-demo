"""
Report template
"""
import numpy as np
import pandas as pd
import streamlit as st

from .static_xai import compute_corrcoef, xai_summary, model_xai, indiv_xai
from .static_fai import get_fmeasures, plot_fmeasures_bar, color_red, alg_fai
    
    
def report(config, feature_names, category_map, config_fairness, output):
    model = output["model"]

    x_sample = output["x_sample"]
    shap_values = output["shap_values"]
    base_value = output["base_value"]
    sample_idx = output["sample_idx"]

    x_valid = output["x_valid"]
    y_pred = output["y_pred"]
    val_class = output["val_class"]
    pred_class = output["pred_class"]

    shap_df = compute_corrcoef(x_sample, y_pred, shap_values, config["num_top_features"])
    top_features = shap_df["feature"].tolist()
    
    # Set fairness threshold
    fthresh = config["fairness_threshold"]
    
    st.title("Model Explainability and Fairness Report")
    
    st.header("I. Model Description")
    st.markdown(
        "<span style='background-color: yellow'>*[For user to complete. Below is a sample.]*</span>",
        unsafe_allow_html=True)
    st.markdown(
        "<span style='color: blue'>{}</span>".format(config["model_description"]),
        unsafe_allow_html=True)
    
    st.header("II. List of Prohibited Features")
    st.text("religion, nationality, birth place, gender, race")
    
    st.header("III. Algorithmic Fairness")
    st.write("Algorithmic fairness assesses the models based on two technical definitions of fairness. "
             "If all are met, the model is deemed to be fair.")
    st.write("The cutoff is set at **{}**.".format(config["cutoff"]))
    st.write(f"Fairness deviation threshold is set at **{fthresh}**. "
             "Absolute fairness is 1, so a model is considered fair for the metric when "
             f"**fairness metric is between {1-fthresh:.2f} and {1+fthresh:.2f}**.")
    
    final_fairness = []
    for col, dct in config_fairness.items():
        st.subheader(f"Prohibited Feature: `{col}`")
        
        # Compute fairness measures
        fmeasures, _ = get_fmeasures(x_valid, val_class, pred_class,
                                     dct["bias_info"], dct["privileged_info"],
                                     fthresh, config["fairness_metrics"])
        st.dataframe(
            fmeasures[["Metric", "Ratio", "Fair?"]]
            .style.applymap(color_red, subset=["Fair?"])
        )
        st.altair_chart(plot_fmeasures_bar(fmeasures, fthresh), use_container_width=True)
        if np.mean(fmeasures["Fair?"] == "Yes") > 0.6:
            st.write("Overall: **Fair**")
            final_fairness.append([col, "Yes"])
        else:
            st.write("Overall: **Not fair**")
            final_fairness.append([col, "No"])
    final_fairness = pd.DataFrame(final_fairness, columns=["Prohibited Variable", "Fair?"])
    
    st.header("IV. Model Explainability")
    xai_summary(shap_df, x_sample, shap_values, feature_names, config["num_top_features"])
    
    st.markdown(
        "<span style='background-color: yellow'>*[For user to complete. Below is a sample.]*</span>",
        unsafe_allow_html=True)
    st.markdown(
        "<span style='color: blue'>{}</span>".format(config["explainability"]),
        unsafe_allow_html=True)
    
    st.header("V. Model Performance")
    st.text(output["text_model_perf"])
    
    st.header("VI. Conclusion")
    st.markdown(
        "<span style='background-color: yellow'>*[For user to complete. Below is a sample.]*</span>",
        unsafe_allow_html=True)
    
    st.markdown(
        "<span style='color: blue'>**Model performance**: {}</span>".format(
            config["conclusion"]["model_performance"]),
        unsafe_allow_html=True)
    
    st.markdown(
        "<span style='color: blue'>**Explainability**: {}</span>".format(
            config["conclusion"]["explainability"]),
        unsafe_allow_html=True)
    feats_ = shap_df.query("corrcoef > 0")["feature"].values
    st.write("The top features that have positive correlation with their model output are `"
             + "`, `".join(feats_) + "`.")
    feats_ = shap_df.query("corrcoef < 0")["feature"].values
    st.write("The top features that have negative correlation with their model output are `"
             + "`, `".join(feats_) + "`.")
    
    fair = "fair" if np.mean(final_fairness["Fair?"] == "Yes") == 1 else "not fair"
    st.write("**Fairness**: From the table below, since the model is fair for "
             f"all prohibited variables, overall the model is considered **{fair}**.")
    st.dataframe(final_fairness)

    st.text("=" * 90 + "\n" + "=" * 90)
    
    st.title("Appendix")
    
    st.header("Model Explainability")
    model_xai(model, x_sample, top_features, feature_names, category_map)

    st.header("Sample Individual Explainability")
    for k, row in sample_idx.items():
        st.write(f"Class = {k}")
        indiv_xai(x_sample.iloc[row: row + 1], base_value, shap_values[row], config["num_top_features"])

    st.header("Algorithmic Fairness")
    for col, dct in config_fairness.items():

        st.subheader(f"Prohibited Feature = `{col}`")

        # Compute fairness measures
        fmeasures, model_metric = get_fmeasures(x_valid, val_class, pred_class,
                                                dct["bias_info"], dct["privileged_info"],
                                                fthresh, config["fairness_metrics"])
        alg_fai(fmeasures, model_metric, fthresh)
        st.text("=" * 90)

    st.subheader("Notes")
    st.write("**Statistical parity**:")
    st.latex(r"\frac{\text{Selection Rate}(D=\text{unprivileged})}{\text{Selection Rate}(D=\text{privileged})}")
    
    st.write("**Equal opportunity**:")
    st.latex(r"\frac{\text{FNR}(D=\text{unprivileged})}{\text{FNR}(D=\text{privileged})}")
    
    st.write("**Equal opportunity**:")
    st.latex(r"\frac{\text{PPV}(D=\text{unprivileged})}{\text{PPV}(D=\text{privileged})}")
