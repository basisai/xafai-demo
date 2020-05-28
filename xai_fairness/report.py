"""
Report template
"""
import numpy as np
import pandas as pd
import streamlit as st

from .static_xai import (
    get_top_features,
    compute_corrcoef,
    xai_summary,
    model_xai,
    indiv_xai,
)
from .static_fai import (
    get_fmeasures,
    plot_fmeasures_bar,
    color_red,
    alg_fai,
)
from util import page_break, add_header


def get_sample_idx(y_class):
    """Function to select one sample from your class for individual XAI."""
    sample_idx = {}
    for c in set(y_class):
        row = next(i for i in range(len(y_class)) if y_class[i] == c)
        sample_idx[str(c)] = row
    return sample_idx


def check_values(shap_values, base_value):
    """
    Check shape of shap_values and base_value.
    len(base_value) == len(shap_values) and type(shap_values) must be a list
    """
    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 2:
        shap_values = [shap_values]
    return shap_values, np.array(base_value).reshape(-1)


def generate_report(config,
                    feature_names,
                    category_map,
                    config_fai,
                    output):

    model = output["model"]
    valid = output["valid"]
    x_valid = valid[feature_names]
    y_score = output["y_score"]
    shap_values = output["shap_values"]
    base_value = output["base_value"]
    sample_idx = output["sample_idx"]
    true_class = output["true_class"]
    pred_class = output["pred_class"]

    # Flag if model is multiclass
    is_multiclass = (len(base_value) > 1)

    # Get top features by shap_values
    output_df = get_top_features(shap_values, feature_names, config["num_top_features"])
    top_features = output_df["feature"].tolist()

    # Get unique fairness classes
    unique_classes = np.unique(true_class)
    # If there are 2 classes, select the latter
    if len(unique_classes) == 2:
        unique_classes = unique_classes[1:]

    # Set fairness threshold
    fthresh = config["fairness_threshold"]

    cover_page_path = config["cover_page_path"] or "../report_style/dbs/assets/cover_full.png"
    add_header(cover_page_path)
    page_break()
    add_header("../report_style/dbs/assets/header.png")

    st.header("I. Model Description")
    st.markdown(
        "<span style='background-color: yellow'>*[For user to complete. Below is a sample.]*</span>",
        unsafe_allow_html=True)
    st.markdown(
        "<span style='color: blue'>{}</span>".format(config["model_description"]),
        unsafe_allow_html=True)
    
    st.header("II. List of Prohibited Features")
    st.write("religion, nationality, birth place, gender, race")
    
    st.header("III. Algorithmic Fairness")
    st.write("Algorithmic fairness assesses the models based on two technical definitions of fairness. "
             "If all are met, the model is deemed to be fair.")
    st.write(f"Fairness deviation threshold is set at **{fthresh}**. "
             "Absolute fairness is 1, so a model is considered fair for the metric when "
             f"**fairness metric is between {1-fthresh:.2f} and {1+fthresh:.2f}**.")
    
    final_fairness = []
    for attr, attr_values in config_fai.items():
        st.subheader(f"Prohibited Feature: `{attr}`")
        
        for fcl in unique_classes:
            _true_class = (true_class == fcl).astype(int)
            _pred_class = (pred_class == fcl).astype(int)
            
            # Compute fairness measures
            fmeasures, _ = get_fmeasures(x_valid,
                                         _true_class,
                                         _pred_class,
                                         attr,
                                         attr_values["privileged_attribute_values"],
                                         attr_values["unprivileged_attribute_values"],
                                         fthresh=fthresh,
                                         fairness_metrics=config["fairness_metrics"])

            if len(unique_classes) > 2:
                st.subheader(f"Fairness Class `{fcl}` vs rest")
            st.dataframe(
                fmeasures[["Metric", "Ratio", "Fair?"]]
                .style.applymap(color_red, subset=["Fair?"])
            )
            st.altair_chart(plot_fmeasures_bar(fmeasures, fthresh), use_container_width=True)
            if np.mean(fmeasures["Fair?"] == "Yes") > 0.6:
                st.write("Overall: **Fair**")
                final_fairness.append([f"{attr}-class{fcl}", "Yes"])
            else:
                st.write("Overall: **Not fair**")
                final_fairness.append([f"{attr}-class{fcl}", "No"])
                
    final_fairness = pd.DataFrame(final_fairness, columns=["Prohibited Variable", "Fair?"])
    
    st.header("IV. Model Explainability")
    if is_multiclass:
        for lb, shap_val in enumerate(shap_values):
            st.subheader(f"Target Class `{lb}`")
            corr_df = compute_corrcoef(output_df, x_valid, y_score[:, lb])
            xai_summary(corr_df,
                        x_valid,
                        shap_val,
                        feature_names,
                        config["num_top_features"])
    else:
        corr_df = compute_corrcoef(output_df, x_valid, y_score)
        xai_summary(corr_df,
                    x_valid,
                    shap_values[0],
                    feature_names,
                    config["num_top_features"])

    feats_ = output_df["feature"].values[:5]
    st.write("The top features are `" + "`, `".join(feats_) + "`.")

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
    if not is_multiclass:
        feats_ = corr_df.query("corrcoef > 0")["feature"].values
        st.write("The top features that have positive correlation with their model output are `"
                 + "`, `".join(feats_) + "`.")
        feats_ = corr_df.query("corrcoef < 0")["feature"].values
        st.write("The top features that have negative correlation with their model output are `"
                 + "`, `".join(feats_) + "`.")
    
    fair = "fair" if np.mean(final_fairness["Fair?"] == "Yes") == 1 else "not fair"
    st.write("**Fairness**: From the table below, since the model is fair for "
             f"all prohibited variables, overall the model is considered **{fair}**.")
    st.dataframe(final_fairness.style.applymap(color_red, subset=["Fair?"]))

    page_break()
    add_header("../report_style/dbs/assets/header.png")
    st.title("Appendix")

    st.header("Model Explainability")
    model_xai(model, x_valid, top_features, feature_names, category_map)

    st.header("Sample Individual Explainability")
    for fcl, row in sample_idx.items():
        st.subheader(f"Fairness Class {fcl}")
        for lb, (base_val, shap_val) in enumerate(zip(base_value, shap_values)):
            if len(base_value) == 1:
                title = None
            else:
                title = f"Target Class {lb}"
            indiv_xai(x_valid.iloc[row: row + 1],
                      base_val,
                      shap_val[row],
                      title=title,
                      max_display=config["num_top_features"])

    st.header("Algorithmic Fairness")
    for attr, attr_values in config_fai.items():
        st.subheader(f"Prohibited Feature: `{attr}`")
        for fcl in unique_classes:
            _true_class = (true_class == fcl).astype(int)
            _pred_class = (pred_class == fcl).astype(int)

            # Compute fairness measures
            fmeasures, model_metric = get_fmeasures(x_valid,
                                                    _true_class,
                                                    _pred_class,
                                                    attr,
                                                    attr_values["privileged_attribute_values"],
                                                    attr_values["unprivileged_attribute_values"],
                                                    fthresh=fthresh,
                                                    fairness_metrics=config["fairness_metrics"])
            
            if len(unique_classes) > 2:
                st.subheader(f"Fairness Class `{fcl}` vs rest")
            alg_fai(fmeasures, model_metric, fthresh)

    st.header("Notes")
    st.write("**Equal opportunity**:")
    st.latex(r"\frac{\text{FNR}(D=\text{unprivileged})}{\text{FNR}(D=\text{privileged})}")
    st.write("**Predictive parity**:")
    st.latex(r"\frac{\text{PPV}(D=\text{unprivileged})}{\text{PPV}(D=\text{privileged})}")
    st.write("**Statistical parity**:")
    st.latex(r"\frac{\text{Selection Rate}(D=\text{unprivileged})}{\text{Selection Rate}(D=\text{privileged})}")
