"""
Helpers for XAI
"""
import numpy as np
import pandas as pd
import shap
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
from pdpbox import pdp


@st.cache(allow_output_mutation=True)
def compute_pdp_isolate(model, dataset, model_features, feature):
    pdp_isolate_out = pdp.pdp_isolate(
        model=model,
        dataset=dataset,
        model_features=model_features,
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
def compute_pdp_interact(model, dataset, model_features, features):
    pdp_interact_out = pdp.pdp_interact(
        model=model,
        dataset=dataset,
        model_features=model_features,
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


def get_top_features(shap_values, features, max_display):
    output_df = pd.DataFrame({
        "feature": features,
        "value": np.abs(shap_values).mean(axis=0).mean(axis=0),
    })
    output_df.sort_values("value", ascending=False, inplace=True, ignore_index=True)
    return output_df.iloc[:max_display]


def compute_corrcoef(df, x_sample, pred_sample):
    output_df = df.copy()
    corrcoef = []
    for col in output_df["feature"].values:
        df_ = pd.DataFrame({"x": x_sample[col], "y": pred_sample})
        corrcoef.append(df_.corr(method='pearson').values[0, 1])
    output_df["corrcoef"] = corrcoef
    return output_df


def xai_summary(corr_df, x_sample, shap_values, feature_names, max_display):
    st.write("**SHAP Summary Plots of Top Features**")

    chart = alt.Chart(corr_df).mark_bar().encode(
        x=alt.X("value", title="mean(|SHAP value|) (average impact on model output magnitude)"),
        y=alt.Y("feature", title="", sort="-x"),
        color=alt.condition(
            "datum.corrcoef > 0",
            alt.value("#FF0D57"),
            alt.value("#1E88E5"),
        ),
        tooltip=["feature", "value"],
    )
    st.altair_chart(chart, use_container_width=True)
    st.write("Note:\n"
             "- Red: positive Pearson correlation between the feature and model output;\n"
             "- Blue: negative correlation.")

    shap.summary_plot(shap_values,
                      x_sample,
                      feature_names=feature_names,
                      max_display=max_display,
                      plot_size=[12, 6],
                      show=False)
    plt.gcf().tight_layout()
    st.pyplot()
    
    
def model_xai(model, x_sample, top_features, feature_names, category_map, target_names=None):
    st.write("**Partial Dependence Plots of Top Features**")
    # PDPbox does not allow NaNs
    _x_sample = x_sample.fillna(0)
    rev_category_map = {e: k for k, v in category_map.items() for e in v}
    _top_features = []
    for feature_name in top_features:
        if feature_name in rev_category_map.keys() and rev_category_map[feature_name] not in _top_features:
            _top_features.append(rev_category_map[feature_name])
        else:
            _top_features.append(feature_name)

    for feature_name in _top_features:
        feature = category_map.get(feature_name) or feature_name
        pdp_isolate_out = compute_pdp_isolate(model, _x_sample, feature_names, feature)
        
        if not isinstance(pdp_isolate_out, list):
            pdp_isolate_out = [pdp_isolate_out]
        
        for i, pdp_out in enumerate(pdp_isolate_out):
            chart = pdp_chart(pdp_out, feature_name)
            if target_names is not None:
                chart = chart.properties(title=f"Target Class {target_names[i]}")
            elif len(pdp_isolate_out) > 1:
                chart = chart.properties(title=f"Target Class {i}")
            st.altair_chart(chart, use_container_width=True)


def make_source_waterfall(instance, base_value, shap_values, max_display=10):
    df = pd.melt(instance)
    df.columns = ["feature", "feature_value"]
    df["shap_value"] = shap_values

    df["val_"] = df["shap_value"].abs()
    df = df.sort_values("val_", ascending=False)

    df["val_"] = df["shap_value"]
    remaining = df["shap_value"].iloc[max_display:].sum()
    output_value = df["shap_value"].sum() + base_value

    _df = df.iloc[:max_display]

    df0 = pd.DataFrame({"feature": ["Average Model Output"],
                        "shap_value": [base_value],
                        "val_": [base_value]})
    df1 = _df.query("shap_value > 0").sort_values("shap_value", ascending=False)
    df2 = pd.DataFrame({"feature": ["Others"],
                        "shap_value": [remaining],
                        "val_": [remaining]})
    df3 = _df.query("shap_value < 0").sort_values("shap_value")
    df4 = pd.DataFrame({"feature": ["Individual Observation"],
                        "shap_value": [output_value],
                        "val_": [0]})
    source = pd.concat([df0, df1, df2, df3, df4], axis=0, ignore_index=True)

    source["close"] = source["val_"].cumsum()
    source["open"] = source["close"].shift(1)
    source.loc[len(source) - 1, "open"] = 0
    source["open"].fillna(0, inplace=True)

    source["feature_value"] = source["feature_value"].round(6).astype(str)
    source["shap_value"] = source["shap_value"].round(6).astype(str)
    return source


def waterfall_chart(source):
    chart = alt.Chart(source).mark_bar().encode(
        alt.X("feature:O", sort=source["feature"].tolist()),
        alt.Y("open:Q", title="", scale=alt.Scale(zero=False)),
        alt.Y2("close:Q"),
        color=alt.condition(
            "datum.open <= datum.close",
            alt.value("#FF0D57"),
            alt.value("#1E88E5"),
        ),
        tooltip=["feature", "feature_value", "shap_value"],
    )
    chart2 = chart.encode(
        color=alt.condition(
            "datum.feature == 'Average Model Output' || datum.feature == 'Individual Observation'",
            alt.value("#F7E0B6"),
            alt.value(""),
        ),
    )
    return chart + chart2


def indiv_xai(instance, base_value, shap_values, title=None, max_display=20):
    source = make_source_waterfall(instance, base_value, shap_values, max_display=max_display)
    chart = waterfall_chart(source)
    if title is not None:
        chart = chart.properties(title=title)
    st.altair_chart(chart, use_container_width=True)
