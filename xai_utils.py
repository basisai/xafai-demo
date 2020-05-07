import pandas as pd
import altair as alt
import streamlit as st


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
    _df1 = pd.DataFrame({"feature": ["Others"],
                         "shap_value": [remaining],
                         "val_": [remaining]})
    _df = pd.concat([_df, _df1], axis=0, ignore_index=True)
    
    df0 = pd.DataFrame({"feature": ["Average Model Output"],
                        "shap_value": [base_value],
                        "val_": [base_value]})
    df1 = _df.query("shap_value > 0").sort_values("shap_value", ascending=False)
    df2 = _df.query("shap_value < 0").sort_values("shap_value")
    df3 = pd.DataFrame({"feature": ["Individual Observation"],
                        "shap_value": [output_value],
                        "val_": [0]})
    source = pd.concat([df0, df1, df2, df3], axis=0, ignore_index=True)
    
    source["close"] = source["val_"].cumsum()
    source["open"] = source["close"].shift(1)
    source.loc[len(source)-1, "open"] = 0
    source["open"].fillna(0, inplace=True)

    source["feature_value"] = source["feature_value"].round(6).astype(str)
    source["shap_value"] = source["shap_value"].round(6).astype(str)
    return source


def waterfall_chart(source):
    chart = alt.Chart(source).mark_bar().encode(
        alt.X("feature:O", sort=source["feature"].tolist()),
        alt.Y("open:Q", title="log odds", scale=alt.Scale(zero=False)),
        alt.Y2("close:Q"),
        color=alt.condition(
            "datum.open <= datum.close", alt.value("#FF0D57"), alt.value("#1E88E5")),
        tooltip=["feature", "feature_value", "shap_value"],
    )
    return chart


def plot_shap_waterfall(explainer, _instance, max_display=10):
    _shap_values = explainer.shap_values(_instance)[1][0]
    _base_value = explainer.expected_value[1]
    source = make_source_waterfall(_instance, _base_value, _shap_values, max_display=max_display)
    st.altair_chart(waterfall_chart(source), use_container_width=True)
