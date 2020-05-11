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
