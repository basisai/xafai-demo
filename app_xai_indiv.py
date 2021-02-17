"""
App for individual XAI.
"""
import altair as alt
import streamlit as st

from data.constants import FEATURES, TARGET, TARGET_CLASSES
from data.utils import load_model, load_data, predict, compute_shap_values
from xai_fairness.static_xai import make_source_waterfall, waterfall_chart


def plot_hist(source):
    """Plot custom histogram."""
    base = alt.Chart(source)
    chart = base.mark_area(
        opacity=0.5, interpolate="step",
    ).encode(
        alt.X("Prediction:Q", bin=alt.Bin(maxbins=10), title="Prediction"),
        alt.Y("count()", stack=None),
    ).properties(
        width=280,
        height=200,
    )
    return chart


def xai_indiv():
    st.title("Individual Instance Explainability")
    
    # Load model, valid data. Compute SHAP values
    clf = load_model("models/lgb_clf.pkl")
    sample = load_data("data/valid.csv")
    x_sample = sample[FEATURES]
    y_sample = sample[TARGET].values
    preds = predict(clf, x_sample)
    all_shap_values, all_base_value = compute_shap_values(clf, x_sample)

    if TARGET_CLASSES is not None and len(TARGET_CLASSES) > 2:
        idx = st.selectbox(
            "Select class", list(range(len(TARGET_CLASSES))), 1,
            format_func=lambda i: TARGET_CLASSES[i])
        scores = preds[:, idx]
    else:
        idx = 0
        scores = preds[:, 1]

    # TODO
    score_df = sample[[TARGET]].copy()
    score_df["Prediction"] = scores
    charts = [plot_hist(score_df[score_df[TARGET] == lb]).properties(title=f"Class = {lb}")
              for lb in TARGET_CLASSES]
    st.altair_chart(alt.concat(*charts, columns=2), use_container_width=True)

    # customized
    bin_options = [f"{i/10:.1f} - {(i+1)/10:.1f}" for i in range(10)]
    scores_bin = (scores * 10).astype(int)

    c0, c1 = st.beta_columns(2)
    select_class = c0.selectbox("Select class", TARGET_CLASSES, 1)
    class_idx = TARGET_CLASSES.index(select_class)
    select_bin = c1.selectbox("Select prediction bin", bin_options)
    bin_idx = bin_options.index(select_bin)
    select_samples = sample.index[(y_sample == class_idx) & (scores_bin == bin_idx)]

    if len(select_samples) == 0:
        st.write("**No instances found.**")
        return

    # Select instance
    _row_idx = st.slider("Select instance", 0, len(select_samples) - 1, 0)
    row_idx = select_samples[_row_idx]
    instance = x_sample.iloc[row_idx: row_idx + 1]

    st.write(f"**Actual label: `{y_sample[row_idx]}`**")
    st.write(f"**Prediction: `{scores[row_idx]:.4f}`**")

    # Compute SHAP values
    st.subheader("Feature SHAP contribution to prediction")
    shap_values = all_shap_values[idx][row_idx]
    base_value = all_base_value[idx]
    source = make_source_waterfall(instance, base_value, shap_values, max_display=20)
    st.altair_chart(waterfall_chart(source), use_container_width=True)

    df = instance.copy().T
    df.columns = ["feature_value"]
    df["shap_value"] = shap_values
    st.write(df)    


if __name__ == "__main__":
    xai_indiv()
