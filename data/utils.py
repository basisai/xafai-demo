import pickle

import pandas as pd
import streamlit as st
from sklearn import metrics

from xai_fairness.toolkit_xai import compute_shap, get_explainer, compute_corrcoef

from data.constants import FEATURES, TARGET, PROTECTED_FEATURES
# from data.constants import FEATURES, TARGET, TARGET_CLASSES, CONFIG_FAI


@st.cache(allow_output_mutation=True)
def load_model(filename):
    return pickle.load(open(filename, "rb"))


@st.cache(allow_output_mutation=True)
def load_data(filename, sample_size=None, random_state=0):
    df = pd.read_csv(filename)
    if sample_size is None:
        return df
    return df.sample(sample_size, random_state=random_state)


@st.cache(allow_output_mutation=True)
def predict(clf, x):
    return clf.predict_proba(x)


@st.cache(allow_output_mutation=True)
def compute_shap_values(clf, x):
    explainer = get_explainer(model=clf, model_type="tree")
    return compute_shap(explainer, x)


@st.cache
def xai_data():
    clf = load_model("models/lgb_clf.pkl")
    valid = load_data("data/valid.csv", sample_size=3000)
    x_valid = valid[FEATURES]
    all_shap_values, _ = compute_shap_values(clf, x_valid)
    all_corrs = compute_corrcoef(x_valid, all_shap_values)
    return clf, x_valid, all_shap_values, all_corrs


@st.cache
def xai_indiv_data():
    clf = load_model("models/lgb_clf.pkl")
    sample = load_data("data/valid.csv")
    x_sample = sample[FEATURES]
    preds = predict(clf, x_sample)
    all_shap_values, all_base_value = compute_shap_values(clf, x_sample)
    return sample, preds, all_shap_values, all_base_value


@st.cache
def fai_data():
    clf = load_model("models/lgb_clf.pkl")
    valid = load_data("data/valid.csv")
    x_fai = valid[list(PROTECTED_FEATURES.keys())]
    y_valid = valid[TARGET].values
    y_score = predict(clf, valid[FEATURES])
    return x_fai, y_valid, y_score


def print_model_perf(y_val, y_pred):
    text = ""
    text += "Model accuracy = {:.4f}\n".format(metrics.accuracy_score(y_val, y_pred))
    text += "Weighted Average Precision = {:.4f}\n".format(
        metrics.precision_score(y_val, y_pred, average="weighted"))
    text += "Weighted Average Recall = {:.4f}\n\n".format(
        metrics.recall_score(y_val, y_pred, average="weighted"))
    text += metrics.classification_report(y_val, y_pred, digits=4)
    return text
