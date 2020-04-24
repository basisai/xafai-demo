"""
Script containing commonly used functions.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def save_pkl(filename, model):
    """Save pickle model."""
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    return


def load_pkl(filename):
    """Load pickle model."""
    return pickle.load(open(filename, "rb"))


def compute_log_metrics(y_val, y_prob):
    """Compute and log metrics."""
    y_pred = (y_prob > 0.5).astype(int)
    print(f"Accuracy = {metrics.accuracy_score(y_val, y_pred):.6f}")
    print(f"ROC AUC = {metrics.roc_auc_score(y_val, y_prob):.6f}")
    print(f"Average precision = {metrics.average_precision_score(y_val, y_prob):.6f}")
    return


# Keras model history
def plot_history(history_arr1, history_arr2, title=None, ylabel=None):
    """Plot history."""
    fig, ax = plt.subplots()
    ax.plot(history_arr1)
    ax.plot(history_arr2)
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')
    return fig


# ROC(tpr-fpr) curve
def plot_roc_curve(actual, pred):
    """Plot ROC."""
    fpr, tpr, _ = metrics.roc_curve(actual, pred)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC AUC = {:.4f}'.format(
        metrics.roc_auc_score(actual, pred)))
    return fig


# Precision-recall curve
def plot_pr_curve(actual, pred):
    """Plot PR curve."""
    precision, recall, _ = metrics.precision_recall_curve(actual, pred)

    fig, ax = plt.subplots()
    ax.step(recall, precision, color='b', alpha=0.2, where='post')
    ax.fill_between(recall, precision, alpha=0.2, color='b', step='post')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Avg precision = {:.4f}'.format(
        metrics.average_precision_score(actual, pred)))
    return fig
