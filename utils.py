"""
Script containing commonly used functions.
"""
import pickle
import time

import seaborn as sns
import matplotlib.pyplot as plt
from contextlib import contextmanager
from sklearn import metrics

# sns.set()


@contextmanager
def timer(task="task"):
    t0 = time.time()
    yield
    print("Time taken for {task} = {(time.time() - t0) / 60:.2f} mins")


def save_pkl(filename, model):
    """Save pickle model."""
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    return


def load_pkl(filename):
    """Load pickle model."""
    return pickle.load(open(filename, "rb"))


def lgb_roc_auc_score(y_true, y_pred):
    return "roc_auc", metrics.roc_auc_score(y_true, y_pred), True


def print_results(actual, probs):
    preds = (probs > 0.5).astype(int)
    print("Confusion matrix:")
    print(metrics.confusion_matrix(actual, preds), "\n")
    print(metrics.classification_report(actual, preds), "\n")

    roc_auc = metrics.roc_auc_score(actual, probs)
    avg_prc = metrics.average_precision_score(actual, probs)
    print(f"  ROC AUC           = {roc_auc:.6f}")
    print(f"  Average precision = {avg_prc:.6f}")


# Keras model history
def plot_history(train_history, test_history, ax=None, title=None, ylabel=None):
    """Plot history."""
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(train_history)
    ax.plot(test_history)
    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')
    return ax


# ROC(tpr-fpr) curve
def plot_roc_curve(actual, pred, ax=None):
    """Plot ROC."""
    fpr, tpr, _ = metrics.roc_curve(actual, pred)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC AUC = {:.4f}'.format(
        metrics.roc_auc_score(actual, pred)))
    return ax


# Precision-recall curve
def plot_pr_curve(actual, pred, ax=None):
    """Plot PR curve."""
    precision, recall, _ = metrics.precision_recall_curve(actual, pred)

    if ax is None:
        fig, ax = plt.subplots()
        
    ax.step(recall, precision, color='b', alpha=0.2, where='post')
    ax.fill_between(recall, precision, alpha=0.2, color='b', step='post')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Avg precision = {:.4f}'.format(
        metrics.average_precision_score(actual, pred)))
    return ax
