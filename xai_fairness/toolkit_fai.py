"""
Toolkit for fairness.
"""
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics.classification_metric import ClassificationMetric


def prepare_dataset(
        features,
        labels,
        protected_attribute,
        privileged_attribute_values,
        unprivileged_attribute_values,
        favorable_label=1.,
        unfavorable_label=0.,
    ):
    """Prepare dataset for computing fairness metrics."""
    df = features.copy()
    df['outcome'] = labels

    return BinaryLabelDataset(
        df=df,
        label_names=['outcome'],
        scores_names=list(),
        protected_attribute_names=[protected_attribute],
        privileged_protected_attributes=[np.array(privileged_attribute_values)],
        unprivileged_protected_attributes=[np.array(unprivileged_attribute_values)],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
    )


def get_aif_metric(
        valid,
        true_class,
        pred_class,
        protected_attribute,
        privileged_attribute_values,
        unprivileged_attribute_values,
        favorable_label=1.,
        unfavorable_label=0.,
    ):
    """Get aif metric wrapper."""
    grdtruth = prepare_dataset(
        valid,
        true_class,
        protected_attribute,
        privileged_attribute_values,
        unprivileged_attribute_values,
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
    )

    predicted = prepare_dataset(
        valid,
        pred_class,
        protected_attribute,
        privileged_attribute_values,
        unprivileged_attribute_values,
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
    )

    aif_metric = ClassificationMetric(
        grdtruth,
        predicted,
        unprivileged_groups=[{protected_attribute: v} for v in unprivileged_attribute_values],
        privileged_groups=[{protected_attribute: v} for v in privileged_attribute_values],
    )
    return aif_metric


def hmean(x, y):
    """Harmonic mean of x and y."""
    return 2 / (1 / x + 1 / y)


def compute_fairness_measures(aif_metric):
    """Compute fairness measures."""
    fmeasures = list()

    # Equal opportunity: equal FNR
    fnr_ratio = aif_metric.false_negative_rate_ratio()
    fmeasures.append([
        "Equal opportunity",
        "Separation",
        aif_metric.false_negative_rate(),
        aif_metric.false_negative_rate(False),
        aif_metric.false_negative_rate(True),
        fnr_ratio,
    ])

    # Statistical parity
    disparate_impact = aif_metric.disparate_impact()
    fmeasures.append([
        "Statistical parity",
        "Independence",
        aif_metric.selection_rate(),
        aif_metric.selection_rate(False),
        aif_metric.selection_rate(True),
        disparate_impact,
    ])

    # Predictive equality: equal FPR
    fpr_ratio = aif_metric.false_positive_rate_ratio()
    fmeasures.append([
        "Predictive equality (equal FPR)",
        "Separation",
        aif_metric.false_positive_rate(),
        aif_metric.false_positive_rate(False),
        aif_metric.false_positive_rate(True),
        fpr_ratio,
    ])

    # Equal TPR
    tpr_ratio = aif_metric.true_positive_rate(False) / aif_metric.true_positive_rate(True)
    fmeasures.append([
        "Equal TPR",
        "Separation",
        aif_metric.true_positive_rate(),
        aif_metric.true_positive_rate(False),
        aif_metric.true_positive_rate(True),
        tpr_ratio,
    ])

    # # Equalized odds: equal TPR and equal FPR
    # # using harmonic mean
    # eqodds_all = hmean(
    #     aif_metric.true_positive_rate(),
    #     aif_metric.false_positive_rate(),
    # )
    # eqodds_up = hmean(
    #     aif_metric.true_positive_rate(False),
    #     aif_metric.false_positive_rate(False),
    # )
    # eqodds_p = hmean(
    #     aif_metric.true_positive_rate(True),
    #     aif_metric.false_positive_rate(True),
    # )
    # eqodds_ratio = eqodds_up / eqodds_p
    # fmeasures.append([
    #     "Equalized odds",
    #     "Separation",
    #     eqodds_all,
    #     eqodds_up,
    #     eqodds_p,
    #     eqodds_ratio,
    # ])

    # Predictive parity: equal PPV
    ppv_all = aif_metric.positive_predictive_value()
    ppv_up = aif_metric.positive_predictive_value(False)
    ppv_p = aif_metric.positive_predictive_value(True)
    ppv_ratio = ppv_up / ppv_p
    fmeasures.append([
        "Predictive parity (equal PPV)",
        "Sufficiency",
        ppv_all,
        ppv_up,
        ppv_p,
        ppv_ratio,
    ])

    # Equal NPV
    tpr_ratio = aif_metric.negative_predictive_value(False) / aif_metric.negative_predictive_value(True)
    fmeasures.append([
        "Equal NPV",
        "Sufficiency",
        aif_metric.negative_predictive_value(),
        aif_metric.negative_predictive_value(False),
        aif_metric.negative_predictive_value(True),
        tpr_ratio,
    ])

    # # Conditional use accuracy equality: equal PPV and equal NPV
    # # using harmonic mean
    # acceq_all = hmean(
    #     aif_metric.positive_predictive_value(False),
    #     aif_metric.negative_predictive_value(False),
    # )
    # acceq_up = hmean(
    #     aif_metric.positive_predictive_value(False),
    #     aif_metric.negative_predictive_value(False),
    # )
    # acceq_p = hmean(
    #     aif_metric.positive_predictive_value(True),
    #     aif_metric.negative_predictive_value(True),
    # )
    # acceq_ratio = acceq_up / acceq_p
    # fmeasures.append([
    #     "Conditional use accuracy equality",
    #     "Sufficiency",
    #     acceq_all,
    #     acceq_up,
    #     acceq_p,
    #     acceq_ratio,
    # ])

    df = pd.DataFrame(fmeasures, columns=[
        "Metric", "Criterion", "All", "Unprivileged", "Privileged", "Ratio"])
    df.index.name = "order"
    df.reset_index(inplace=True)
    return df


def get_perf_measure_by_group(aif_metric, metric_name):
    """Get performance measures by group."""
    perf_measures = ['TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV', 'FDR', 'FOR', 'ACC']

    func_dict = {
        'selection_rate': lambda x: aif_metric.selection_rate(privileged=x),
        'precision': lambda x: aif_metric.precision(privileged=x),
        'recall': lambda x: aif_metric.recall(privileged=x),
        'sensitivity': lambda x: aif_metric.sensitivity(privileged=x),
        'specificity': lambda x: aif_metric.specificity(privileged=x),
        'power': lambda x: aif_metric.power(privileged=x),
        'error_rate': lambda x: aif_metric.error_rate(privileged=x),
    }

    if metric_name in perf_measures:
        metric_func = lambda x: aif_metric.performance_measures(privileged=x)[metric_name]
    elif metric_name in func_dict.keys():
        metric_func = func_dict[metric_name]
    else:
        raise NotImplementedError

    df = pd.DataFrame({
        'Group': ['all', 'privileged', 'unprivileged'],
        metric_name: [metric_func(group) for group in [None, True, False]],
    })
    return df
