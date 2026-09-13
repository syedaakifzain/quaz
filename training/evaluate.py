"""Evaluation — compute classification metrics for each LOSO fold.

Phase 13: Accuracy, Precision, Recall, F1, Sensitivity, Specificity,
Confusion Matrix, and per-subject + aggregate reporting.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from core.models import FoldMetrics

logger = logging.getLogger(__name__)


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> FoldMetrics:
    """Compute all required metrics for one fold.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        FoldMetrics with accuracy, precision, recall, f1,
        sensitivity, specificity, and confusion matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0.0))
    rec = float(recall_score(y_true, y_pred, zero_division=0.0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0.0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics = FoldMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1_score=f1,
        sensitivity=sensitivity,
        specificity=specificity,
        roc_auc=None,  # VQC may not provide probabilities
        confusion_matrix=cm.tolist(),
    )

    logger.info(
        "Fold metrics: acc=%.3f prec=%.3f rec=%.3f f1=%.3f "
        "sens=%.3f spec=%.3f  CM=%s",
        acc, prec, rec, f1, sensitivity, specificity, cm.tolist(),
    )

    return metrics


def aggregate_fold_metrics(
    fold_metrics: list[FoldMetrics],
) -> dict[str, dict[str, float | None]]:
    """Compute mean ± std across folds.

    Args:
        fold_metrics: List of per-fold metrics.

    Returns:
        Dict mapping metric name to {mean, std}.
    """
    if not fold_metrics:
        return {}

    metric_names = ["accuracy", "precision", "recall", "f1_score",
                    "sensitivity", "specificity"]

    result: dict[str, dict[str, float | None]] = {}
    for name in metric_names:
        values = [getattr(fm, name) for fm in fold_metrics]
        result[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    # ROC AUC (may be None)
    auc_values = [fm.roc_auc for fm in fold_metrics if fm.roc_auc is not None]
    if auc_values:
        result["roc_auc"] = {
            "mean": float(np.mean(auc_values)),
            "std": float(np.std(auc_values)),
        }
    else:
        result["roc_auc"] = {"mean": None, "std": None}

    logger.info("Aggregate metrics across %d folds:", len(fold_metrics))
    for name, vals in result.items():
        if vals["mean"] is not None:
            logger.info("  %s: mean=%.3f std=%.3f", name, vals["mean"], vals["std"])

    return result
