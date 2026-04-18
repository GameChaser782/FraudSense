"""
Evaluation utilities: AUC, precision-recall, F1 at optimal threshold.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, roc_curve, precision_recall_curve, average_precision_score
)


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray,
                        metric: str = "f1") -> float:
    """Find threshold that maximises chosen metric on given set."""
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = []
    for t in thresholds:
        preds = (y_score >= t).astype(int)
        if metric == "f1":
            scores.append(f1_score(y_true, preds, zero_division=0))
        elif metric == "precision":
            scores.append(precision_score(y_true, preds, zero_division=0))
        elif metric == "recall":
            scores.append(recall_score(y_true, preds, zero_division=0))
    return float(thresholds[np.argmax(scores)])


def full_report(y_true: np.ndarray, y_score: np.ndarray,
                model_name: str = "model") -> dict:
    """Return dict of all relevant metrics."""
    auc = roc_auc_score(y_true, y_score)
    ap  = average_precision_score(y_true, y_score)
    threshold = find_best_threshold(y_true, y_score)
    preds = (y_score >= threshold).astype(int)

    report = {
        "model":     model_name,
        "auc":       round(auc, 4),
        "avg_prec":  round(ap, 4),
        "threshold": round(threshold, 3),
        "precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_true, preds, zero_division=0), 4),
        "f1":        round(f1_score(y_true, preds, zero_division=0), 4),
    }
    return report
