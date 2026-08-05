from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    loss: float
    threshold: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    mcc: float
    roc_auc: float
    pr_auc: float
    brier: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    best_threshold = 0.5
    best_key = (-np.inf, -np.inf, -np.inf)
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        mcc = matthews_corrcoef(labels, predictions)
        f1 = f1_score(labels, predictions, zero_division=0)
        balanced = balanced_accuracy_score(labels, predictions)
        key = (mcc, f1, balanced)
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    loss: float,
    threshold: float,
) -> EvaluationResult:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / max(1, tn + fp)
    return EvaluationResult(
        loss=float(loss),
        threshold=float(threshold),
        accuracy=float(accuracy_score(labels, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        specificity=float(specificity),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        mcc=float(matthews_corrcoef(labels, predictions)),
        roc_auc=float(roc_auc_score(labels, probabilities)),
        pr_auc=float(average_precision_score(labels, probabilities)),
        brier=float(brier_score_loss(labels, probabilities)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )
