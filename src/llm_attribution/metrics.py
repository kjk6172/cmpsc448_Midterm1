"""Metrics and prompt-cluster bootstrap intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def classification_metrics(y_true, y_pred, labels: list[str]) -> dict:
    label_ids = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=label_ids, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)),
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                                "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_ids).tolist(),
        "confusion_matrix_order": labels,
        "n_test_rows": len(y_true),
    }


def cluster_bootstrap_accuracy(y_true, y_pred, prompt_ids: list[str], seed: int = 2026,
                               repetitions: int = 1000) -> list[float]:
    """Resample entire prompts, retaining their correlated model answers."""
    correctness = np.asarray(y_true) == np.asarray(y_pred)
    groups = sorted(set(prompt_ids))
    group_correct = np.asarray([correctness[np.asarray(prompt_ids) == pid].sum() for pid in groups])
    group_sizes = np.asarray([(np.asarray(prompt_ids) == pid).sum() for pid in groups])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(repetitions, len(groups)))
    sample_means = group_correct[indices].sum(axis=1) / group_sizes[indices].sum(axis=1)
    return [float(value) for value in np.quantile(sample_means, [0.025, 0.975])]
