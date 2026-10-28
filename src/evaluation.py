"""Métricas, matrices de confusión, curvas PR e importancia de variables.

Todas las métricas se calculan durante la ejecución; ninguna cifra de este
proyecto está precargada o inventada.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """Métricas para la clase de falla (positiva) bajo un umbral fijo.

    `y_score` es la probabilidad/score de la clase positiva. El accuracy se
    reporta como métrica complementaria, nunca como criterio de selección.
    """
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "average_precision": round(float(average_precision_score(y_true, y_score)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "n_positive": int(np.sum(y_true == 1)),
        "n_total": int(len(y_true)),
    }


def pr_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Puntos de la curva precision-recall para graficar."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return {
        "recall": recall.tolist(),
        "precision": precision.tolist(),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }


def permutation_importance_scores(model, X, y, n_repeats: int = 10, seed: int = 42) -> dict:
    """Permutation importance calculada sobre una partición dada.

    En este proyecto se aplica sobre la partición de PRUEBA del modelo ya
    congelado, únicamente como análisis descriptivo complementario: la
    importancia no se usa para seleccionar el modelo ni para reentrenar.
    """
    from sklearn.inspection import permutation_importance as sk_pi

    result = sk_pi(model, X, y, n_repeats=n_repeats, random_state=seed, n_jobs=-1, scoring="average_precision")
    return {
        "means": result.importances_mean.tolist(),
        "stds": result.importances_std.tolist(),
    }
