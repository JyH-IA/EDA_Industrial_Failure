"""Generación de figuras del análisis exploratorio y del modelado.

Todas las figuras se guardan en `reports/figures/` y se calculan a partir
de los datos reales en cada ejecución del pipeline.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

matplotlib.use("Agg")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110


def _save(fig: plt.Figure, name: str) -> Path:
    path = config.FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_target_balance(df: pd.DataFrame) -> Path:
    counts = df[config.TARGET].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Sin falla", "Falla"], [counts.get(0, 0), counts.get(1, 0)], color=["#4c72b0", "#c44e52"])
    for bar in bars:
        ax.annotate(f"{int(bar.get_height()):,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=10)
    ax.set_title("Balance de la variable objetivo")
    ax.set_ylabel("Registros")
    return _save(fig, "target_balance")


def plot_failure_by_type(df: pd.DataFrame) -> Path:
    grouped = df.groupby("Type")[config.TARGET].agg(["sum", "count"])
    grouped["rate"] = 100.0 * grouped["sum"] / grouped["count"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(grouped.index, grouped["rate"], color="#c44e52", alpha=0.85)
    for bar, (typ, row) in zip(bars, grouped.iterrows()):
        ax.annotate(f"{row['rate']:.1f}%\n({int(row['sum'])}/{int(row['count'])})",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_title("Tasa de falla por tipo de máquina (L, M, H)")
    ax.set_xlabel("Tipo de máquina")
    ax.set_ylabel("Tasa de falla (%)")
    ax.set_ylim(0, max(grouped["rate"]) * 1.35)
    return _save(fig, "failure_by_type")


def plot_numeric_distributions(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), config.NUMERIC_COLS):
        for value, label, color in [(0, "Sin falla", "#4c72b0"), (1, "Falla", "#c44e52")]:
            subset = df.loc[df[config.TARGET] == value, col]
            ax.hist(subset, bins=30, alpha=0.55, density=True, label=label, color=color)
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=8)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Distribuciones de variables operativas por condición de falla", y=1.02)
    return _save(fig, "numeric_distributions")


def plot_boxplots_by_failure(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, col in zip(axes, config.NUMERIC_COLS):
        sns.boxplot(data=df, x=config.TARGET, y=col, hue=config.TARGET, ax=ax,
                    legend=False, palette=["#4c72b0", "#c44e52"])
        ax.set_xlabel("Falla")
        ax.set_title(col, fontsize=10)
    fig.suptitle("Comparación de variables operativas: sin falla vs. falla", y=1.03)
    return _save(fig, "boxplots_by_failure")


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    cols = config.NUMERIC_COLS + [config.TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlaciones (lineales) entre variables operativas y falla")
    return _save(fig, "correlation_heatmap")


def plot_scatter_torque_rpm(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df.sample(2500, random_state=config.RANDOM_SEED),
                   x="Rotational speed [rpm]", y="Torque [Nm]",
                   hue=config.TARGET, palette=["#4c72b0", "#c44e52"],
                   alpha=0.35, s=18, ax=ax)
    ax.set_title("Torque vs. velocidad de rotación (muestra de 2,500 registros)")
    return _save(fig, "scatter_torque_rpm")


def plot_pr_curves(pr_data: dict, selected_model: str, threshold: float) -> Path:
    """Curvas precision-recall por modelo sobre la partición de prueba."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, curve in pr_data.items():
        label = f"{name} (AP = {curve['average_precision']:.3f})"
        lw = 2.6 if name == selected_model else 1.6
        ls = "-" if name == selected_model else "--"
        ax.plot(curve["recall"], curve["precision"], label=label, lw=lw, ls=ls)
    ax.set_xlabel("Recall (fallas detectadas)")
    ax.set_ylabel("Precision")
    ax.set_title(f"Curvas Precision-Recall (prueba). Umbral congelado: {threshold:.2f}")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    return _save(fig, "pr_curves")


def plot_confusion_matrix(cm: dict, model_name: str) -> Path:
    """Matriz de confusión del modelo seleccionado en prueba."""
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt=",", cmap="Blues",
                xticklabels=["Pred: sin falla", "Pred: falla"],
                yticklabels=["Real: sin falla", "Real: falla"], ax=ax)
    ax.set_title(f"Matriz de confusión — {model_name} (prueba)")
    return _save(fig, "confusion_matrix")


def plot_permutation_importance(names: list[str], means: list[float], stds: list[float]) -> Path:
    order = np.argsort(means)
    names = [names[i] for i in order]
    means_arr = np.array(means)[order]
    stds_arr = np.array(stds)[order]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(names, means_arr, xerr=stds_arr, color="#55a868", alpha=0.85)
    ax.set_xlabel("Disminución de average precision al permutar la variable")
    ax.set_title("Permutation importance (partición de prueba, análisis descriptivo)")
    return _save(fig, "permutation_importance")
