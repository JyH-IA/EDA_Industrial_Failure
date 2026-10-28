"""Definición y entrenamiento de los modelos de clasificación.

Protocolo (fijado antes de ver resultados):
- Los tres modelos comparten exactamente las mismas particiones.
- El preprocesamiento (one-hot, escalado) vive dentro de un `Pipeline` de
  scikit-learn y se ajusta solo con entrenamiento.
- La selección del modelo se hace por average precision (PR-AUC) en la
  partición de VALIDACIÓN.
- El umbral de decisión se ajusta únicamente con la partición de validación
  (criterio: maximizar el F1 de la clase de falla) y luego se congela.
- La partición de PRUEBA solo se usa para el reporte final.
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def build_model_pipelines() -> dict:
    """Devuelve los tres pipelines comparables del proyecto."""
    dummy = Pipeline([
        ("clf", DummyClassifier(strategy="prior", random_state=config.RANDOM_SEED)),
    ])

    logreg = Pipeline([
        ("preprocess", _column_transformer(scale=True)),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
        )),
    ])

    rf = Pipeline([
        ("preprocess", _column_transformer(scale=False)),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )),
    ])

    return {
        "DummyClassifier (prior)": dummy,
        "Regresión logística": logreg,
        "Random Forest": rf,
    }


def _column_transformer(scale: bool) -> Pipeline:
    """Codifica `Type` con one-hot y escala las numéricas si se pide."""
    from sklearn.compose import ColumnTransformer

    num = Pipeline([("scaler", StandardScaler())]) if scale else "passthrough"
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CATEGORICAL_COLS),
        ("num", num, config.NUMERIC_COLS),
    ])


def select_threshold(y_val: np.ndarray, score_val: np.ndarray) -> tuple[float, float]:
    """Elige el umbral que maximiza el F1 de la clase de falla en validación.

    Se prueba una rejilla de umbrales sobre las puntuaciones de validación
    (nunca de prueba). Devuelve (umbral, mejor_f1_val).
    """
    from sklearn.metrics import f1_score

    best_t, best_f1 = config.DEFAULT_THRESHOLD, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_val, (score_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(round(t, 2))
    return best_t, float(best_f1)


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Nombres de las variables tras el preprocesamiento (para importancias)."""
    pre = pipeline.named_steps["preprocess"]
    cat_features = list(pre.named_transformers_["cat"].get_feature_names_out(config.CATEGORICAL_COLS))
    return cat_features + list(config.NUMERIC_COLS)
