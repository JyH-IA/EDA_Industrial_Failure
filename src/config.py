"""Configuración central del proyecto: rutas, semilla, particiones y columnas.

Todas las decisiones metodológicas fijas (semilla, tamaños de partición,
predictores permitidos y excluidos) se declaran aquí para que el flujo
completo sea reproducible y auditable.
"""
from pathlib import Path

# --- Rutas -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "ai4i2020.csv"
DB_PATH = DATA_DIR / "ai4i2020.db"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SQL_RESULTS_DIR = REPORTS_DIR / "sql_results"
METRICS_PATH = REPORTS_DIR / "metrics.json"
DATA_QUALITY_PATH = REPORTS_DIR / "data_quality.md"

SQL_FILE = PROJECT_ROOT / "sql" / "analysis_queries.sql"

# --- Reproducibilidad --------------------------------------------------
RANDOM_SEED = 42  # semilla única y documentada para particiones y modelos

# --- Esquema del dataset ----------------------------------------------
TARGET = "Machine failure"

# Identificadores: nunca se usan como predictores.
IDENTIFIER_COLS = ["UDI", "Product ID"]

# Etiquetas de modos de falla: se excluyen de los predictores para evitar
# fuga de información (el objetivo es predecir `Machine failure`).
FAILURE_MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

CATEGORICAL_COLS = ["Type"]
NUMERIC_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
PREDICTOR_COLS = CATEGORICAL_COLS + NUMERIC_COLS

# --- Particiones -------------------------------------------------------
# Estratificadas por la variable objetivo: 60% entrenamiento, 20% validación,
# 20% prueba. La validación se usa para seleccionar modelo y umbral; la
# prueba solo para el reporte final de métricas.
TRAIN_SIZE = 0.60
VAL_SIZE = 0.20  # proporción sobre el total
TEST_SIZE = 0.20

# --- Umbrales y criterios ---------------------------------------------
# Umbral por defecto si el ajuste fino no mejora el F1 de la clase de falla.
DEFAULT_THRESHOLD = 0.5
