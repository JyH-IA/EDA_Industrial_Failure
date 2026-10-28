"""Ingestión, validación y limpieza del dataset AI4I 2020.

Todas las funciones son puras respecto a los datos: no modifican el archivo
fuente y registran cada decisión de limpieza en el informe de calidad.
"""
from __future__ import annotations

import pandas as pd

from src import config


def load_data(path=None) -> pd.DataFrame:
    """Carga el CSV crudo (utf-8-sig elimina el BOM del archivo original)."""
    path = path or config.RAW_DATA_PATH
    return pd.read_csv(path, encoding="utf-8-sig")


def validate_data(df: pd.DataFrame) -> dict:
    """Valida dimensiones, tipos, valores faltantes, duplicados y coherencia.

    Devuelve un diccionario con los hallazgos de calidad de datos. No lanza
    excepciones por diferencias menores: las documenta para el informe.
    """
    checks: dict = {}

    checks["n_rows"], checks["n_cols"] = df.shape
    checks["expected_rows"] = 10_000
    checks["rows_match_source"] = df.shape[0] == 10_000

    checks["missing_values_total"] = int(df.isna().sum().sum())
    checks["duplicate_rows"] = int(df.duplicated().sum())
    checks["duplicate_udis"] = int(df["UDI"].duplicated().sum())

    # La variable objetivo debe ser binaria.
    target_values = sorted(df[config.TARGET].unique().tolist())
    checks["target_values"] = target_values
    checks["target_is_binary"] = set(target_values) <= {0, 1}

    # Las etiquetas de modos de falla deben ser binarias.
    checks["failure_modes_binary"] = all(
        set(df[c].unique()) <= {0, 1} for c in config.FAILURE_MODE_COLS
    )

    # Coherencia: un registro con Machine failure=1 debería tener algún modo
    # de falla activo; y un modo activo implica Machine failure=1. Cualquier
    # discrepancia se reporta (no se corrige automáticamente).
    any_mode = df[config.FAILURE_MODE_COLS].max(axis=1)
    checks["inconsistent_label_plus_mode"] = int(
        ((df[config.TARGET] == 1) & (any_mode == 0)).sum()
    )
    checks["inconsistent_mode_without_label"] = int(
        ((df[config.TARGET] == 0) & (any_mode == 1)).sum()
    )

    # Rangos plausibles de las variables operativas.
    ranges = {
        "Air temperature [K]": (290, 305),
        "Process temperature [K]": (300, 315),
        "Rotational speed [rpm]": (1100, 3000),
        "Torque [Nm]": (0, 100),
        "Tool wear [min]": (0, 300),
    }
    out_of_range = {}
    for col, (lo, hi) in ranges.items():
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n:
            out_of_range[col] = n
    checks["out_of_range_values"] = out_of_range

    checks["type_values"] = sorted(df["Type"].unique().tolist())
    return checks


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el DataFrame de análisis con predictores, objetivo y modos de falla.

    Decisiones de limpieza (documentadas también en data_quality.md):
    - No hay valores faltantes ni duplicados que corregir en este dataset.
    - Se conservan UDI y Product ID solo para trazabilidad; se excluyen de
      los predictores vía config.IDENTIFIER_COLS.
    - Las etiquetas de modos de falla se conservan para el análisis
      descriptivo pero se excluyen del modelado.
    """
    cleaned = df.copy()
    return cleaned


def failure_rate_summary(df: pd.DataFrame) -> dict:
    """Resumen básico de la variable objetivo y su desbalance."""
    n = len(df)
    failures = int(df[config.TARGET].sum())
    return {
        "n_total": n,
        "n_failures": failures,
        "failure_rate_pct": round(100.0 * failures / n, 2),
        "n_no_failure": n - failures,
    }


def make_splits(df: pd.DataFrame) -> dict:
    """Partición estratificada 60/20/20: entrenamiento, validación y prueba.

    La estratificación preserva la proporción de la clase minoritaria en las
    tres particiones. La semilla está fijada en config.RANDOM_SEED.
    """
    from sklearn.model_selection import train_test_split

    y = df[config.TARGET]
    X_index = df.index

    idx_trainval, idx_test = train_test_split(
        X_index,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )
    # Dentro del 80% restante, 25% va a validación -> 20% del total.
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=config.VAL_SIZE / (config.TRAIN_SIZE + config.VAL_SIZE),
        stratify=y.loc[idx_trainval],
        random_state=config.RANDOM_SEED,
    )

    splits = {
        "train": df.loc[idx_train],
        "val": df.loc[idx_val],
        "test": df.loc[idx_test],
    }
    sizes = {name: len(part) for name, part in splits.items()}
    rates = {
        name: round(100.0 * part[config.TARGET].mean(), 2)
        for name, part in splits.items()
    }
    return {"splits": splits, "sizes": sizes, "failure_rates_pct": rates}


def write_data_quality_report(df: pd.DataFrame, checks: dict, summary: dict, split_info: dict, path=None) -> None:
    """Escribe el informe de calidad de datos en Markdown."""
    path = path or config.DATA_QUALITY_PATH
    describe = df[config.NUMERIC_COLS].describe()
    lines = [
        "# Informe de calidad de datos — AI4I 2020",
        "",
        f"- Registros leídos: **{checks['n_rows']}** (esperados: {checks['expected_rows']}; "
        f"coinciden: **{checks['rows_match_source']}**)",
        f"- Valores faltantes: **{checks['missing_values_total']}**",
        f"- Filas duplicadas: **{checks['duplicate_rows']}** (UDI duplicados: {checks['duplicate_udis']})",
        f"- Valores de la variable objetivo: {checks['target_values']} "
        f"(binaria: **{checks['target_is_binary']}**)",
        f"- Tipos de máquina: {', '.join(map(str, checks['type_values']))}",
        f"- Modos de falla binarios: **{checks['failure_modes_binary']}**",
        f"- Registros con `Machine failure=1` sin ningún modo activo: "
        f"**{checks['inconsistent_label_plus_mode']}**",
        f"- Modos de falla activos con `Machine failure=0`: "
        f"**{checks['inconsistent_mode_without_label']}**",
        f"- Valores fuera de rango plausible: "
        f"{checks['out_of_range_values'] or 'ninguno'}",
        "",
        "## Desbalance de clases",
        "",
        f"- Fallas: **{summary['n_failures']}** de {summary['n_total']} registros "
        f"({summary['failure_rate_pct']}%)",
        f"- Sin falla: {summary['n_no_failure']} registros",
        "",
        "## Particiones estratificadas",
        "",
        "| Partición | Registros | % de falla |",
        "|---|---:|---:|",
    ]
    for name in ("train", "val", "test"):
        lines.append(
            f"| {name} | {split_info['sizes'][name]} | "
            f"{split_info['failure_rates_pct'][name]}% |"
        )
    lines += [
        "",
        "## Decisiones de limpieza",
        "",
        "1. El dataset no requiere imputaciones ni eliminación de duplicados.",
        "2. `UDI` y `Product ID` se excluyen de los predictores (identificadores).",
        "3. `TWF`, `HDF`, `PWF`, `OSF` y `RNF` se excluyen de los predictores "
        "para evitar fuga de información hacia `Machine failure`.",
        "4. Las transformaciones (codificación, escalado) se ajustan únicamente "
        "sobre la partición de entrenamiento, dentro de `Pipeline`.",
        "5. Semilla aleatoria fija y documentada: "
        f"`RANDOM_SEED = {config.RANDOM_SEED}`.",
        "",
        "## Nota sobre el objetivo",
        "",
        "No se reconstruye la variable objetivo a partir de los modos de falla: "
        "las discrepancias entre etiquetas se reportan, no se corrigen.",
        "",
        "## Tipos de datos",
        "",
        "| Columna | Tipo |",
        "|---|---|",
    ]
    for col in df.columns:
        lines.append(f"| {col} | {str(df[col].dtype)} |")
    lines += [
        "",
        "## Estadísticas descriptivas (variables numéricas)",
        "",
        "| Estadística | " + " | ".join(config.NUMERIC_COLS) + " |",
        "|---|" + "---:|" * len(config.NUMERIC_COLS),
    ]
    for stat in describe.index:
        values = " | ".join(f"{describe.loc[stat, c]:.2f}" for c in config.NUMERIC_COLS)
        lines.append(f"| {stat} | {values} |")
    path.write_text("\n".join(lines), encoding="utf-8")
