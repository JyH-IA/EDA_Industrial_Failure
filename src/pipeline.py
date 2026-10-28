"""Pipeline ejecutable de principio a fin: `python -m src.pipeline`.

Etapas (en orden):
1. Ingestión, validación y reporte de calidad de datos.
2. Partición estratificada 60/20/20 (entrenamiento / validación / prueba).
3. Base de datos SQLite y ejecución de las consultas de `sql/analysis_queries.sql`.
4. Figuras del análisis exploratorio.
5. Entrenamiento de los tres modelos y selección por average precision en validación.
6. Ajuste del umbral SOLO con validación, congelamiento y reporte en prueba.
7. Permutation importance sobre prueba (análisis descriptivo).
8. Cálculo de hallazgos y escritura de `reports/metrics.json`.

Todas las cifras que aparecen en el dashboard y el README provienen de esta
ejecución; nada está precargado.
"""
from __future__ import annotations

import json
import platform
import re
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np

from src import config, figures
from src.evaluation import (
    compute_metrics,
    permutation_importance_scores,
    pr_curve_points,
)
from src.modeling import build_model_pipelines, select_threshold
from src.preprocessing import (
    failure_rate_summary,
    load_data,
    make_splits,
    validate_data,
    write_data_quality_report,
)

SQL_NAME_RE = re.compile(r"--\s*name:\s*([a-z0-9_]+)")


def build_database(df) -> None:
    """Crea la base SQLite `data/ai4i2020.db` con la tabla `observations`."""
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
    with sqlite3.connect(config.DB_PATH) as conn:
        df.to_sql("observations", conn, index=False, if_exists="replace")
        conn.execute(
            "CREATE INDEX idx_obs_type_failure ON observations(Type, \"Machine failure\")"
        )


def run_sql_queries() -> dict[str, int]:
    """Ejecuta las consultas comentadas de `sql/analysis_queries.sql`.

    Cada bloque con `-- name: xxx` se exporta a `reports/sql_results/xxx.csv`
    y devuelve el número de filas por consulta (verificable contra pandas).
    """
    import pandas as pd

    config.SQL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    text = config.SQL_FILE.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split(";") if b.strip()]
    results = {}
    with sqlite3.connect(config.DB_PATH) as conn:
        for block in blocks:
            match = SQL_NAME_RE.search(block)
            if not match:
                continue
            name = match.group(1)
            out = pd.read_sql_query(block, conn)
            out.to_csv(config.SQL_RESULTS_DIR / f"{name}.csv", index=False)
            results[name] = len(out)
    return results


def train_and_evaluate(splits: dict) -> dict:
    """Entrena, selecciona por validación y reporta en prueba.

    Reglas del protocolo (fijadas antes de ver resultados):
    - Selección del modelo: mayor average precision en VALIDACIÓN.
    - Umbral: el que maximiza el F1 de la clase de falla en VALIDACIÓN.
    - La partición de prueba no interviene en ninguna decisión.
    """
    train, val, test = splits["train"], splits["val"], splits["test"]

    pipelines = build_model_pipelines()
    validation_scores: dict = {}
    fitted: dict = {}

    for name, pipe in pipelines.items():
        pipe.fit(train[config.PREDICTOR_COLS], train[config.TARGET])
        score_val = pipe.predict_proba(val[config.PREDICTOR_COLS])[:, 1]
        from sklearn.metrics import average_precision_score

        ap_val = average_precision_score(val[config.TARGET], score_val)
        threshold, f1_val = select_threshold(val[config.TARGET].to_numpy(), score_val)
        validation_scores[name] = {
            "average_precision": round(float(ap_val), 4),
            "best_threshold": threshold,
            "f1_val_at_threshold": round(f1_val, 4),
        }
        fitted[name] = pipe

    selected = max(validation_scores, key=lambda k: validation_scores[k]["average_precision"])
    threshold = validation_scores[selected]["best_threshold"]

    test_results: dict = {}
    pr_curves: dict = {}
    for name, pipe in fitted.items():
        score_test = pipe.predict_proba(test[config.PREDICTOR_COLS])[:, 1]
        # Para los modelos no seleccionados se usa el umbral por defecto 0.5,
        # salvo el seleccionado, que usa su umbral congelado de validación.
        thr = threshold if name == selected else config.DEFAULT_THRESHOLD
        test_results[name] = compute_metrics(
            test[config.TARGET].to_numpy(), score_test, thr
        )
        pr_curves[name] = pr_curve_points(test[config.TARGET].to_numpy(), score_test)

    return {
        "pipelines": fitted,
        "validation": validation_scores,
        "selected_model": selected,
        "threshold": threshold,
        "test": test_results,
        "pr_curves": pr_curves,
    }


def compute_findings(df, model_results: dict, importance: dict) -> list[str]:
    """Tres hallazgos derivados exclusivamente de los resultados reales."""
    summary = failure_rate_summary(df)
    by_type = df.groupby("Type")[config.TARGET].agg(["sum", "count"])
    by_type = by_type.assign(rate=100 * by_type["sum"] / by_type["count"])
    top_type = by_type["rate"].idxmax()
    low_type = by_type["rate"].idxmin()

    means = df.groupby(config.TARGET)[config.NUMERIC_COLS].mean()
    # Diferencia estandarizada (d de Cohen) para ordenar de forma comparable
    # entre variables con unidades distintas; se reporta con dirección y medias.
    stds = df.groupby(config.TARGET)[config.NUMERIC_COLS].std()
    pooled = np.sqrt((stds.loc[0] ** 2 + stds.loc[1] ** 2) / 2)
    diffs = (means.loc[1] - means.loc[0]) / pooled
    top = diffs.reindex(diffs.abs().sort_values(ascending=False).index)[:3]
    var_text = ", ".join(
        f"{col} ({'mayor' if d > 0 else 'menor'}: {means.loc[1, col]:.1f} "
        f"vs. {means.loc[0, col]:.1f})"
        for col, d in top.items()
    )

    sel = model_results["selected_model"]
    thr = model_results["threshold"]
    m = model_results["test"][sel]
    imp_names, imp_means = importance["names"], importance["means"]
    top_feature = imp_names[int(np.argmax(imp_means))]

    return [
        f"El {summary['failure_rate_pct']}% de los {summary['n_total']:,} registros "
        f"({summary['n_failures']} casos) presenta falla. El desbalance es marcado, por lo "
        f"que el accuracy no es una métrica informativa: el modelo trivial que nunca "
        f"predice falla alcanza ~{round(100 - summary['failure_rate_pct'], 2)}% de accuracy.",
        f"La tasa de falla difiere por tipo de máquina: {top_type} presenta la tasa más "
        f"alta ({by_type.loc[top_type, 'rate']:.2f}%) y {low_type} la más baja "
        f"({by_type.loc[low_type, 'rate']:.2f}%), con tamaños de grupo explícitos en las "
        f"consultas SQL. Al comparar medias de registros con y sin falla, las "
        f"diferencias más marcadas (estandarizadas) están en {var_text}; "
        f"son asociaciones descriptivas, no evidencia de causalidad.",
        f"El modelo seleccionado ({sel}, umbral {thr:.2f} elegido en validación) obtiene "
        f"en prueba: precision {m['precision']:.2%}, recall {m['recall']:.2%}, F1 {m['f1']:.3f} "
        f"y PR-AUC {m['average_precision']:.3f}, con {m['confusion_matrix']['fn']} falsos "
        f"negativos y {m['confusion_matrix']['fp']} falsos positivos. La variable con mayor "
        f"permutation importance es {top_feature}.",
    ]


def main() -> None:
    started = datetime.now(timezone.utc)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.SQL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/7] Carga y validación de datos")
    df = load_data()
    checks = validate_data(df)
    summary = failure_rate_summary(df)
    split_info = make_splits(df)
    write_data_quality_report(df, checks, summary, split_info)
    print(f"      {summary['n_failures']} fallas de {summary['n_total']} registros "
          f"({summary['failure_rate_pct']}%)")

    print("[2/7] Base de datos SQLite y consultas SQL")
    build_database(df)
    sql_counts = run_sql_queries()
    print(f"      {len(sql_counts)} consultas exportadas: {sql_counts}")

    print("[3/7] Figuras del análisis exploratorio")
    figures.plot_target_balance(df)
    figures.plot_failure_by_type(df)
    figures.plot_numeric_distributions(df)
    figures.plot_boxplots_by_failure(df)
    figures.plot_correlation_heatmap(df)
    figures.plot_scatter_torque_rpm(df)

    print("[4/7] Entrenamiento y selección de modelos (validación)")
    model_results = train_and_evaluate(split_info["splits"])
    selected, threshold = model_results["selected_model"], model_results["threshold"]
    print(f"      Seleccionado: {selected} | umbral congelado: {threshold:.2f}")
    for name, m in model_results["validation"].items():
        print(f"      [val] {name}: AP={m['average_precision']:.4f} F1@thr={m['f1_val_at_threshold']:.4f}")

    print("[5/7] Evaluación final en la partición de prueba")
    for name, m in model_results["test"].items():
        cm = m["confusion_matrix"]
        print(f"      [test] {name}: P={m['precision']:.4f} R={m['recall']:.4f} "
              f"F1={m['f1']:.4f} AP={m['average_precision']:.4f} "
              f"(FN={cm['fn']}, FP={cm['fp']})")

    print("[6/7] Figuras del modelo y permutation importance")
    figures.plot_pr_curves(model_results["pr_curves"], selected, threshold)
    sel_metrics = model_results["test"][selected]
    figures.plot_confusion_matrix(sel_metrics["confusion_matrix"], selected)

    test = split_info["splits"]["test"]
    pipe = model_results["pipelines"][selected]
    importances = permutation_importance_scores(
        pipe, test[config.PREDICTOR_COLS], test[config.TARGET],
        n_repeats=10, seed=config.RANDOM_SEED,
    )
    # La permutación actúa sobre las columnas de entrada del pipeline,
    # por lo que las etiquetas son los predictores sin expandir.
    feature_names = list(config.PREDICTOR_COLS)
    figures.plot_permutation_importance(feature_names, importances["means"], importances["stds"])

    print("[7/7] Hallazgos y metrics.json")
    findings = compute_findings(df, model_results, {"names": feature_names, **importances})
    for i, f in enumerate(findings, 1):
        print(f"      Hallazgo {i}: {f[:120]}...")

    import sklearn
    import pandas
    report = {
        "metadata": {
            "generated_at_utc": started.isoformat(),
            "random_seed": config.RANDOM_SEED,
            "split": {
                "train": split_info["sizes"]["train"],
                "val": split_info["sizes"]["val"],
                "test": split_info["sizes"]["test"],
                "failure_rate_pct": split_info["failure_rates_pct"],
            },
            "predictors": config.PREDICTOR_COLS,
            "excluded": {
                "identifiers": config.IDENTIFIER_COLS,
                "failure_modes": config.FAILURE_MODE_COLS,
            },
            "versions": {
                "python": platform.python_version(),
                "pandas": pandas.__version__,
                "scikit-learn": sklearn.__version__,
                "numpy": np.__version__,
            },
        },
        "failure_summary": summary,
        "validation": model_results["validation"],
        "selected_model": selected,
        "threshold": threshold,
        "test": model_results["test"],
        "pr_curves_test": model_results["pr_curves"],
        "permutation_importance": {
            "names": feature_names,
            "means": importances["means"],
            "stds": importances["stds"],
        },
        "sql_row_counts": sql_counts,
        "findings": findings,
    }
    config.METRICS_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"      Métricas escritas en {config.METRICS_PATH}")
    print("Pipeline completado.")


if __name__ == "__main__":
    sys.exit(main())
