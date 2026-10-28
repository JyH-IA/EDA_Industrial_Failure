"""Dashboard Streamlit — Industrial Failure Intelligence.

Presenta los resultados reales generados por `python -m src.pipeline`:
- Resumen: volumen de datos, desbalance y métricas del modelo seleccionado.
- Exploración: distribuciones y comparaciones con filtros interactivos.
- Modelo: comparación de modelos, matriz de confusión y curvas PR.
- Factores asociados: importancia de variables y filtros exploratorios.
- Interpretación: hallazgos, limitaciones y advertencias.

Los filtros solo afectan las vistas exploratorias; las métricas del modelo
provienen de la partición de prueba fija y no cambian con los filtros.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src import config

matplotlib.use("Agg")
st.set_page_config(
    page_title="Industrial Failure Intelligence",
    layout="wide",
)

FIG = config.FIGURES_DIR


@st.cache_data(show_spinner=False)
def load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_DATA_PATH, encoding="utf-8-sig")
    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))


def require_artifacts(metrics: dict | None) -> None:
    if metrics is None:
        st.error(
            "No se encontró `reports/metrics.json`. Ejecuta primero el pipeline: "
            "`python -m src.pipeline` y vuelve a cargar esta página."
        )
        st.stop()


def header(metrics: dict) -> None:
    st.title("Industrial Failure Intelligence")
    st.caption(
        "Prototipo de analítica de mantenimiento predictivo sobre el dataset "
        "sintético AI4I 2020 (UCI). Los resultados presentados provienen de una "
        "ejecución reproducible del pipeline; nada está precargado."
    )


def tab_resumen(df: pd.DataFrame, metrics: dict) -> None:
    summary = metrics["failure_summary"]
    sel = metrics["selected_model"]
    m = metrics["test"][sel]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observaciones totales", f"{summary['n_total']:,}")
    col2.metric(
        "Registros con falla",
        f"{summary['n_failures']:,}",
        f"{summary['failure_rate_pct']}% del total",
    )
    col3.metric("Modelo seleccionado", sel)
    col4.metric("Umbral (congelado en validación)", f"{metrics['threshold']:.2f}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision (falla)", f"{m['precision']:.1%}")
    col2.metric("Recall (falla)", f"{m['recall']:.1%}")
    col3.metric("F1-score (falla)", f"{m['f1']:.3f}")
    col4.metric("PR-AUC (prueba)", f"{m['average_precision']:.3f}")

    st.markdown(
        "Métricas calculadas sobre la **partición de prueba fija** "
        f"({m['n_total']:,} registros, {m['n_positive']} fallas). "
        f"Falsos negativos: **{m['confusion_matrix']['fn']}** · "
        f"falsos positivos: **{m['confusion_matrix']['fp']}**."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Balance de la variable objetivo")
        st.image(str(FIG / "target_balance.png"))
    with right:
        st.subheader("Tasa de falla por tipo de máquina")
        st.image(str(FIG / "failure_by_type.png"))


def tab_exploracion(df: pd.DataFrame) -> None:
    st.subheader("Exploración con filtros")
    st.caption(
        "Los filtros solo afectan esta vista exploratoria; no modifican las "
        "métricas del modelo (sección Modelo)."
    )

    with st.expander("Filtros", expanded=True):
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        types = sorted(df["Type"].unique())
        chosen_types = fcol1.multiselect("Tipo de máquina", types, default=types)
        torque_max = float(df["Torque [Nm]"].max())
        wear_max = float(df["Tool wear [min]"].max())
        torque_range = fcol2.slider(
            "Torque [Nm]", 0.0, torque_max, (0.0, torque_max)
        )
        wear_range = fcol3.slider("Desgaste de herramienta [min]", 0.0, wear_max, (0.0, wear_max))
        rpm_range = fcol4.slider(
            "Velocidad de rotación [rpm]",
            float(df["Rotational speed [rpm]"].min()),
            float(df["Rotational speed [rpm]"].max()),
            (
                float(df["Rotational speed [rpm]"].min()),
                float(df["Rotational speed [rpm]"].max()),
            ),
        )

    filtered = df[
        df["Type"].isin(chosen_types)
        & df["Torque [Nm]"].between(*torque_range)
        & df["Tool wear [min]"].between(*wear_range)
        & df["Rotational speed [rpm]"].between(*rpm_range)
    ].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Registros filtrados", f"{len(filtered):,}")
    if len(filtered):
        rate = 100.0 * filtered[config.TARGET].mean()
        col2.metric("Fallas en el filtro", f"{int(filtered[config.TARGET].sum()):,}")
        col3.metric("Tasa de falla del filtro", f"{rate:.2f}%")
    else:
        st.info("No hay registros con los filtros actuales.")
        return

    variable = st.selectbox("Variable operativa a examinar", config.NUMERIC_COLS)

    fig, ax = plt.subplots(figsize=(7, 4))
    for value, label, color in [(0, "Sin falla", "#4c72b0"), (1, "Falla", "#c44e52")]:
        subset = filtered.loc[filtered[config.TARGET] == value, variable]
        ax.hist(subset, bins=30, alpha=0.6, density=True, label=label, color=color)
    ax.set_title(f"{variable} — distribución por condición de falla (con filtros)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    left, right = st.columns(2)
    with left:
        st.subheader("Comparación sin falla vs. falla (con filtros)")
        fig, ax = plt.subplots(figsize=(6.5, 4))
        sns.boxplot(data=filtered, x=config.TARGET, y=variable, hue=config.TARGET,
                    ax=ax, legend=False, palette=["#4c72b0", "#c44e52"])
        ax.set_xlabel("Falla")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Correlaciones (con filtros)")
        fig, ax = plt.subplots(figsize=(6.5, 5))
        cols = config.NUMERIC_COLS + [config.TARGET]
        sns.heatmap(filtered[cols].corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
        st.pyplot(fig)
        plt.close(fig)


def tab_modelo(metrics: dict) -> None:
    st.subheader("Comparación de modelos")
    st.caption(
        "Selección por average precision en la partición de **validación**; "
        "reporte final en la partición de **prueba**. El umbral del modelo "
        "seleccionado se ajustó solo con validación y luego se congeló."
    )

    rows = []
    for name, val in metrics["validation"].items():
        test = metrics["test"][name]
        rows.append(
            {
                "Modelo": name,
                "AP (validación)": val["average_precision"],
                "AP (prueba)": test["average_precision"],
                "Precision": test["precision"],
                "Recall": test["recall"],
                "F1 (falla)": test["f1"],
                "Accuracy": test["accuracy"],
                "Falsos negativos": test["confusion_matrix"]["fn"],
                "Falsos positivos": test["confusion_matrix"]["fp"],
            }
        )
    st.dataframe(pd.DataFrame(rows).set_index("Modelo"), width="stretch")

    sel = metrics["selected_model"]
    left, right = st.columns(2)
    with left:
        st.subheader(f"Matriz de confusión — {sel}")
        st.image(str(FIG / "confusion_matrix.png"))
        cm = metrics["test"][sel]["confusion_matrix"]
        st.markdown(
            f"**{cm['fn']} falsos negativos** (fallas no detectadas) frente a "
            f"**{cm['fp']} falsos positivos** (alertas sin falla real). "
            "En mantenimiento predictivo el costo de un falso negativo suele "
            "ser mayor: una falla no detectada puede detener la producción."
        )
    with right:
        st.subheader("Curvas Precision-Recall (prueba)")
        st.image(str(FIG / "pr_curves.png"))
        st.markdown(
            "La curva muestra el compromiso entre detectar más fallas "
            "(recall) y generar falsas alarmas (precision). El punto de "
            f"operación del modelo seleccionado usa el umbral congelado "
            f"({metrics['threshold']:.2f}), elegido en validación."
        )


def tab_factores(df: pd.DataFrame, metrics: dict) -> None:
    st.subheader("Factores asociados a las fallas")
    st.caption(
        "La permutation importance se calculó sobre la partición de prueba del "
        "modelo ya congelado, como análisis descriptivo complementario. "
        "**La importancia no demuestra causalidad.**"
    )

    left, right = st.columns([3, 2])
    with left:
        st.image(str(FIG / "permutation_importance.png"))
    with right:
        st.markdown("**Explora la relación entre torque, velocidad y falla**")
        types = sorted(df["Type"].unique())
        chosen = st.multiselect("Tipo de máquina (gráfico)", types, default=types)
        sample = df[df["Type"].isin(chosen)].sample(
            min(2500, len(df[df["Type"].isin(chosen)])),
            random_state=config.RANDOM_SEED,
        )
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for value, label, color in [(0, "Sin falla", "#4c72b0"), (1, "Falla", "#c44e52")]:
            subset = sample[sample[config.TARGET] == value]
            ax.scatter(
                subset["Rotational speed [rpm]"], subset["Torque [Nm]"],
                alpha=0.35, s=16, label=label, color=color,
            )
        ax.set_xlabel("Velocidad de rotación [rpm]")
        ax.set_ylabel("Torque [Nm]")
        ax.set_title("Torque vs. velocidad de rotación")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown(
            "Las fallas tienden a concentrarse en los extremos operativos "
            "(torque alto con baja velocidad, o torque bajo con alta "
            "velocidad) — una asociación observable en los datos, no una "
            "relación causal demostrada."
        )


def tab_interpretacion(metrics: dict) -> None:
    st.subheader("Hallazgos")
    for finding in metrics["findings"]:
        st.markdown(f"- {finding}")

    st.subheader("Limitaciones")
    limitations = [
        "- El dataset **AI4I 2020 es sintético**: simula condiciones de "
        "maquinaria y no proviene de sensores reales. Las métricas no son "
        "evidencia de desempeño en planta.",
        "- La evaluación usa una única partición fija (sin intervalos de "
        "confianza ni validación cruzada repetida), adecuada para un "
        "prototipo pero no para un sistema en producción.",
        "- El umbral y la selección del modelo se hicieron con validación; "
        "aun así, resultados ligeramente distintos con otras semillas son "
        "esperables dado el pequeño número de fallas (339).",
        "- Las importancias describen asociaciones del modelo con estos "
        "datos; no establecen causalidad.",
        "- No se optimizaron hiperparámetros de forma exhaustiva (fuera de "
        "alcance del prototipo).",
    ]
    st.markdown("\n".join(limitations))

    st.warning(
        "**Advertencia:** este prototipo demuestra un flujo analítico "
        "reproducible. No debe utilizarse como sistema operativo de "
        "mantenimiento ni para tomar decisiones sobre maquinaria real sin "
        "validación adicional con datos industriales."
    )


def main() -> None:
    df = load_dataframe()
    metrics = load_metrics() if config.METRICS_PATH.exists() else None
    require_artifacts(metrics)

    header(metrics)
    tab_res, tab_exp, tab_mod, tab_fac, tab_int = st.tabs(
        ["Resumen", "Exploración", "Modelo", "Factores asociados", "Interpretación"]
    )
    with tab_res:
        tab_resumen(df, metrics)
    with tab_exp:
        tab_exploracion(df)
    with tab_mod:
        tab_modelo(metrics)
    with tab_fac:
        tab_factores(df, metrics)
    with tab_int:
        tab_interpretacion(metrics)

    st.divider()
    st.caption(
        "Industrial Failure Intelligence · Dataset AI4I 2020 (UCI, CC BY 4.0) · "
        "Código y metodología en el repositorio de GitHub."
    )


if __name__ == "__main__":
    main()
