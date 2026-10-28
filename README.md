# Industrial Failure Intelligence

Prototipo de analítica de mantenimiento predictivo que integra preparación de datos, consultas SQL, análisis exploratorio, comparación de modelos de clasificación y un dashboard interactivo. El proyecto evalúa la detección de fallas con métricas sensibles al desbalance de clases y documenta las limitaciones de usar datos sintéticos.

**Dataset:** AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository, licencia CC BY 4.0).
**Stack:** Python, SQL (SQLite), pandas, scikit-learn, Matplotlib/Seaborn y Streamlit.

---

## Estructura del repositorio

```text
industrial-failure-intelligence/
├── app.py                     # Dashboard Streamlit
├── requirements.txt
├── Makefile
├── data/
│   ├── ai4i2020.csv           # Dataset (CC BY 4.0, ver data/README.md)
│   └── README.md
├── notebooks/
│   └── analysis.ipynb         # Análisis exploratorio ejecutado
├── src/
│   ├── config.py              # Rutas, semilla, particiones y columnas
│   ├── preprocessing.py       # Ingestión, validación y limpieza
│   ├── modeling.py            # Pipelines de modelos y selección de umbral
│   ├── evaluation.py          # Métricas y permutation importance
│   ├── figures.py             # Generación de figuras
│   └── pipeline.py            # Pipeline ejecutable de principio a fin
├── sql/
│   └── analysis_queries.sql   # Consultas comentadas (numerador/denominador)
└── reports/
    ├── metrics.json           # Métricas reales de la ejecución
    ├── data_quality.md        # Informe de calidad de datos
    ├── figures/               # Gráficos (EDA y modelo)
    ├── sql_results/           # Salidas de las consultas SQL
    └── screenshots/           # Capturas del dashboard
```

## Instalación y ejecución

Requisitos: Python 3.12 o superior. Entorno verificado: Python 3.14 con pandas 3.0.6, scikit-learn 1.9.1, numpy 2.5.4, matplotlib 3.11.2, seaborn 0.13.2 y streamlit 1.66.1 (Linux).

```bash
# 1. Clonar y crear entorno virtual
git clone https://github.com/<tu-usuario>/industrial-failure-intelligence.git
cd industrial-failure-intelligence
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar el pipeline completo (datos, SQLite, SQL, modelos, figuras)
python -m src.pipeline

# 4. Lanzar el dashboard
streamlit run app.py
```

---

## Metodología

### Preparación de datos

- 10,000 registros, sin valores faltantes ni duplicados (ver [reports/data_quality.md](reports/data_quality.md)).
- **Predictores:** `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.
- **Excluidos:** `UDI` y `Product ID` (identificadores) y `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (etiquetas de modos de falla: incluirlas sería fuga de información).
- **Partición estratificada 60/20/20** (entrenamiento/validación/prueba) que preserva la tasa de falla (~3.4% en las tres particiones).
- Todas las transformaciones (one-hot, escalado) se ajustan solo con entrenamiento, dentro de `Pipeline` de scikit-learn.

### SQL con SQLite

El pipeline crea `data/ai4i2020.db` y ejecuta las consultas comentadas de [sql/analysis_queries.sql](sql/analysis_queries.sql); cada resultado se exporta a `reports/sql_results/`. Todas las tasas se calculan con **numerador y denominador explícitos** junto con el tamaño del grupo.

### Modelado


1. Baseline: `DummyClassifier` (estrategia `prior`).
2. Regresión logística con `class_weight="balanced"` y escalado.
3. Random Forest (300 árboles, `class_weight="balanced"`, `min_samples_leaf=2`).
4. **Selección del modelo:** mayor average precision (PR-AUC) en la partición de **validación**.
5. **Umbral de decisión:** el que maximiza el F1 de la clase de falla en **validación** (nunca en prueba); después se congela.
6. Reporte final en la partición de **prueba**, que no interviene en ninguna decisión.

### Evaluación

Métricas para la clase de falla (positiva): precision, recall, F1, PR-AUC (average precision), matriz de confusión y accuracy como métrica complementaria.

---

## Resultados (partición de prueba, 2,000 registros con 68 fallas)

| Modelo | AP (val.) | AP (prueba) | Precision | Recall | F1 | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| DummyClassifier (prior) | 0.034 | 0.034 | 0.000 | 0.000 | 0.000 | 68 | 0 |
| Regresión logística | 0.368 | 0.393 | 0.147 | 0.824 | 0.249 | 12 | 326 |
| **Random Forest (seleccionado)** | **0.657** | **0.694** | **0.580** | **0.750** | **0.654** | **17** | **37** |

El modelo seleccionado es **Random Forest** con umbral congelado **0.54** (elegido en validación): precision 57.95%, recall 75.00%, F1 0.654 y PR-AUC 0.694 en prueba, con 17 falsos negativos y 37 falsos positivos. La regresión logística detecta más fallas (recall 82%) al precio de 326 falsos positivos: el compromiso precision-recall se aprecia en las curvas PR de `reports/figures/pr_curves.png`.

### Variables más asociadas (permutation importance, prueba)

`Rotational speed [rpm]` (−0.424 de AP al permutar), `Torque [Nm]` (−0.388), `Air temperature [K]` (−0.309) y `Tool wear [min]` (−0.199). Las fallas se concentran en los extremos operativos: en bandas exploratorias de torque, la tasa de falla es 52.6% en la banda baja (<15 Nm) y 24.3% en la alta (>55 Nm), frente a 1.7% en la intermedia; y el desgaste en el cuarto cuartil concentra 6.9% de fallas frente a ~2.1–2.4% en el resto. **Asociación, no causalidad.**

## Hallazgos

1. Solo el **3.39%** de los 10,000 registros (339) presenta falla: el desbalance es marcado y el accuracy es engañoso (el modelo trivial alcanza 96.6%).
2. La tasa de falla difiere por tipo de máquina: **L 3.92%** (235/6,000), **M 2.77%** (83/2,997) y **H 2.09%** (21/1,003). Al comparar medias con y sin falla, las diferencias estandarizadas más marcadas están en **torque** (50.2 vs. 39.6 Nm), **desgaste de herramienta** (143.8 vs. 106.7 min) y **temperatura del aire** (300.9 vs. 300.0 K): asociaciones descriptivas, no causalidad.
3. El Random Forest seleccionado equilibra mejor detección y falsas alarmas (F1 0.654) que la regresión logística (F1 0.249), aunque 17 de 68 fallas de prueba siguen sin detectarse.

Estas cifras provienen de la ejecución del pipeline con semilla 42; cada métrica es calculada durante la ejecución y se registra en `reports/metrics.json`.

## Licencias y cita

- **Código:** MIT (ver [LICENSE](LICENSE)).
- **Dataset:** AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. DOI: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C), licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Referencia del método del dataset: Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications. Third International Conference on Artificial Intelligence for Industries.
