-- ============================================================================
-- Industrial Failure Intelligence — Consultas SQL de análisis (SQLite)
-- ============================================================================
-- Se ejecutan sobre la tabla `observations`, creada por `python -m src.pipeline`
-- en `data/ai4i2020.db`. Cada consulta comienza con un comentario `-- name:`
-- que el pipeline usa para exportar el resultado a reports/sql_results/.
--
-- Convención del proyecto: toda tasa de falla se presenta con su numerador
-- y denominador explícitos junto con el tamaño del grupo, para evitar
-- conclusiones engañosas en grupos pequeños.
-- ============================================================================

-- name: failure_by_type
-- Conteo de registros y fallas por tipo de máquina, con tasa explícita
-- (numerador = fallas, denominador = registros del grupo).
SELECT
    Type,
    COUNT(*)                                  AS total_registros,
    SUM("Machine failure")                    AS total_fallas,
    ROUND(100.0 * SUM("Machine failure") / COUNT(*), 2) AS tasa_falla_pct
FROM observations
GROUP BY Type
ORDER BY Type;

-- name: failure_modes_count
-- Frecuencia de cada modo de falla. Nota: un registro puede presentar más
-- de un modo simultáneamente, por lo que la suma puede exceder el total de
-- fallas. RNF (aleatoria) es la excepción metodológica del dataset original.
SELECT
    SUM(TWF) AS tool_wear_failure,
    SUM(HDF) AS heat_dissipation_failure,
    SUM(PWF) AS power_failure,
    SUM(OSF) AS overstrain_failure,
    SUM(RNF) AS random_failure,
    SUM("Machine failure") AS total_machine_failure
FROM observations;

-- name: summary_by_failure
-- Resumen de variables operativas por grupo de falla (medias).
SELECT
    "Machine failure" AS falla,
    COUNT(*) AS n,
    ROUND(AVG("Air temperature [K]"), 2)      AS temp_aire_k,
    ROUND(AVG("Process temperature [K]"), 2) AS temp_proceso_k,
    ROUND(AVG("Rotational speed [rpm]"), 1) AS velocidad_rpm,
    ROUND(AVG("Torque [Nm]"), 2)            AS torque_nm,
    ROUND(AVG("Tool wear [min]"), 1)         AS desgaste_min
FROM observations
GROUP BY "Machine failure";

-- name: summary_by_type_and_failure
-- Resumen de torque y desgaste por tipo de máquina y grupo de falla,
-- con tamaño de grupo para contextualizar cada promedio.
SELECT
    Type,
    "Machine failure" AS falla,
    COUNT(*) AS n,
    ROUND(AVG("Torque [Nm]"), 2)    AS torque_nm,
    ROUND(AVG("Tool wear [min]"), 1) AS desgaste_min,
    ROUND(AVG("Rotational speed [rpm]"), 1) AS velocidad_rpm
FROM observations
GROUP BY Type, "Machine failure"
ORDER BY Type, falla;

-- name: failure_by_wear_quartile
-- Tasa de falla por cuartil de desgaste de herramienta (numerador y
-- denominador explícitos por cuartil).
SELECT
    quartile              AS cuartil_desgaste,
    COUNT(*)              AS total_registros,
    SUM("Machine failure") AS total_fallas,
    ROUND(100.0 * SUM("Machine failure") / COUNT(*), 2) AS tasa_falla_pct
FROM (
    SELECT "Machine failure",
           NTILE(4) OVER (ORDER BY "Tool wear [min]") AS quartile
    FROM observations
)
GROUP BY quartile
ORDER BY quartile;

-- name: extreme_torque_failures
-- Bandas EXPLORATORIAS de torque (baja < 15 Nm, alta > 55 Nm, intermedia).
-- Los cortes se definen solo para explorar la asociación entre extremos
-- de torque y falla, y no son límites operativos validados del equipo.
SELECT
    CASE
        WHEN "Torque [Nm]" < 15 THEN 'torque_bajo'
        WHEN "Torque [Nm]" > 55 THEN 'torque_alto'
        ELSE 'torque_intermedio'
    END AS banda_torque,
    COUNT(*) AS n_registros,
    SUM("Machine failure") AS n_fallas,
    ROUND(100.0 * SUM("Machine failure") / COUNT(*), 2) AS tasa_falla_pct
FROM observations
GROUP BY banda_torque
ORDER BY tasa_falla_pct DESC;

-- name: cooccurrence_of_modes
-- Registros con más de un modo de falla activo simultáneamente.
SELECT
    COUNT(*) AS registros_con_falla,
    SUM(CASE WHEN (TWF + HDF + PWF + OSF + RNF) > 1 THEN 1 ELSE 0 END) AS multifalla
FROM observations
WHERE "Machine failure" = 1;
