# Informe de calidad de datos — AI4I 2020

- Registros leídos: **10000** (esperados: 10000; coinciden: **True**)
- Valores faltantes: **0**
- Filas duplicadas: **0** (UDI duplicados: 0)
- Valores de la variable objetivo: [0, 1] (binaria: **True**)
- Tipos de máquina: H, L, M
- Modos de falla binarios: **True**
- Registros con `Machine failure=1` sin ningún modo activo: **9**
- Modos de falla activos con `Machine failure=0`: **18**
- Valores fuera de rango plausible: ninguno

## Desbalance de clases

- Fallas: **339** de 10000 registros (3.39%)
- Sin falla: 9661 registros

## Particiones estratificadas

| Partición | Registros | % de falla |
|---|---:|---:|
| train | 6000 | 3.38% |
| val | 2000 | 3.4% |
| test | 2000 | 3.4% |

## Decisiones de limpieza

1. El dataset no requiere imputaciones ni eliminación de duplicados.
2. `UDI` y `Product ID` se excluyen de los predictores (identificadores).
3. `TWF`, `HDF`, `PWF`, `OSF` y `RNF` se excluyen de los predictores para evitar fuga de información hacia `Machine failure`.
4. Las transformaciones (codificación, escalado) se ajustan únicamente sobre la partición de entrenamiento, dentro de `Pipeline`.
5. Semilla aleatoria fija y documentada: `RANDOM_SEED = 42`.

## Nota sobre el objetivo

No se reconstruye la variable objetivo a partir de los modos de falla: las discrepancias entre etiquetas se reportan, no se corrigen.

## Tipos de datos

| Columna | Tipo |
|---|---|
| UDI | int64 |
| Product ID | str |
| Type | str |
| Air temperature [K] | float64 |
| Process temperature [K] | float64 |
| Rotational speed [rpm] | int64 |
| Torque [Nm] | float64 |
| Tool wear [min] | int64 |
| Machine failure | int64 |
| TWF | int64 |
| HDF | int64 |
| PWF | int64 |
| OSF | int64 |
| RNF | int64 |

## Estadísticas descriptivas (variables numéricas)

| Estadística | Air temperature [K] | Process temperature [K] | Rotational speed [rpm] | Torque [Nm] | Tool wear [min] |
|---|---:|---:|---:|---:|---:|
| count | 10000.00 | 10000.00 | 10000.00 | 10000.00 | 10000.00 |
| mean | 300.00 | 310.01 | 1538.78 | 39.99 | 107.95 |
| std | 2.00 | 1.48 | 179.28 | 9.97 | 63.65 |
| min | 295.30 | 305.70 | 1168.00 | 3.80 | 0.00 |
| 25% | 298.30 | 308.80 | 1423.00 | 33.20 | 53.00 |
| 50% | 300.10 | 310.10 | 1503.00 | 40.10 | 108.00 |
| 75% | 301.50 | 311.10 | 1612.00 | 46.80 | 162.00 |
| max | 304.50 | 313.80 | 2886.00 | 76.60 | 253.00 |