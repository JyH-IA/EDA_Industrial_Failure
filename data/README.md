# Datos del proyecto

## AI4I 2020 Predictive Maintenance Dataset

- **Fuente oficial:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Cita sugerida:** AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. DOI: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)
- **Licencia del dataset:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Permite compartir y adaptar para cualquier propósito con atribución.

## Contenido de esta carpeta

- `ai4i2020.csv` — copia del dataset original (10,000 registros, ~500 KB), incluida en el repositorio por comodidad y porque su licencia CC BY 4.0 permite la redistribución con atribución. Si prefieres descargarla desde la fuente oficial, elimina el archivo y ejecuta el pipeline tras descargarlo de nuevo con el mismo nombre.

## Notas

- La base de datos `ai4i2020.db` (SQLite) se genera automáticamente con `python -m src.pipeline` y no se versiona.
- El dataset es **sintético**: fue diseñado para investigación en mantenimiento predictivo y no representa mediciones reales de planta.
