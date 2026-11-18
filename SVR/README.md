# SVR - Regresión por Vectores de Soporte

**Archivo:** `ejemplo_svr.py`

## ¿Qué hace?
Ajusta la curva `y = sin(x) + ruido` (valor continuo) y compara tres kernels: **lineal, polinomial y RBF**.
La SVR traza un **tubo de ancho epsilon** alrededor de la función: los puntos dentro del tubo no se
penalizan, y los que quedan fuera son los **vectores de soporte**.

## Cómo ejecutarlo
```bash
pip install -r requirements.txt
python ejemplo_svr.py
```

## Salida
- En consola: tabla con MSE, R² y número de vectores de soporte por kernel, y predicciones para valores nuevos.
- Gráfica `resultado_svr.png`: la curva de cada kernel con su tubo epsilon y sus vectores de soporte.

## Parámetros para experimentar
| Parámetro | Efecto |
|---|---|
| `C` | Penalización por puntos fuera del tubo. |
| `epsilon` | Ancho del tubo. Más ancho → menos vectores de soporte, modelo más suave. |
| `gamma` (RBF) | Grande → curva muy flexible (riesgo de sobreajuste). |
