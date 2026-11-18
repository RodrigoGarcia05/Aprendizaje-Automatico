# SVM - Máquina de Vectores de Soporte (clasificación)

**Archivo:** `ejemplo_svm.py`

## ¿Qué hace?
Clasifica alumnos en **Aprueba / Reprueba** a partir de horas de estudio y horas de sueño.
La SVM busca la recta que separa ambas clases con el **mayor margen** posible; los puntos
sobre el margen son los **vectores de soporte**.

## Cómo ejecutarlo
```bash
pip install -r requirements.txt
python ejemplo_svm.py
```

## Salida
- En consola: pesos del hiperplano (`w`, `b`), ancho del margen, exactitud, matriz de confusión y predicciones para alumnos nuevos.
- Gráfica `resultado_svm.png`: frontera de decisión (línea continua), márgenes (líneas punteadas) y vectores de soporte (círculos).

## Parámetros para experimentar
| Parámetro | Efecto |
|---|---|
| `C` | Grande → margen estrecho, pocos errores. Pequeño → margen amplio, tolera errores. |
| `kernel` | `"linear"`, `"rbf"`, `"poly"` para fronteras no lineales. |
