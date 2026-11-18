# Red Neuronal - Perceptrones

**Archivo:** `ejemplo_perceptron.py` (implementado desde cero con NumPy, sin librerías de ML)

## ¿Qué hace?
1. **Perceptrón simple** (una neurona, función escalón): aprende las compuertas lógicas **AND** y **OR**.
   Se muestra que **no puede aprender XOR**, porque XOR no es linealmente separable.
2. **Perceptrón multicapa** (2 → 4 → 1, función sigmoide, retropropagación): **sí aprende XOR**.

Regla de aprendizaje del perceptrón simple:
```
y = escalon(w · x + b)
w = w + tasa * (objetivo - y) * x
b = b + tasa * (objetivo - y)
```

## Cómo ejecutarlo
```bash
pip install -r requirements.txt
python ejemplo_perceptron.py
```

## Salida
- En consola: pesos aprendidos, épocas usadas y la tabla de verdad predicha para cada compuerta.
- Gráfica `resultado_perceptron.png`: frontera de decisión de cada caso (AND, OR, XOR simple y XOR multicapa).

## Parámetros para experimentar
| Parámetro | Efecto |
|---|---|
| `tasa` | Tamaño del paso de aprendizaje. |
| `epocas` | Número de pasadas por los datos. |
| `n_ocultas` | Neuronas de la capa oculta en el perceptrón multicapa. |
