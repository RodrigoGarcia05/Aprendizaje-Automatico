"""
=====================================================================
 Aprendizaje Automatico - Ejemplo sencillo de SVR
 (Support Vector Regression / Regresion por Vectores de Soporte)
=====================================================================
 Problema:
   Predecir un valor CONTINUO. Usamos una curva con ruido:
       y = sin(x) + ruido
   y comparamos tres kernels: lineal, polinomial y RBF.

 Idea de la SVR:
   En lugar de separar clases (como la SVM), la SVR busca una
   funcion que pase lo mas cerca posible de los datos, permitiendo
   un "tubo" de tolerancia de ancho epsilon (e). Los errores dentro
   del tubo NO se penalizan; solo los puntos fuera del tubo (los
   vectores de soporte) influyen en el modelo.

 Ejecutar:
   pip install -r requirements.txt
   python ejemplo_svr.py
=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------------
# 1. DATOS
# ---------------------------------------------------------------
def generar_datos(n=80, semilla=0):
    """Genera puntos de y = sin(x) con un poco de ruido."""
    rng = np.random.default_rng(semilla)
    X = np.sort(rng.uniform(0, 2 * np.pi, n)).reshape(-1, 1)
    y = np.sin(X).ravel() + rng.normal(0, 0.15, n)
    return X, y


# ---------------------------------------------------------------
# 2. MODELOS
# ---------------------------------------------------------------
def crear_modelos():
    """
    C       -> penalizacion por puntos fuera del tubo.
    epsilon -> ancho del tubo de tolerancia.
    gamma   -> (RBF) que tan "local" es la influencia de cada punto.
    """
    return {
        "Lineal":     SVR(kernel="linear", C=1.0, epsilon=0.1),
        "Polinomial": SVR(kernel="poly", degree=3, coef0=1, C=1.0, epsilon=0.1),
        "RBF":        SVR(kernel="rbf", C=10.0, gamma=0.5, epsilon=0.1),
    }


# ---------------------------------------------------------------
# 3. GRAFICA
# ---------------------------------------------------------------
def graficar(modelos, X, y, archivo="resultado_svr.png"):
    X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    colores = {"Lineal": "tab:orange", "Polinomial": "tab:green", "RBF": "tab:blue"}

    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (nombre, modelo) in zip(ejes, modelos.items()):
        y_plot = modelo.predict(X_plot)
        eps = modelo.epsilon

        ax.scatter(X, y, c="gray", s=20, label="Datos")
        ax.plot(X_plot, y_plot, c=colores[nombre], lw=2, label=f"SVR {nombre}")
        # Tubo epsilon
        ax.fill_between(X_plot.ravel(), y_plot - eps, y_plot + eps,
                        color=colores[nombre], alpha=0.2, label="Tubo epsilon")
        # Vectores de soporte
        sv = modelo.support_
        ax.scatter(X[sv], y[sv], s=60, facecolors="none",
                   edgecolors="k", label="Vectores de soporte")

        ax.set_title(f"Kernel {nombre}")
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
    ejes[0].set_ylabel("y")
    ejes[0].legend(loc="lower left", fontsize=8)

    fig.suptitle("SVR: comparacion de kernels sobre y = sin(x) + ruido")
    plt.tight_layout()
    plt.savefig(archivo, dpi=120)
    print(f"\nGrafica guardada en: {archivo}")
    plt.show()


# ---------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------------
def main():
    print("=" * 55)
    print(" EJEMPLO SVR - Regresion: y = sin(x) + ruido")
    print("=" * 55)

    # 1. Datos
    X, y = generar_datos()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    print(f"Datos de entrenamiento: {len(X_train)} | prueba: {len(X_test)}")

    # 2. Entrenamiento y 3. Evaluacion
    modelos = crear_modelos()
    print(f"\n{'Kernel':<12}{'MSE':>10}{'R^2':>10}{'Vect. soporte':>16}")
    print("-" * 48)
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{nombre:<12}{mse:>10.4f}{r2:>10.3f}{len(modelo.support_):>16}")

    # 4. Prediccion con valores nuevos (usando el mejor: RBF)
    nuevos = np.array([[np.pi / 2], [np.pi], [3 * np.pi / 2]])
    print("\nPredicciones con SVR RBF para valores nuevos:")
    for x, pred in zip(nuevos.ravel(), modelos["RBF"].predict(nuevos)):
        print(f"  x = {x:.3f}  ->  y_pred = {pred:+.3f}   (real sin(x) = {np.sin(x):+.3f})")

    # 5. Grafica
    graficar(modelos, X_train, y_train)


if __name__ == "__main__":
    main()
