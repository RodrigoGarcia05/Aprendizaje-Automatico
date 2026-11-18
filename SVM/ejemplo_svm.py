"""
=====================================================================
 Aprendizaje Automatico - Ejemplo sencillo de SVM
 (Support Vector Machine / Maquina de Vectores de Soporte)
=====================================================================
 Problema:
   Clasificar si un alumno APRUEBA (1) o REPRUEBA (0) un examen
   usando dos caracteristicas:
       x1 = horas de estudio
       x2 = horas de sueno la noche anterior

 Idea de la SVM:
   Encontrar la recta (hiperplano) que separa las dos clases
   dejando el MAYOR MARGEN posible entre ellas. Los puntos que
   quedan sobre el borde del margen se llaman VECTORES DE SOPORTE
   y son los unicos que definen la frontera.

 Ejecutar:
   pip install -r requirements.txt
   python ejemplo_svm.py
=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# ---------------------------------------------------------------
# 1. DATOS
# ---------------------------------------------------------------
def generar_datos(n=40, semilla=0):
    """Genera datos sinteticos de alumnos que aprueban y reprueban."""
    rng = np.random.default_rng(semilla)

    # Reprueban: pocas horas de estudio y de sueno
    reprueban = rng.normal(loc=[3, 4], scale=1.0, size=(n, 2))
    # Aprueban: mas horas de estudio y de sueno
    aprueban = rng.normal(loc=[7, 7], scale=1.0, size=(n, 2))

    X = np.vstack([reprueban, aprueban])
    y = np.array([0] * n + [1] * n)
    return X, y


# ---------------------------------------------------------------
# 2. MODELO
# ---------------------------------------------------------------
def crear_modelo():
    """
    kernel='linear' -> frontera de decision recta.
    C -> penalizacion por errores. C grande = margen estrecho y
         pocos errores; C pequeno = margen amplio y tolera errores.
    """
    return SVC(kernel="linear", C=1.0)


# ---------------------------------------------------------------
# 3. GRAFICA
# ---------------------------------------------------------------
def graficar(modelo, X, y, archivo="resultado_svm.png"):
    plt.figure(figsize=(8, 6))

    # Puntos de cada clase
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="tab:red", label="Reprueba (0)")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:blue", label="Aprueba (1)")

    # Malla para dibujar la frontera y los margenes
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = modelo.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Z = 0 es la frontera; Z = -1 y Z = +1 son los margenes
    plt.contour(xx, yy, Z, levels=[-1, 0, 1],
                colors="k", linestyles=["--", "-", "--"])

    # Vectores de soporte
    sv = modelo.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], s=150, facecolors="none",
                edgecolors="k", linewidths=1.5, label="Vectores de soporte")

    plt.title("SVM lineal: frontera de decision y margen")
    plt.xlabel("Horas de estudio")
    plt.ylabel("Horas de sueno")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(archivo, dpi=120)
    print(f"\nGrafica guardada en: {archivo}")
    plt.show()


# ---------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------------
def main():
    print("=" * 55)
    print(" EJEMPLO SVM - Clasificacion: aprueba / reprueba")
    print("=" * 55)

    # 1. Datos
    X, y = generar_datos()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    print(f"Datos de entrenamiento: {len(X_train)} | prueba: {len(X_test)}")

    # 2. Entrenamiento
    modelo = crear_modelo()
    modelo.fit(X_train, y_train)

    w = modelo.coef_[0]
    b = modelo.intercept_[0]
    print("\nHiperplano aprendido:  w1*x1 + w2*x2 + b = 0")
    print(f"  w = [{w[0]:.3f}, {w[1]:.3f}]   b = {b:.3f}")
    print(f"  Ancho del margen = 2/||w|| = {2 / np.linalg.norm(w):.3f}")
    print(f"  Numero de vectores de soporte: {len(modelo.support_vectors_)}")

    # 3. Evaluacion
    y_pred = modelo.predict(X_test)
    print(f"\nExactitud (accuracy) en prueba: {accuracy_score(y_test, y_pred):.2%}")
    print("Matriz de confusion:")
    print(confusion_matrix(y_test, y_pred))

    # 4. Prediccion con alumnos nuevos
    nuevos = np.array([[2, 5], [5, 5], [8, 6]])
    print("\nPredicciones para alumnos nuevos:")
    for (estudio, sueno), clase in zip(nuevos, modelo.predict(nuevos)):
        texto = "APRUEBA" if clase == 1 else "REPRUEBA"
        print(f"  Estudio={estudio} h, Sueno={sueno} h  ->  {texto}")

    # 5. Grafica
    graficar(modelo, X_train, y_train)


if __name__ == "__main__":
    main()
