"""
=====================================================================
 Aprendizaje Automatico - Ejemplo sencillo de Red Neuronal
 con PERCEPTRONES (implementado desde cero con NumPy)
=====================================================================
 Parte 1 - Perceptron simple (una sola neurona):
   Aprende las compuertas logicas AND y OR.
   Tambien se muestra que NO puede aprender XOR, porque XOR no es
   linealmente separable (no existe una recta que separe las clases).

 Parte 2 - Perceptron multicapa (red neuronal):
   Con una capa oculta y el algoritmo de retropropagacion
   (backpropagation) la red SI aprende XOR.

 Formulas del perceptron simple:
   Salida:       y = escalon(w . x + b)
   Aprendizaje:  w = w + tasa * (objetivo - y) * x
                 b = b + tasa * (objetivo - y)

 Ejecutar:
   pip install -r requirements.txt
   python ejemplo_perceptron.py
=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# 1. DATOS (compuertas logicas)
# ---------------------------------------------------------------
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

COMPUERTAS = {
    "AND": np.array([0, 0, 0, 1]),
    "OR":  np.array([0, 1, 1, 1]),
    "XOR": np.array([0, 1, 1, 0]),
}


# ---------------------------------------------------------------
# 2. MODELOS
# ---------------------------------------------------------------
class Perceptron:
    """Perceptron simple de Rosenblatt (una neurona, funcion escalon)."""

    def __init__(self, n_entradas, tasa=0.1, epocas=20):
        self.w = np.zeros(n_entradas)   # pesos
        self.b = 0.0                    # sesgo (bias)
        self.tasa = tasa
        self.epocas = epocas
        self.errores = []               # errores por epoca

    @staticmethod
    def escalon(z):
        return np.where(z >= 0, 1, 0)

    def predecir(self, X):
        return self.escalon(X @ self.w + self.b)

    def entrenar(self, X, y):
        for _ in range(self.epocas):
            errores_epoca = 0
            for xi, objetivo in zip(X, y):
                error = objetivo - self.predecir(xi)
                self.w += self.tasa * error * xi
                self.b += self.tasa * error
                errores_epoca += int(error != 0)
            self.errores.append(errores_epoca)
            if errores_epoca == 0:      # ya clasifica todo bien
                break
        return self


class PerceptronMulticapa:
    """
    Red neuronal 2 -> n_ocultas -> 1 con funcion sigmoide,
    entrenada con retropropagacion (descenso de gradiente).
    """

    def __init__(self, n_entradas=2, n_ocultas=4, tasa=0.5, epocas=10000, semilla=1):
        rng = np.random.default_rng(semilla)
        self.W1 = rng.normal(0, 1, (n_entradas, n_ocultas))
        self.b1 = np.zeros(n_ocultas)
        self.W2 = rng.normal(0, 1, (n_ocultas, 1))
        self.b2 = np.zeros(1)
        self.tasa = tasa
        self.epocas = epocas
        self.errores = []

    @staticmethod
    def sigmoide(z):
        return 1 / (1 + np.exp(-z))

    def propagar(self, X):
        self.h = self.sigmoide(X @ self.W1 + self.b1)        # capa oculta
        return self.sigmoide(self.h @ self.W2 + self.b2)      # capa de salida

    def predecir(self, X):
        return (self.propagar(X) >= 0.5).astype(int).ravel()

    def entrenar(self, X, y):
        y = y.reshape(-1, 1)
        for _ in range(self.epocas):
            # Propagacion hacia adelante
            salida = self.propagar(X)
            error = y - salida
            self.errores.append(float(np.mean(error ** 2)))

            # Retropropagacion (derivada de la sigmoide: s * (1 - s))
            delta_salida = error * salida * (1 - salida)
            delta_oculta = (delta_salida @ self.W2.T) * self.h * (1 - self.h)

            # Actualizacion de pesos
            self.W2 += self.tasa * self.h.T @ delta_salida
            self.b2 += self.tasa * delta_salida.sum(axis=0)
            self.W1 += self.tasa * X.T @ delta_oculta
            self.b1 += self.tasa * delta_oculta.sum(axis=0)
        return self


# ---------------------------------------------------------------
# 3. GRAFICA
# ---------------------------------------------------------------
def dibujar_frontera(ax, modelo, y, titulo):
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    Z = modelo.predecir(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
                colors=["tab:red", "tab:blue"], alpha=0.2)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="tab:red", s=120, edgecolors="k", label="0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:blue", s=120, edgecolors="k", label="1")
    ax.set_title(titulo)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.grid(alpha=0.3)


def graficar(resultados, mlp, archivo="resultado_perceptron.png"):
    fig, ejes = plt.subplots(1, 4, figsize=(18, 4.5))

    for ax, (nombre, (modelo, exactitud)) in zip(ejes, resultados.items()):
        dibujar_frontera(ax, modelo, COMPUERTAS[nombre],
                         f"Perceptron simple - {nombre}\nexactitud {exactitud:.0%}")

    exactitud_mlp = np.mean(mlp.predecir(X) == COMPUERTAS["XOR"])
    dibujar_frontera(ejes[3], mlp, COMPUERTAS["XOR"],
                     f"Perceptron multicapa - XOR\nexactitud {exactitud_mlp:.0%}")
    ejes[0].legend(title="Clase", loc="upper left")

    fig.suptitle("Fronteras de decision aprendidas")
    plt.tight_layout()
    plt.savefig(archivo, dpi=120)
    print(f"\nGrafica guardada en: {archivo}")
    plt.show()


# ---------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------------
def main():
    print("=" * 55)
    print(" EJEMPLO RED NEURONAL - Perceptrones")
    print("=" * 55)

    # ---- Parte 1: perceptron simple ----
    print("\nPARTE 1: Perceptron simple")
    resultados = {}
    for nombre, y in COMPUERTAS.items():
        p = Perceptron(n_entradas=2, tasa=0.1, epocas=20).entrenar(X, y)
        y_pred = p.predecir(X)
        exactitud = np.mean(y_pred == y)
        resultados[nombre] = (p, exactitud)

        print(f"\n  Compuerta {nombre}")
        print(f"    Pesos w = [{p.w[0]:.2f}, {p.w[1]:.2f}]   b = {p.b:.2f}")
        print(f"    Epocas usadas: {len(p.errores)}   Exactitud: {exactitud:.0%}")
        for xi, real, pred in zip(X, y, y_pred):
            marca = "ok" if real == pred else "ERROR"
            print(f"      {xi} -> esperado {real}, predicho {pred}  {marca}")

    print("\n  Conclusion: AND y OR se aprenden al 100%, pero XOR no,")
    print("  porque no se puede separar con una sola recta.")

    # ---- Parte 2: perceptron multicapa ----
    print("\nPARTE 2: Perceptron multicapa (2 -> 4 -> 1) para XOR")
    y_xor = COMPUERTAS["XOR"]
    mlp = PerceptronMulticapa(n_ocultas=4, tasa=0.5, epocas=10000).entrenar(X, y_xor)
    salidas = mlp.propagar(X).ravel()
    print(f"    Error cuadratico medio final: {mlp.errores[-1]:.5f}")
    for xi, real, s in zip(X, y_xor, salidas):
        print(f"      {xi} -> esperado {real}, salida red {s:.3f} -> clase {int(s >= 0.5)}")
    print(f"    Exactitud: {np.mean(mlp.predecir(X) == y_xor):.0%}")

    # ---- Grafica ----
    graficar(resultados, mlp)


if __name__ == "__main__":
    main()
