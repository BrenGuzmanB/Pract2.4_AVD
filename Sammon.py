"""
Created on Fri Nov  3 22:53:11 2023

@author: Bren Guzmán

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(5)

'''
MAPEO DE SAMMON: PROCEDIMIENTO DE MITADES

PARÁMETROS:
X: dataframe con las características
n: número de dimensiones
Y: Etiquetas de X
max_iterations: Número máximo de iteraciones
maxhalves: Número máximo de mitades
convergence_threshold: umbral de convergencia

SALIDA:
X_mapped: Mapeo de sammon a n dimensiones
E: Valor del estrés

EJEMPLO DE USO:
X_mapped, E = Sammon(X, 2, Y)
'''

def Sammon_Mapping(X, n, Y, max_iterations = 500, maxhalves = 20, convergence_threshold = 1e-4):
    
    # Crear matriz de distancias
    D = cdist(X, X)
    
    # Inicialización
    N = X.shape[0]
    escala = 0.5 / D.sum()
    D = D + np.eye(N)

    Dinv = 1 / D
    X_mapped = np.random.normal(0.0, 1.0, [N, n])

    uno = np.ones([N, n])
    d = cdist(X_mapped, X_mapped) + np.eye(N)
    dinv = 1. / d
    delta = D - d
    E = ((delta ** 2) * Dinv).sum()
    
    for i in range(max_iterations):
        delta = dinv - Dinv
        deltauno = np.dot(delta, uno)
        g = np.dot(delta, X_mapped) - (X_mapped * deltauno)
        dinv3 = dinv ** 3
        X_mapped2 = X_mapped ** 2
        H = np.dot(dinv3, X_mapped2) - deltauno - np.dot(2, X_mapped) * np.dot(dinv3, X_mapped) + X_mapped2 * np.dot(dinv3, uno)
        s = -g.flatten(order='F') / np.abs(H.flatten(order='F'))
        X_mapped_antiguo = X_mapped

        # Usar procedimiento de mitades para garantizar el progreso
        for j in range(maxhalves):
            s_reshape = np.reshape(s, (-1, n), order='F')
            X_mapped = X_mapped_antiguo + s_reshape
            d = cdist(X_mapped, X_mapped) + np.eye(N)
            dinv = 1 / d
            delta = D - d
            E_nuevo = ((delta ** 2) * Dinv).sum()
            if E_nuevo < E:
                break
            else:
                s = 0.5 * s

        # Salir si se requieren demasiadas iteraciones de mitad
        if j == maxhalves - 1:
            print('Se superó el límite de iteraciones de mitad. El mapeo de Sammon puede no converger...')

        # Evaluar el criterio de terminación
        if abs((E - E_nuevo) / E) < convergence_threshold:
            print('Optimización terminada')
            break

        # Imprimir información
        E = E_nuevo
        print('Iteración = %d : E = %12.10f' % (i + 1, E * escala))

        # Visualización en cada iteración
        plt.scatter(X_mapped[:, 0], X_mapped[:, 1], c=Y)
        plt.title(f"Iteración {i + 1}")
        plt.show()

    if i == max_iterations - 1:
        print(f"No se alcanzó la convergencia después de {max_iterations} iteraciones.")

    # Ajustar el estrés
    E = E * escala

    return [X_mapped, E]
