import numpy as np

def matriz_rotacion_z(angulo):
    radianes = np.radians(angulo)
    coseno = np.cos(radianes)
    seno = np.sin(radianes)

    matriz = np.array([[coseno, seno, 0],
                      [-seno, coseno, 0],
                      [0, 0, 1]])

    return matriz


angulo = -90  # Ángulo en grados
matriz_rotacion = matriz_rotacion_z(angulo)

print(matriz_rotacion @ np.array([[2],[-2],[0]]))
print(matriz_rotacion)