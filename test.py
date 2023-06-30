import numpy as np

def matriz_rotacion_z(angulo):
    th = angulo
    matriz =  np.array([[np.cos(th), np.sin(th), 0.0], [-np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])

    return matriz


angulo = -1.570796 # Ángulo en grados
matriz_rotacion = matriz_rotacion_z(angulo)

print(matriz_rotacion @ np.array([[2],[-2],[0]]))
print(matriz_rotacion)