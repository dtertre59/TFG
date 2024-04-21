import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Supongamos que pose_matrix es tu matriz de transformación
pose_matrix = np.array([[ 0.98949365, -0.02275008,  0.14277518, -0.04343225],
                        [-0.11622167,  0.46221815,  0.87911712,  0.04289319],
                        [-0.08599327, -0.88647438,  0.45471787,  1.52128815],
                        [ 0,         0,          0,          1        ]])

# Extraer la parte de rotación y traslación de la matriz de transformación
pose_R = pose_matrix[:3, :3]
pose_t = pose_matrix[:3, 3]

# Invertir la matriz de transformación
inverse_pose_matrix = np.linalg.inv(pose_matrix)

print(inverse_pose_matrix)

# Definir ejes de la cámara
apriltag_axis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Ejes unitarios de la cámara
print('Ejes april: ', apriltag_axis)

# Aplicar transformación de rotación a los ejes de la cámara
cam_axis = np.dot(pose_R, apriltag_axis.T).T
print('Ejes camera: ', cam_axis.T)


# La última columna de la matriz inversa (las primeras tres son la rotación y la última es la traslación) te da la posición de la cámara con respecto al AprilTag
camera_position = inverse_pose_matrix[:3, 3]
print("Posición de la cámara con respecto al AprilTag:", camera_position)

# Posicion de los ejes antiguos respecto al apriltag



# La parte de traslación de la matriz de transformación representa la posición del AprilTag con respecto a la cámara
apriltag_position = pose_matrix[:3, 3]
print("Posición del AprilTag con respecto a la cámara:", apriltag_position)

# ----- CALCULOS RESPECTO AL APRILTAG -----

apriltag_position = np.array([0,0,0])
print("Posición del AprilTag con respecto al apriltag:", apriltag_position)

# Crear una figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



points = [camera_position, apriltag_position, (1.5,1.5,1.5), (-1.5,-1.5,-1.5)]
print(points)
# Trazar un mini eje de coordenadas en cada punto
counter = 0
for point in points:
    if counter < 1: #camera axis
        ax.scatter(point[0], point[1], point[2], c='b', marker='o', label='Camara')
        ax.quiver(point[0], point[1], point[2], cam_axis[0][0], cam_axis[0][1], cam_axis[0][2], length=0.5, color='r')  # Eje X (rojo)
        ax.quiver(point[0], point[1], point[2], cam_axis[1][0], cam_axis[1][1], cam_axis[1][2], length=0.5, color='g')  # Eje Y (verde)
        ax.quiver(point[0], point[1], point[2], cam_axis[2][0], cam_axis[2][1], cam_axis[2][2], length=0.5, color='b')  # Eje Z (azul)
    elif counter < 2: # apriltag
        ax.scatter(point[0], point[1], point[2], c='r', marker='o', label='Centro del AprilTag')
        ax.quiver(point[0], point[1], point[2], apriltag_axis[0][0], apriltag_axis[0][1], apriltag_axis[0][2], length=0.5, color='r')  # Eje X (rojo)
        ax.quiver(point[0], point[1], point[2], apriltag_axis[1][0], apriltag_axis[1][1], apriltag_axis[1][2], length=0.5, color='g')  # Eje Y (verde)
        ax.quiver(point[0], point[1], point[2], apriltag_axis[2][0], apriltag_axis[2][1], apriltag_axis[2][2], length=0.5, color='b')  # Eje Z (azul)
    
    else:
        ax.scatter(point[0], point[1], point[2], c='k', marker='o')

    
    counter += 1






# Mostrar leyenda
ax.legend()

# Mostrar el gráfico
plt.show()


