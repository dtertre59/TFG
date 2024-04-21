import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Supongamos que pose_matrix es tu matriz de transformación
# pose_matrix = np.array([[1, 0, 0, 0],
#                         [0, 1, 0, 0],
#                         [0, 0, 1 ,1],
#                         [0, 0, 0, 1]])

# Supongamos que pose_matrix es tu matriz de transformación
pose_matrix = np.array([[ 0.98949365, -0.02275008,  0.14277518, -0.04343225],
                        [-0.11622167,  0.46221815,  0.87911712,  0.04289319],
                        [-0.08599327, -0.88647438,  0.45471787,  1.52128815],
                        [ 0,         0,          0,          1        ]])

# # Ángulos de rotación en radianes
# theta_x = np.pi / 2  # Por ejemplo, 45 grados en el eje X
# theta_y = np.pi / 2  # Por ejemplo, 30 grados en el eje Y
# theta_z = np.pi / 2  # Por ejemplo, 60 grados en el eje Z

# # Matrices de rotación para cada eje
# R_x = np.array([[1, 0, 0],
#                 [0, np.cos(theta_x), -np.sin(theta_x)],
#                 [0, np.sin(theta_x), np.cos(theta_x)]])

# R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
#                 [0, 1, 0],
#                 [-np.sin(theta_y), 0, np.cos(theta_y)]])

# R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
#                 [np.sin(theta_z), np.cos(theta_z), 0],
#                 [0, 0, 1]])

# # Combinar las matrices de rotación en una sola
# R = np.dot(R_z, np.dot(R_y, R_x))

# # Vector de traslación (por ejemplo, traslación en el eje X, Y, Z)
# t = np.array([0, 0, 1])

# # Construir matriz de transformación homogénea
# T = np.concatenate([np.concatenate([R, t.reshape(-1, 1)], axis=1), [[0, 0, 0, 1]]], axis=0)

# pose_R = R
# pose_t = t
# pose_matrix = T



# reinvertimos orientacion eje z
# Matriz de reflexión en el plano XY para invertir la dirección del eje Z
M_xy = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])


# Aplicar la reflexión en el plano XY a la matriz de rotación alrededor del eje Z
pose_matrix = np.dot(M_xy, pose_matrix)


# Extraer la parte de rotación y traslación de la matriz de transformación
pose_R = pose_matrix[:3, :3]
pose_t = pose_matrix[:3, 3]

# Invertir la matriz de transformación
inverse_pose_matrix = np.linalg.inv(pose_matrix)

# Extraer la parte de rotación y traslación de la matriz de transformación
inverse_pose_R = inverse_pose_matrix[:3, :3]
inverse_pose_t = inverse_pose_matrix[:3, 3]


# Definir ejes base
axis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Ejes unitarios
# print('Ejes base: ',axis)

print('M TRANSFORMACION: ', pose_matrix)

class Transf():
    def __init__(self,t, ref, comp, n_axis):
        self.t = t
        self.ref = ref
        self.comp = comp
        self.n_axis = n_axis



# ----- CALCULOPS RESPECTO A LA CAMARA -----

camera_position = np.array([0,0,0])
print("Posición de la camara con respecto al camara:", camera_position)

# La parte de traslación de la matriz de transformación representa la posición del AprilTag con respecto a la cámara
apriltag_position = pose_matrix[:3, 3]
print("Posición del AprilTag con respecto a la cámara:", apriltag_position)

# Matriz de rotación para una rotación de 180 grados alrededor del eje X
# pose_R = np.array([[-1, 0, 0],
#                     [0, -1, 0],
#                     [0, 0, -1]])
# Aplicar transformación de rotación a los ejes de la cámara
n_axis = np.dot(pose_R, axis.T).T
# print('Ejes nuevos: ', n_axis.T)

data_cam = Transf(pose_matrix, camera_position, apriltag_position, n_axis)

# ----- CALCULOS RESPECTO AL APRILTAG -----

apriltag_position = np.array([0,0,0])
print("Posición del AprilTag con respecto al apriltag:", apriltag_position)

# La última columna de la matriz inversa (las primeras tres son la rotación y la última es la traslación) te da la posición de la cámara con respecto al AprilTag
camera_position = inverse_pose_matrix[:3, 3]
print("Posición de la cámara con respecto al AprilTag:", camera_position)

## Aplicar transformación de rotación a los ejes de la cámara
n_axis = np.dot(inverse_pose_R, axis.T).T

data_april = Transf(inverse_pose_matrix, apriltag_position, camera_position, n_axis)
#------------------------------------------

# Crear la figura y los ejes en 3D
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

# Crear la figura y los ejes en 3D
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')


# PAINT 3D
def paint_3d(ax , ref_p, comp_p, n_axis):
    points = [ref_p, comp_p, (1.5,1.5,1.5), (-1.5,-1.5,-1.5)]
    # Trazar un mini eje de coordenadas en cada punto
    counter = 0
    for point in points:
        if counter < 1: # ref axis
            ax.scatter(point[0], point[1], point[2], c='b', marker='o', label='Ref')
            ax.quiver(point[0], point[1], point[2], axis[0][0], axis[0][1], axis[0][2], length=0.5, color='r')  # Eje X (rojo)
            ax.quiver(point[0], point[1], point[2], axis[1][0], axis[1][1], axis[1][2], length=0.5, color='g')  # Eje Y (verde)
            ax.quiver(point[0], point[1], point[2], axis[2][0], axis[2][1], axis[2][2], length=0.5, color='b')  # Eje Z (azul)
        elif counter < 2: # apriltag
            ax.scatter(point[0], point[1], point[2], c='r', marker='o', label='Comp')
            ax.quiver(point[0], point[1], point[2], n_axis[0][0], n_axis[0][1], n_axis[0][2], length=0.5, color='r')  # Eje X (rojo)
            ax.quiver(point[0], point[1], point[2], n_axis[1][0], n_axis[1][1], n_axis[1][2], length=0.5, color='g')  # Eje Y (verde)
            ax.quiver(point[0], point[1], point[2], n_axis[2][0], n_axis[2][1], n_axis[2][2], length=0.5, color='b')  # Eje Z (azul)
        
        else: # Others
            ax.scatter(point[0], point[1], point[2], c='k', marker='o')
   
        counter += 1

    # union del ref con el otro
    length = 1
    ax.quiver(points[0][0], points[0][1], points[0][2], points[1][0], points[1][1], points[1][2], length=length, color='k')
    # Mostrar leyenda
    ax.legend()
    return



paint_3d(ax1, data_cam.ref, data_cam.comp, data_cam.n_axis)
paint_3d(ax2, data_april.ref, data_april.comp, data_april.n_axis)

# Cambiar el título de la figura
fig1.suptitle('Sistema de ref: camara')

# Cambiar el título de la figura
fig2.suptitle('Sistema ref: apriltag')

plt.show()


