import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D





def init_3d_rep():
    # Crear la figura y los ejes en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # puntos de referencia
    ax.scatter(1, 1, 1, c='k', marker='o')
    ax.scatter(-1, -1, 0, c='k', marker='o')
    # camera = ref point
    # Crear una matriz identidad 4x4
    t_i = np.eye(4)
    #t_i[1,1] = -1

    return fig, ax, t_i

def end_3d_rep(fig, ax, name: str = ''):
    # Cambiar el título de la figura
    fig.suptitle(name)
    ax.legend()

    plt.show()

def print_3d_rep(ax, t, scale: float = 1, c: str = 'k', pointname: str = None, ax_ref: bool = False):
    axis = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])  # Ejes unitarios
    if ax_ref == True:
        axis = np.array([[scale, 0, 0], [0, -scale, 0], [0, 0, -scale]])  # Ejes unitarios
    point = t[:3, 3]
    rot = t[:3, :3]
    rot_x = t[:3, 0]
    rot_y = t[:3, 1]
    rot_z = t[:3, 2]
    # print('Point: ',point)
    # transformacion de ejes
    axis = np.dot(rot, axis.T).T
    # point
    ax.scatter(point[0], point[1], point[2], c=c, marker='o', label=pointname)
    # axis
    # ax.quiver(point[0], point[1], point[2], rot_x[0], rot_x[1], rot_x[2], length=0.5, color='r')  # Eje X (rojo)
    # ax.quiver(point[0], point[1], point[2],  rot_y[0], rot_y[1], rot_y[2], length=0.5, color='g')  # Eje Y (verde)
    # ax.quiver(point[0], point[1], point[2],  rot_z[0], rot_z[1], rot_z[2], length=0.5, color='b')  # Eje Z (azul)
    ax.quiver(point[0], point[1], point[2], axis[0][0], axis[0][1], axis[0][2], length=0.5, color='r')  # Eje X (rojo)
    ax.quiver(point[0], point[1], point[2], axis[1][0], axis[1][1], axis[1][2], length=0.5, color='g')  # Eje Y (verde)
    ax.quiver(point[0], point[1], point[2], axis[2][0], axis[2][1], axis[2][2], length=0.5, color='b')  # Eje Z (azul)

    # unir con el ref -> Dibujar la línea que conecta los puntos
    ax.plot([0,point[0]], [0, point[1]], [0, point[2]], color='y')


# AXIS
def print_axis(ax, point, axis):
    # axis
    ax.quiver(point[0], point[1], point[2], axis[0][0], axis[0][1], axis[0][2], length=0.5, color='r')  # Eje X (rojo)
    ax.quiver(point[0], point[1], point[2], axis[1][0], axis[1][1], axis[1][2], length=0.5, color='g')  # Eje Y (verde)
    ax.quiver(point[0], point[1], point[2], axis[2][0], axis[2][1], axis[2][2], length=0.5, color='b')  # Eje Z (azul)

# POINT
def print_point(ax, point, name: str, c: str = 'k'):
    ax.scatter(point[0], point[1], point[2], c=c, marker='o', label=name)

# POINT + AXIS
def print_point_with_axis(ax, point, name, c: str = 'k', scale: float = 1):
    axis = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])  # Ejes unitarios

    ax.scatter(point[0], point[1], point[2], c=c, marker='o', label=name)
    print_axis(ax, point, axis)

# LINE
def print_line(ax, point1, point2, c: str = 'y'):
    ax.plot([point1[0],point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color=c)

# DISTANCE MATRIX
def distance(t1, t2):
    
    o = np.array([0,0,0,1])

    t1_i = np.linalg.inv(t1)
    t2_i = np.linalg.inv(t2)

    crc =o
    a1ra1 = o
    a2ra2 = o


    a1rc = np.dot(t1, o)
    a2rc = np.dot(t2, o)
    
    cra1 = np.dot(t1_i, o)
    cra2 = np.dot(t2_i, o)

    a2ra1 = np.dot(t1_i, a2rc)
    a1ra2 = np.dot(t2_i, a1rc)
    

    # return crc, a1rc, a2rc
    # return cra1, a1ra1, a2ra1
    return cra2, a1ra2, a2ra2
    
    
    

# POINTS DISTANCE
def points_distance(point1: np.array, point2: np.array) -> float:
    # Calcular la distancia euclidiana
    distance = np.linalg.norm(point2 - point1)
    return distance





# ----- PRUEBAS -----



# # apriltag
# pose_matrix = np.array([[ 0.98949412, -0.02267868,  0.1427833, -0.04347489],
#                         [-0.11625551,  0.46227919,  0.87908055,  0.04290879],
#                         [-0.08594214, -0.88644437,  0.45478602,  1.5226692 ],
#                         [ 0.,          0.,          0.,          1.        ]])

# fig, ax, t_camera = init_3d_rep()
# print_3d_rep(ax, t_camera, c='c')
# print_3d_rep(ax, pose_matrix, 'r')

# end_3d_rep('Sistema de Referencia: Camara')

