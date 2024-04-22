import numpy as np
import math
import matplotlib.pyplot as plt


# POINT reference system transformation 
def point_tansf(T: np.ndarray, point: np.ndarray) -> np.array:
    # Agrega una coordenada homogénea al punto
    h_point= np.append(point, 1)
    # Multiplica la matriz de transformación por el punto homogéneo
    h_t_point = np.dot(T, h_point)

    # Normaliza dividiendo por la coordenada homogénea final
    t_point = np.array([h_t_point[0] / h_t_point[3], 
               h_t_point[1] / h_t_point[3],
               h_t_point[2] / h_t_point[3]])
    return t_point


# POINTS disctance
def points_distance(point1: np.array, point2: np.array) -> float:
    # 1. forma -> trigonometria
    # point = point2 - point1
    # point *= point
    # distance = math.sqrt(point[0]+point[1]+ point[2])
    # 2. Calcular la distancia euclidiana
    distance = np.linalg.norm(point2 - point1)
    return distance



# ---------- VISUALIZATION ------------------------------

def init_3d_rep():
    # Crear la figura y los ejes en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # puntos de referencia
    ax.scatter(1, 1, 1, c='w', marker='o')
    ax.scatter(-1, -1, 1, c='w', marker='o')
    # camera = ref point
    # Crear una matriz identidad 4x4
    t_i = np.eye(4)
    #t_i[1,1] = -1

    return fig, ax, t_i

# END 3d representatios
def show_3d_rep(fig, ax, name: str = ''):
    # Cambiar el título de la figura
    fig.suptitle(name)
    ax.legend()

    plt.show()

# PRINT 3D representation
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


#-------------------PRUEBAS ----------------------------


t_april1_to_camera = np.array([[-0.95688569, -0.23952592, -0.16430795,  0.13719614],
                     [ 0.05192871, -0.69762885,  0.71457498,  0.03184161],
                     [-0.28578519,  0.67523428,  0.67998933,  0.70670315],
                     [ 0.,          0.,          0.,          1.        ]])

t_camera_to_april1 = np.linalg.inv(t_april1_to_camera)

t_april2_to_camera = np.array([[ 0.98549024,  0.15646889, -0.06577595,  0.11766756],
                            [-0.08862948,  0.80488383,  0.58677665, -0.0666505 ],
                            [ 0.14475429, -0.57243298,  0.80707291,  0.741747  ],
                            [ 0.,          0.,          0.,          1.        ]])

t_april1_to_robot = np.array([[1, 0, 0, -0.1075],
                                [0, 1, 0, -0.3839],
                                [0, 0, 1, 0.0108],
                                [0, 0, 0, 1]])

t_robot_to_april1 = np.linalg.inv(t_april1_to_robot)



punto_pieza_respecto_april = np.array([0.3, 0.3, 0.3])

# t_point = point_tansf(punto_pieza_respecto_april, t_apriltag_to_robot)


# PUNTO OBJETIVO -> ppieza
# SISTEMA DE REFERENCIA INICIAL -> EL DE LA CAMARA
# SISTEMA DE REF IMTERMEDIO -> APRILTAG 1
# SISTEMA DE REF FINAL -> BASE DEL ROBOT

pcrc = np.array([0, 0, 0])
parc = t_april1_to_camera[:3, 3]
ppiezarc = t_april2_to_camera[:3, 3]

para = pcrc
pcra = point_tansf(t_camera_to_april1, pcrc)
ppiezara = point_tansf(t_camera_to_april1, ppiezarc)
prra = point_tansf(t_robot_to_april1, [0,0,0])

parr = point_tansf(t_april1_to_robot, para)
ppiezarr = point_tansf(t_april1_to_robot, ppiezara)
pcrr = point_tansf(t_april1_to_robot, pcra)

dr = points_distance(ppiezarr, pcrr)
da = points_distance(ppiezara, pcra)
dc = points_distance(ppiezarc, pcrc)

print(dr)
print(da)
print(dc)


# VIS

fig, ax, t_camera = init_3d_rep()

print_point_with_axis(ax, [0,0,0], 'robot')
print_point(ax, ppiezarr, 'pieza', 'r')
print_point(ax, pcrr, 'camara', 'b')

show_3d_rep(fig, ax, 'Sistema de Referencia: Robot')


fig2, ax2, t_camera = init_3d_rep()

print_point(ax2, prra, 'robot')
print_point(ax2, ppiezara, 'pieza', 'c')
print_point_with_axis(ax2, para, 'april', 'r')
print_point(ax2, pcra, 'camara', 'b')

show_3d_rep(fig2, ax2, 'Sistema de Referencia: April')

