import numpy as np
import math
import matplotlib.pyplot as plt
import open3d as o3d

# ---------- TRANSFORMACIONES

# POINT reference system transformation 
def point_tansf(T: np.ndarray, point: np.ndarray) -> np.ndarray:
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
def points_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    # 1. forma -> trigonometria
    # point = point2 - point1
    # point *= point
    # distance = math.sqrt(point[0]+point[1]+ point[2])
    # 2. Calcular la distancia euclidiana
    distance = np.linalg.norm(point2 - point1)
    return distance



# ---------- VISUALIZATION ------------------------------

# ----- MATPLOT -----

# INIT
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

# SHOW 3d representatios
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
def print_point_with_axis(ax, point, axis, name, c: str = 'k'):
    ax.scatter(point[0], point[1], point[2], c=c, marker='o', label=name)
    print_axis(ax, point, axis)

# LINE
def print_line(ax, point1, point2, c: str = 'y'):
    ax.plot([point1[0],point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color=c)


# ----- OPEN3D -----


# CREATE Pointcloud
def create_pointcloud(points) -> o3d.geometry.PointCloud: 
    # Crear un objeto PointCloud
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    return pointcloud

# CREATE LINE
def create_line(point1: np.ndarray, point2: np.ndarray, color: np.ndarray = np.array([0, 0, 0])):
    points = np.array([point1, point2])
    lines = np.array([[0, 1]])
    colors = np.array([color])
    # Crea el objeto LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

# CREATE CUBE
def create_cube(point: np.ndarray, size: float = 50, color: np.ndarray = np.array([0, 1, 0])):
    # Definir una caja delimitadora (en este ejemplo, se define manualmente)
    min_bound = np.array([0, 0, 0])  # Límites mínimos de la caja
    max_bound = np.array([size, size, size]) # Límites máximos de la caja

    # Crear una malla para visualizar la caja delimitadora
    bbox_mesh = o3d.geometry.TriangleMesh.create_box(width=max_bound[0]-min_bound[0],
                                                    height=max_bound[1]-min_bound[1],
                                                    depth=max_bound[2]-min_bound[2])
    bbox_mesh.compute_vertex_normals()
    # Cambiar el color de la caja a rojo
    bbox_mesh.paint_uniform_color(color)  # Color rojo
    
    # Mover la caja delimitadora a una nueva posición
    center= ((max_bound-min_bound)/2)
    new_pos = point - center
    bbox_mesh.translate(new_pos)

    return bbox_mesh

# CREATE axis
def create_axis(normalized: bool = True, size: float = 100) -> o3d.geometry.LineSet:
    if normalized:
        # Crear el eje de coordenadas
        axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
        return axis_mesh
    else:
        # Crear los puntos de los extremos de los ejes
        points = np.array([[0, 0, 0],  # Origen
                        [size, 0, 0],  # Punto en el eje x
                        [0, size, 0],  # Punto en el eje y
                        [0, 0, size]]) # Punto en el eje z

        # Crear las líneas que representan los ejes
        lines = [[0, 1], [0, 2], [0, 3]]  # Conexiones entre los puntos (origen, x) (origen, y) (origen, z)

        # Definimos un color para la línea (en este caso, rojo)
        colors = np.array([[1, 0, 0],  # Color rojo para el x
                            [0, 1, 0],  # Color verde para el y
                            [0, 0, 1]]) # Color rojo para el z
        
        # Crear el objeto LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
    
        return line_set

# CREATE axis withpoints
def create_axis_with_points(center: np.ndarray, x: np.ndarray, y: np.ndarray) -> o3d.geometry.LineSet:
    # Crear los puntos de los extremos de los ejes
        points = np.array([center,  # Origen
                        x,  # Punto en el eje x
                        y]) # Punto en el eje z

        # Crear las líneas que representan los ejes
        lines = [[0, 1], [0, 2]]  # Conexiones entre los puntos (origen, x) (origen, y) (origen, z)

        # Definimos un color para la línea (en este caso, rojo)
        colors = np.array([[1, 0, 0],  # Color rojo para el x
                            [0, 1, 0]]) # Color verde para el y
        
        # Crear el objeto LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
    
        return line_set


# EXPORT
def export_pointcloud(filename: str, pointcloud):
    # Guardar la nube de puntos en formato PLY
    o3d.io.write_point_cloud(filename, pointcloud)

# IMPORT
def import_pointcloud(filename: str) -> o3d.geometry.PointCloud:
    # Importar el archivo PLY
    pointcloud = o3d.io.read_point_cloud(str(filename))
    # adquirimos puntos
    points = np.asarray(pointcloud.points)
    # invertimos puntos respecto al eje x porque esta en espejo
    points[:,0] *= -1
    #aqui tenemos la nube invertida
    pointcloud.points = o3d.utility.Vector3dVector(points)
    return pointcloud


# PIXEL to 3D point

def pixel_to_point3d(pointcloud: o3d.geometry.PointCloud, resolution, pixel) -> np.ndarray:
    # construimos matriz (no hace falta) 
    points = points = np.asarray(pointcloud.points)
    matrix = points.reshape(-1, resolution[0]*3) # 1280 columnas (x)

    # 1280*720
    number = (resolution[0]*pixel[1] + pixel[0])-1

    point3d =points[number]
    # point3d = points[921600]

    # point3d = matrix[pixel[0], pixel[1]]

    return point3d

# CROP by thresholds
def crop_pointcloud_by_thresholds(pointcloud: o3d.geometry.PointCloud, x_thresholds: np.ndarray | None = None, y_thresholds: np.ndarray | None = None, z_thresholds: np.ndarray| None = None) -> o3d.geometry.PointCloud:
    # Convertir la nube de puntos a un arreglo numpy para un procesamiento más eficiente
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)

    # Filtrar los puntos basados en la coordenada x
    if x_thresholds:
        points_filt = points[(points[:, 0] >= x_thresholds[0]) & (points[:, 0] <= x_thresholds[1])]
        colors_filt = colors[(points[:, 0] >= x_thresholds[0]) & (points[:, 0] <= x_thresholds[1])]
        points = points_filt
        colors = colors_filt   

    # Filtrar los puntos basados en la coordenada y
    if y_thresholds:
        points_filt = points[(points[:, 1] >= y_thresholds[0]) & (points[:, 1] <= y_thresholds[1])]
        colors_filt = colors[(points[:, 1] >= y_thresholds[0]) & (points[:, 1] <= y_thresholds[1])]
        points = points_filt
        colors = colors_filt  
    
    # Filtrar los puntos basados en la coordenada z
    if z_thresholds:
        points_filt = points[(points[:, 2] >= z_thresholds[0]) & (points[:, 2] <= z_thresholds[1])]
        colors_filt = colors[(points[:, 2] >= z_thresholds[0]) & (points[:, 2] <= z_thresholds[1])]
        points = points_filt
        colors = colors_filt


    # Crear una nueva nube de puntos con los puntos filtrados
    pointcloud_filt = o3d.geometry.PointCloud()
    pointcloud_filt.points = o3d.utility.Vector3dVector(points)
    pointcloud_filt.colors = o3d.utility.Vector3dVector(colors)

    return pointcloud_filt

# CROP by pixels
def crop_pointcloud_by_pixels(pointcloud: o3d.geometry.PointCloud, resolution, pixel_min, pixel_max) -> o3d.geometry.PointCloud:
    # construimos matriz (no hace falta) 
    points = points = np.asarray(pointcloud.points)
    # matrix = points.reshape(-1, resolution[0]*3) # 1280 columnas (x)
    
    # filtrado
    filt = []
    for element in range(resolution[1]): # 720 filas (y)
        if element > pixel_min[1] and element < pixel_max[1]:
            filt.extend(list(range(((resolution[0]*element)+pixel_min[0]), ((resolution[0]*element)+pixel_max[0]))))

    pointcloud = pointcloud.select_by_index(filt)

    return pointcloud
    


# ----- VISUALIZER


def init_visualizer(geometries: list) -> o3d.visualization.Visualizer:
    # Visualizar la nube de puntos
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    return visualizer
        
def update_visualizer(visualizer, geometries):
    # Actualizar la visualización
    for geometry in geometries:
        visualizer.update_geometry(geometry)
    visualizer.poll_events()
    visualizer.update_renderer()

# Visualizacion
def visualization(geometries: list, edit: bool = False):
    # Visualizar el objeto PointCloud
    if edit:
        o3d.visualization.draw_geometries_with_editing(geometries)
    else:
        o3d.visualization.draw_geometries(geometries)




#-------------------PRUEBAS ----------------------------


def main_recorte():
    # recorte
    filename = fr'app\assets\pointclouds\crop_testing.ply'
    pointcloud = import_pointcloud(filename)
    resolution = (1280, 720)
    pixel_min = (580, 380)
    pixel_max = (685,550)
    pointcloud = crop_pointcloud_by_pixels(pointcloud, resolution, pixel_min, pixel_max)
    pointcloud = crop_pointcloud_by_thresholds(pointcloud, z_thresholds=[420,490])

    axis = create_axis(normalized=False, size=100)
    cube = create_cube(point=[75,0,75], size=50, color=[1,1,0])
    line = create_line(point1=[0,0,0], point2=[75,0,75])
    visualization([pointcloud, axis, cube, line])



def main_transform():
    name = 'april_square_2_4'
    filename = fr'app\assets\pointclouds\{name}.ply'
    pointcloud = import_pointcloud(filename)

    axis = create_axis(normalized=False, size=100)
    cube = create_cube(point=[75,0,75], size=50, color=[1,1,0])
    line = create_line(point1=[0,0,0], point2=[75,0,75])
    visualization([pointcloud, axis, cube, line])



# main_recorte()
# main_transform()





# t_april1_to_camera = np.array([[-0.95688569, -0.23952592, -0.16430795,  0.13719614],
#                      [ 0.05192871, -0.69762885,  0.71457498,  0.03184161],
#                      [-0.28578519,  0.67523428,  0.67998933,  0.70670315],
#                      [ 0.,          0.,          0.,          1.        ]])

# t_rot_180_x = np.array([[1, 0, 0, 0],
#                         [0, -1, 0, 0],
#                         [0, 0, -1, 0],
#                         [0, 0, 0, 1]])

# t_april1_to_camera = np.dot(t_april1_to_camera, t_rot_180_x)

# print(t_april1_to_camera)

# t_camera_to_april1 = np.linalg.inv(t_april1_to_camera)

# t_april2_to_camera = np.array([[ 0.98549024,  0.15646889, -0.06577595,  0.11766756],
#                             [-0.08862948,  0.80488383,  0.58677665, -0.0666505 ],
#                             [ 0.14475429, -0.57243298,  0.80707291,  0.741747  ],
#                             [ 0.,          0.,          0.,          1.        ]])

# t_april2_to_camera = np.dot(t_april2_to_camera, t_rot_180_x)

# t_camera_to_april2 = np.linalg.inv(t_april2_to_camera)

# t_april1_to_robot = np.array([[1, 0, 0, -0.1075],
#                                 [0, 1, 0, -0.3839],
#                                 [0, 0, 1, 0.0108],
#                                 [0, 0, 0, 1]])

# t_april1_to_robot = np.array([[1, 0, 0, 1],
#                                 [0, 1, 0, 0.],
#                                 [0, 0, 1, 0.],
#                                 [0, 0, 0, 1]])

# t_robot_to_april1 = np.linalg.inv(t_april1_to_robot)



# punto_pieza_respecto_april = np.array([0.3, 0.3, 0.3])

# t_point = point_tansf(punto_pieza_respecto_april, t_apriltag_to_robot)


# PUNTO OBJETIVO -> ppieza
# SISTEMA DE REFERENCIA INICIAL -> EL DE LA CAMARA
# SISTEMA DE REF IMTERMEDIO -> APRILTAG 1
# SISTEMA DE REF FINAL -> BASE DEL ROBOT

# respecto de camera
# pcrc = np.array([0, 0, 0])
# parc = t_april1_to_camera[:3, 3]
# ppiezarc = t_april2_to_camera[:3, 3]

# # respecto april1
# para = pcrc
# pcra = point_tansf(t_camera_to_april1, pcrc)
# ppiezara = point_tansf(t_camera_to_april1, ppiezarc)
# prra = point_tansf(t_robot_to_april1, [0,0,0])

# # respecto del robot
# parr = point_tansf(t_april1_to_robot, para)
# ppiezarr = point_tansf(t_april1_to_robot, ppiezara)
# pcrr = point_tansf(t_april1_to_robot, pcra)

# # distancias entre pieza y camara
# dr = points_distance(ppiezarr, pcrr)
# da = points_distance(ppiezara, pcra)
# dc = points_distance(ppiezarc, pcrc)

# print(dr)
# print(da)
# print(dc)


# VIS

# fig, ax, t_camera = init_3d_rep()

# print_point_with_axis(ax, [0,0,0], 'robot')
# print_point(ax, ppiezarr, 'pieza', 'r')
# print_point(ax, pcrr, 'camara', 'b')

# show_3d_rep(fig, ax, 'Sistema de Referencia: Robot')


# ----- RESPECTO DEL APRIL -----
# scale = 0.5
# axis = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])  # Ejes unitarios
# fig2, ax2, t_camera = init_3d_rep()

# print_point_with_axis(ax2, para,axis , 'april', 'r')

# axis_r = np.dot(t_april1_to_robot[:3, :3], axis.T).T
# print_point_with_axis(ax2, prra, axis_r, 'robot')

# axis_c = np.dot(t_april1_to_camera[:3,:3], axis.T)
# print_point_with_axis(ax2, pcra, axis_c, 'camara', 'b')

# axis_pieza = np.dot(t_camera_to_april2[:3, :3], axis_c.T).T
# print_point_with_axis(ax2, ppiezara, axis_pieza, 'pieza', 'c')




# show_3d_rep(fig2, ax2, 'Sistema de Referencia: April')


# ----- RESPECTO DE LA CAMARA -----

# scale = 0.5
# axis = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])  # Ejes unitarios
# fig3, ax3, t_ident = init_3d_rep()

# print_point_with_axis(ax3, pcrc,axis, 'camara', 'b')

# axis_pieza = np.dot(t_april2_to_camera[:3, :3], axis.T).T
# print_point_with_axis(ax3, ppiezarc,axis_pieza, 'pieza', 'c')

# axis_a = np.dot(t_april1_to_camera[:3, :3], axis.T).T
# print_point_with_axis(ax3, parc, axis_a, 'april', 'r')

# show_3d_rep(fig3, ax3, 'Sistema de Referencia: CAmara')

