"""
        helper_functions.py

    Este script contiene la agrupación de funciones axiliares.
    Se puede subdividir en tres grandes secciones:
        - Operaciones matematicas (transformaciones)
        - Manejo de imagenes 2D (utilizando opencv)
        - Manejo y representacion en 3D (se puede dividir en 2 segun la libreria utilizada):
            . Respresentacion de puntos concretos (matplotlib)
            . Nubes de puntos (open3d)

"""

# -------------------- PACKAGES ------------------------------------------------------------------------------------------ #

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import open3d as o3d
import os
import cv2
from sklearn.cluster import DBSCAN


# -------------------- FUNCTIONS ----------------------------------------------------------------------------------------- #

def obtain_last_number(directory: str, name: str) -> int:
    filenames = os.listdir(directory)
    numbers = [int(filename.split('_')[1].split('.')[0]) for filename in filenames if filename.startswith(f'{name}_')]
    if not numbers:
        return 0
    return max(numbers)





# -------------------- OPERACIONES --------------------------------------------------------------------------------------- #


# GERERAR matriz de transformacion
def transformation_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

# Rotacion respecto al eje z:
def rotation_matrix_z(theta_degrees: float) -> np.ndarray:
    # Convert degrees to radians
    theta = np.radians(theta_degrees)
    
    # Compute the cosine and sine of the angle
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Create the rotation matrix
    R_z = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    
    return R_z



# POINT reference system transformation  FALTA EN INGLES
def point_tansf(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    # Agrega una coordenada homogénea al punto
    if T.shape == (4,4):
        h_point= np.append(point, 1)
    else:
        h_point = point
    # Multiplica la matriz de transformación por el punto homogéneo
    h_t_point = np.dot(T, h_point)

    if T.shape == (4,4):
        # Normaliza dividiendo por la coordenada homogénea final
        t_point = np.array([h_t_point[0] / h_t_point[3], 
                h_t_point[1] / h_t_point[3],
                h_t_point[2] / h_t_point[3]])
    else:
        t_point = h_t_point
    return t_point

# ROTATION
def rotation_transf(T: np.ndarray) -> np.ndarray:
    r11, r12, r13 = T[0][:3]
    r21, r22, r23 = T[1][:3]
    r31, r32, r33 = T[2][:3]

    rx = math.atan2(r32, r33)
    ry = math.atan2(-r31, math.sqrt(r32**2 + r33**2))
    rz = math.atan2(r21, r11)

    return np.array([rx, ry, rz])
    
# POSE reference system transformation
def pose_transf(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    new_point = point_tansf(T, point)
    new_rot = rotation_transf(T)
    pose = np.hstack([new_point, new_rot])
    # print('Pose: ', pose)
    return pose



# POINTS disctance
def points_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    # 1. forma -> trigonometria
    # point = point2 - point1
    # point *= point
    # distance = math.sqrt(point[0]+point[1]+ point[2])
    # 2. Calcular la distancia euclidiana
    distance = np.linalg.norm(point2 - point1)
    return distance

# ANGLE between vectors
def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
     # Calcular el producto punto
    dot_product = np.dot(v1, v2)
    
    # Calcular las magnitudes de los vectores
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    # Calcular el coseno del ángulo
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    
    # Asegurarse de que el valor esté dentro del rango [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calcular el ángulo en radianes
    theta_radians = np.arccos(cos_theta)
    
    # Convertir el ángulo a grados
    theta_degrees = np.degrees(theta_radians)
    
    return theta_degrees

# CENTROIDE de una serie de puntos
def calculate_centroid(points: np.ndarray) -> np.ndarray:
    """
    Calculates the centroid of a list of 2D points.

    :param points: List of tuples (x, y) representing the 2D points.
    :return: A tuple (C_x, C_y) representing the centroid.
    """
    if len(points) == 0:
        raise ValueError("The list of points cannot be empty")
    
    sum_x = 0
    sum_y = 0
    num_points = len(points)

    for (x, y) in points:
        # print(x,y)
        sum_x += x
        sum_y += y

    C_x = sum_x / num_points
    C_y = sum_y / num_points

    return np.array([C_x, C_y])


# METODO de Procrustes. ajuste de puntos:
def procrustes_method(P, P_prime, verbose: bool = False) -> np.ndarray:
    """
    Calcula la matriz de transformación (rotación y traslación) entre dos conjuntos de puntos 3D.

    Parámetros:
    P (np.array): Matriz de puntos en el sistema de coordenadas original (3 x N).
    P_prime (np.array): Matriz de puntos en el nuevo sistema de coordenadas (3 x N).

    Retorna:
    np.array: Matriz de transformación 4x4.
    """

    # Calcular centroides
    C_P = P.mean(axis=1, keepdims=True)
    C_P_prime = P_prime.mean(axis=1, keepdims=True)

    # Centrar los puntos alrededor del origen
    P_cent = P - C_P
    P_prime_cent = P_prime - C_P_prime

    # Calcular la matriz de covarianza
    H = np.dot(P_cent, P_prime_cent.T)

    # Descomposición en valores singulares (SVD)
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Calcular la matriz de rotación
    R = np.dot(V, U.T)

    # Asegurarse de que la rotación es válida
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = np.dot(V, U.T)

    # Calcular el vector de traslación
    t = C_P_prime - np.dot(R, C_P)

    # Crear la matriz de transformación completa
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    
    if verbose:
        print("Matriz de transformación:")
        print(T)

    return T



# -------------------- OPENCV -------------------------------------------------------------------------------------------- #

# recortar imagen por tamaño
def crop_frame(frame: np.ndarray, crop_size) -> np.ndarray:
    # 1. verificar tamaño de la imagen
    height, width = frame.shape[:2]
    if height < crop_size or width < crop_size:
        return
    # 2. Calcular las coordenadas para el recorte centrado
    start_x = width // 2 - crop_size // 2
    start_y = height // 2 - crop_size // 2
    end_x = start_x + crop_size
    end_y = start_y + crop_size

    # 3. Realizar el recorte
    cropped_frame = frame[start_y:end_y, start_x:end_x]
    return cropped_frame

# recortar imagen
def crop_frame_2(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    new_frame = frame[corners[0][1]:corners[1][1], corners[0][0]:corners[1][0]]
    return new_frame

# En escala de grises la imagen ----------- DEPRECATED
def process_frame_wb(frame: np.ndarray):
    # Verificar el número de canales en la imagen
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        # La imagen es en color (BGR), convertir a escala de grises
        print('frame conertido a escala de grises')
        frame_wb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif len(frame.shape) == 2:
        # La imagen ya está en escala de grises
        print('imagen ya en escala de grises')
        frame_wb = frame
    else:
        raise ValueError("Formato de imagen no soportado")

    return frame_wb

# agrupar puntos
def point_agroup(points: np.ndarray) -> list[np.ndarray]:
    # Ahora, vamos a aplicar DBSCAN para agrupar las esquinas
    epsilon = 10  # Radio de la vecindad
    min_samples = 5  # Número mínimo de puntos para formar un cluster
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    dbscan.fit(points)
    labels = dbscan.labels_
    # Encontrar los índices de los puntos que pertenecen a cada grupo
    unique_labels = np.unique(labels)
    # print('unique labels: ',unique_labels)

    grouped_points = [points[labels == label] for label in unique_labels if label != -1]  # Excluir el ruido (-1)
    # print('grouped points: ', grouped_points)

    # Calcular los centroides de cada grupo
    group_centroids = [np.mean(group, axis=0) for group in grouped_points]
    # invertimos coordenadas (y,x) -> (x,y)
    centroids = [centroid[::-1] for centroid in group_centroids]
    # print('Centroides: ', centroids)
    return centroids

# deteccion de corners por el metodo de harris
def detect_corners_harris(frame: np.ndarray, block_size: float = 5, ksize: float = 5, k: float = 0.01, paint_frame: bool = False) -> np.ndarray:
    '''
    block_size = 5  # Tamaño del vecindario
    ksize = 5   # Tamaño del kernel de Sobel para derivadas
    k = 0.01  # Parámetro libre del detector de Harris
'''
    # 1. Aplicar el detector de esquinas de Harris; obtenemos matriz de respuesta de esquina
    dst = cv2.cornerHarris(frame, block_size,ksize, k)
    # 2. Dilatar el resultado para marcar mejor las esquinas (opcional)
    dst = cv2.dilate(dst,None)
    # 3. establecemos umbral y filtramos
    umbral = 0.01*dst.max()
    # posicion de los pixel corners hay muchos. hay que agruparlos en zonas
    corners_position = np.argwhere(dst > umbral)
    corners = point_agroup(corners_position)

    # Dibujar círculos en las esquinas
    if paint_frame:
        for corner in corners:
            corner = corner.astype(int)
            cv2.circle(frame, (corner[0],corner[1]), 3, 0, -1)

    return corners


# -------------------- MATPLOT ------------------------------------------------------------------------------------------- #

# INIT matplot3d
def init_mat3d(ref_point1: np.ndarray = np.array([1,1,1]), ref_point2: np.ndarray = np.array([-1,-1,-1])) -> tuple[Figure, Axes]:
    # Create figure and 3d axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Ref points
    ax.scatter(ref_point1[0], ref_point1[1], ref_point1[2], c='w', marker='o')
    ax.scatter(ref_point2[0], ref_point2[1], ref_point2[2], c='w', marker='o')

    return fig, ax

# SHOW plot
def show_mat3d(fig: Figure, ax: Axes, name: str = '', legend: bool = True) -> None:
    fig.suptitle(name)
    if legend:
        ax.legend()
    plt.show()
    return

# POINT
def add_point(ax: Axes, point: np.ndarray, name: str, c: str = 'k') -> None:
    ax.scatter(point[0], point[1], point[2], c=c, marker='o', label=name)
    return

# LINE
def add_line(ax: Axes, point1: np.ndarray, point2: np.ndarray, c: str = 'y') -> None:
    ax.plot([point1[0],point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color=c)
    return 

# AXES
def add_axes(ax: Axes, point: np.ndarray, axes: np.ndarray, length: float = 1) -> None:
    ax.quiver(point[0], point[1], point[2], axes[0][0], axes[0][1], axes[0][2], length=length, color='r')  # X axis (red)
    ax.quiver(point[0], point[1], point[2], axes[1][0], axes[1][1], axes[1][2], length=length, color='g')  # Y axis (green)
    ax.quiver(point[0], point[1], point[2], axes[2][0], axes[2][1], axes[2][2], length=length, color='b')  # Z axis (blue)
    return 

# POINT + AXES
def add_point_with_axes(ax: Axes, point: np.ndarray, axes: np.ndarray, name: str, c: str = 'k') -> None:
    ax.scatter(point[0], point[1], point[2], c=c, marker='o', label=name)
    add_axes(ax, point, axes)
    return


# -------------------- OPEN3D -------------------------------------------------------------------------------------------- #

# CREATE Pointcloud
def create_pointcloud(points: np.ndarray) -> o3d.geometry.PointCloud: 
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    return pointcloud

# CREATE Line
def create_line(point1: np.ndarray, point2: np.ndarray, color: np.ndarray = np.array([0, 0, 0])) -> o3d.geometry.LineSet:
    points = np.array([point1, point2])
    lines = np.array([[0, 1]])
    colors = np.array([color])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

# CREATE CUBE
def create_cube(point: np.ndarray, size: np.ndarray = np.array([50 ,50, 50]), color: np.ndarray = np.array([0, 1, 0])) -> o3d.geometry.TriangleMesh:
    # Define bounding box
    min_bound = np.array([0, 0, 0])  # min limits
    max_bound = size # max limits

    # Create mesh to visualize
    bbox_mesh = o3d.geometry.TriangleMesh.create_box(width=max_bound[0]-min_bound[0],
                                                    height=max_bound[1]-min_bound[1],
                                                    depth=max_bound[2]-min_bound[2])
    bbox_mesh.compute_vertex_normals()
    
    # Define color
    bbox_mesh.paint_uniform_color(color)
    
    # Traslate to new pos
    bbox_center= ((max_bound-min_bound)/2)
    new_pos = point - bbox_center
    bbox_mesh.translate(new_pos)

    return bbox_mesh

# CREATE axes
def create_axes(point: np.ndarray = np.array([0,0,0]), size: float = 100) -> o3d.geometry.TriangleMesh:
    axes_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=point)
    return axes_mesh

# CREATE Axes with LineSet
def create_axes_with_lineSet(point: np.ndarray = np.array([0,0,0]), size: float = 100) -> o3d.geometry.LineSet:

    # Create the endpoints of the axes
    points = np.array([point,  # origin
                    point + [size, 0, 0],  # X
                    point + [0, size, 0],  # Y
                    point + [0, 0, size]]) # Z

    # Create Lines, conections between points (origin, x) (origin, y) (origin, z)
    lines = [[0, 1], [0, 2], [0, 3]] 

    # Colors
    colors = np.array([[1, 0, 0],  # red to X
                        [0, 1, 0],  # green to Y
                        [0, 0, 1]]) # blue to Z
    
    # LineSet Object
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

# CREATE axis with points
def create_2D_axes_with_points(center: np.ndarray, x: np.ndarray, y: np.ndarray) -> o3d.geometry.LineSet:
    # Create the endpoints of the axes
        points = np.array([center,  #  Origin
                            x,      # X
                            y])     # Y

        # Create Lines, conections between points (origin, x) (origin, y)
        lines = [[0, 1], [0, 2]] 

        # Colors
        colors = np.array([[1, 0, 0],   # red to X
                            [0, 1, 0]]) # green to Y
        
        # LineSet object
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
    
        return line_set


# EXPORT
def export_pointcloud(filename: str, pointcloud) -> None:
    # Save in ply format
    o3d.io.write_point_cloud(filename, pointcloud)
    return

# IMPORT
def import_pointcloud(filename: str) -> o3d.geometry.PointCloud:
    # Import .ply file
    pointcloud = o3d.io.read_point_cloud(str(filename))
    # points
    points = np.asarray(pointcloud.points)

    # We invert points with respect to the x-axis because it's mirrored
    # points[:,0] *= -1

    pointcloud.points = o3d.utility.Vector3dVector(points)
    return pointcloud

# INVERT pointcloud
def invert_pointcloud(pointcloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    # points
    points = np.asarray(pointcloud.points)
    # We invert points with respect to the x-axis because it's mirrored
    # points[:,0] *= -1
    points[:,1] *= -1
    pointcloud.points = o3d.utility.Vector3dVector(points)
    return pointcloud

# PIXEL to 3D point
def pixel_to_point3d(pointcloud: o3d.geometry.PointCloud, resolution: np.ndarray, pixel: np.ndarray) -> np.ndarray:
    points = np.asarray(pointcloud.points)
    # calculate pixel index
    index = (resolution[0]*pixel[1] + pixel[0])-1
    # 3d point
    point3d =points[index]
    return point3d

# CROP by thresholds
def crop_pointcloud_by_thresholds(pointcloud: o3d.geometry.PointCloud, x_thresholds: np.ndarray | None = None, y_thresholds: np.ndarray | None = None, z_thresholds: np.ndarray| None = None) -> o3d.geometry.PointCloud:
    # Acquire pointcloud points and colors to work
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)

    # Filter the points based on the Y coordinate
    if x_thresholds:
        points_filt = points[(points[:, 0] >= x_thresholds[0]) & (points[:, 0] <= x_thresholds[1])]
        colors_filt = colors[(points[:, 0] >= x_thresholds[0]) & (points[:, 0] <= x_thresholds[1])]
        points = points_filt
        colors = colors_filt   

    # Filter the points based on the X coordinate
    if y_thresholds:
        points_filt = points[(points[:, 1] >= y_thresholds[0]) & (points[:, 1] <= y_thresholds[1])]
        colors_filt = colors[(points[:, 1] >= y_thresholds[0]) & (points[:, 1] <= y_thresholds[1])]
        points = points_filt
        colors = colors_filt  
    
    # Filter the points based on the Z coordinate
    if z_thresholds:
        points_filt = points[(points[:, 2] >= z_thresholds[0]) & (points[:, 2] <= z_thresholds[1])]
        colors_filt = colors[(points[:, 2] >= z_thresholds[0]) & (points[:, 2] <= z_thresholds[1])]
        points = points_filt
        colors = colors_filt


    # New pointcloud with only filtered points
    pointcloud_filt = o3d.geometry.PointCloud()
    pointcloud_filt.points = o3d.utility.Vector3dVector(points)
    pointcloud_filt.colors = o3d.utility.Vector3dVector(colors)

    return pointcloud_filt

# CROP by pixels
def crop_pointcloud_by_pixels(pointcloud: o3d.geometry.PointCloud, resolution: np.ndarray, pixel_min: np.ndarray, pixel_max: np.ndarray) -> o3d.geometry.PointCloud:
    filt = []
    for element in range(resolution[1]): # rows -> Y
        if element > pixel_min[1] and element < pixel_max[1]:
            filt.extend(list(range(((resolution[0]*element)+pixel_min[0]), ((resolution[0]*element)+pixel_max[0]))))

    pointcloud = pointcloud.select_by_index(filt)

    return pointcloud
    
# INIT
def init_o3d_viualizer(geometries: list) -> o3d.visualization.Visualizer:
    # Visualizar la nube de puntos
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    return visualizer
        
def update_o3d_visualizer(visualizer, geometries):
    # Actualizar la visualización
    for geometry in geometries:
        visualizer.update_geometry(geometry)
    visualizer.poll_events()
    visualizer.update_renderer()

# Visualizacion
def o3d_visualization(geometries: list, edit: bool = False):
    if edit:
        o3d.visualization.draw_geometries_with_editing(geometries)
    else:
        o3d.visualization.draw_geometries(geometries)


# -------------------- TRAINNING ----------------------------------------------------------------------------------------- #

def main_recorte():
    # recorte
    filename = fr'app\assets\pointclouds\crop_testing.ply'
    pointcloud = import_pointcloud(filename)
    resolution = (1280, 720)
    pixel_min = (580, 380)
    pixel_max = (685,550)
    pointcloud = crop_pointcloud_by_pixels(pointcloud, resolution, pixel_min, pixel_max)
    pointcloud = crop_pointcloud_by_thresholds(pointcloud, z_thresholds=[420,490])

    axis = create_axes(normalized=False, size=100)
    cube = create_cube(point=[75,0,75], size=[50,50,50], color=[1,1,0])
    line = create_line(point1=[0,0,0], point2=[75,0,75])
    o3d_visualization([pointcloud, axis, cube, line])



def main_transform():
    name = 'april_square_2_4'
    filename = fr'app\assets\pointclouds\{name}.ply'
    pointcloud = import_pointcloud(filename)

    axis = create_axes(normalized=False, size=100)
    cube = create_cube(point=[75,0,75], size=[50,50,50], color=[1,1,0])
    line = create_line(point1=[0,0,0], point2=[75,0,75])
    o3d_visualization([pointcloud, axis, cube, line])


