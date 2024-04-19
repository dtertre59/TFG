import open3d as o3d
import numpy as np
import time
from pathlib import Path

# import point cloud
pointcloud = o3d.io.read_point_cloud(str(Path(__file__).resolve().parent / 'clouds' / 'square_lab_1.ply'))

# ----- Crear Eje de coordenadas ----- # 

points = np.array([[0, 0, 0],  # Origen
                        [100, 0, 0],  # Punto en el eje x
                        [0, 100, 0],  # Punto en el eje y
                        [0, 0, 600]]) # Punto en el eje z

# Crear las líneas que representan los ejes
lines = [[0, 1], [0, 2], [0, 3]]  # Conexiones entre los puntos (origen, x) (origen, y) (origen, z)

# Crear el objeto LineSet
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines = o3d.utility.Vector2iVector(lines)

# ----- Crear cubo 3d ----- # 

# Definir una caja delimitadora (en este ejemplo, se define manualmente)
min_bound = [0, 0, 0]  # Límites mínimos de la caja
max_bound = [50.0, 50.0, 50.0]     # Límites máximos de la caja
bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
# Crear una malla para visualizar la caja delimitadora
bbox_mesh = o3d.geometry.TriangleMesh.create_box(width=max_bound[0]-min_bound[0],
                                                  height=max_bound[1]-min_bound[1],
                                                  depth=max_bound[2]-min_bound[2])
bbox_mesh.compute_vertex_normals()
# Cambiar el color de la caja a rojo
bbox_mesh.paint_uniform_color([1, 0, 0])  # Color rojo
# Mover la caja delimitadora a una nueva posición
new_pos = np.array([0, 0, 500.0])  # Nueva posición deseada
bbox_mesh.translate(new_pos)


# # Recortar la nube de puntos utilizando la caja delimitadora
# cropped_pointcloud = o3d.geometry.crop_point_cloud(pointcloud, np.array([0,0,0]), np.array([600,600,600]))


# ----- fitrar nube por distancia a un eje ----- # 

# Convertir la nube de puntos a un arreglo numpy para un procesamiento más eficiente
points = np.asarray(pointcloud.points)
colors = np.asarray(pointcloud.colors)

# Definir el umbrales
umbral_x_min = -200
umbral_x_max = 200
umbral_z = 600


# Filtrar los puntos basados en la coordenada z
points_filt = points[points[:, 2] <= umbral_z]
colors_filt = colors[points[:, 2] <= umbral_z]
# Filtrar los puntos basados en la coordenada x
points_filt2 = points_filt[(points_filt[:, 0] >= umbral_x_min) & (points_filt[:, 0] <= umbral_x_max)]
colors_filt2 = colors_filt[(points_filt[:, 0] >= umbral_x_min) & (points_filt[:, 0] <= umbral_x_max)]
# Crear una nueva nube de puntos con los puntos filtrados
pointcloud_filt = o3d.geometry.PointCloud()
pointcloud_filt.points = o3d.utility.Vector3dVector(points_filt2)
pointcloud_filt.colors = o3d.utility.Vector3dVector(colors_filt2)


o3d.visualization.draw_geometries([pointcloud_filt, bbox_mesh, line_set])
