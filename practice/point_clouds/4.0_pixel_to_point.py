import open3d as o3d
import cv2
import numpy as np

from pathlib import Path

name = 'square_lab_2'


# ----- imagen ----- # 
img = cv2.imread(str(Path(__file__).resolve().parent / 'assets' / f'{name}.png'), cv2.IMREAD_COLOR)

# Obtener la resolución de la imagen
resolution = img.shape[:2]  # Solo se obtienen los dos primeros elementos de la tupla (altura y ancho)
print(f'Resolucion: ({resolution[1], resolution[0]})') # 1280x720 = 720P

# editar marcando el cuadrado
center = (int(resolution[1]/2), int(resolution[0]/2))
pixel_min = (580, 380)
pixel_max = (685,550)

print('center: ', center)
# Dibujar el rectángulo en la imagen
img = cv2.rectangle(img, pixel_min, pixel_max, (0,255,0), 2)
img = cv2.rectangle(img, (0,0), center, (0,0,255), thickness=cv2.FILLED)
# cv2.imshow('imagen', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ------------------- #


# ----- import point cloud ----- #  

# GITHUB: https://github.com/davizinho5/TFG_TFM_ETSIDI
# array_original = np.random.random(1280 * 3)  # Ejemplo de array con longitud múltiplo de 1280
# array = np.asarray(array_original)
# # Reshape para construir una matriz con filas de 1280 elementos
# matriz = array.reshape(3, -1) # filas, columnas

# # Imprimir la matriz resultante
# print("Matriz resultante:")
# print(matriz)
# print('dimension: ', matriz.shape)
# input('END')

pointcloud = o3d.io.read_point_cloud(str(Path(__file__).resolve().parent / 'clouds' / f'{name}.ply'))
# adquirimos puntos
points = np.asarray(pointcloud.points)
# puntos
print('Numero de puntos: ', points.size)
print('Puntos (0-4): ', points[:4])
# invertimos puntos respecto al eje x porque esta en espejo
points[:,0] *= -1
#aqui tenemos la nube invertida
pointcloud.points = o3d.utility.Vector3dVector(points)
# Calcular la resolución de la nube de puntos
# resolucion = pointcloud.get_resolution()
# print('resulution pointcloud: ', resolucion)
# construimos matriz 
matrix = points.reshape(-1, 1280*3)
print('dimension: ', matrix.shape)
# print(matrix)
# input('end')

# recortar pointcloud funcion directa
# lista de recorte Por Pixeles del recuadro en 2D
# asi seria para coger todos los puntos: filt = list(range(1280*720)) #resolucion
filt = []

for element in range(720): # 720 filas
    if element > pixel_min[1] and element < pixel_max[1]:
        filt.extend(list(range(((1280*element)+pixel_min[0]), ((1280*element)+pixel_max[0]))))

pointcloud = pointcloud.select_by_index(filt)

#recortar creado un vector con los puntos que queremos unicamente
# p = points[0:100000]



# ----- eje de coordenadas para la visualizacion ----- # 
axis_points = np.array([[0, 0, 0],  # Origen
                    [640, 0, 0],  # Punto en el eje x
                    [0, 360, 0],  # Punto en el eje y
                    [0, 0, 900]]) # Punto en el eje z
# Colores correspondientes a cada punto en la línea (origen, x), (origen, y), (origen, z)
axis_colors = np.array([[255, 0, 0],  # Rojo para el eje x
                   [0, 255, 0],  # Verde para el eje y
                   [0, 0, 255]]) # Azul para el eje z
# Crear las líneas que representan los ejes
axis_lines = [[0, 1], [0, 2], [0, 3]]  # Conexiones entre los puntos (origen, x) (origen, y) (origen, z)
# Crear el objeto LineSet
axis = o3d.geometry.LineSet()
axis.points = o3d.utility.Vector3dVector(axis_points)
axis.lines = o3d.utility.Vector2iVector(axis_lines)
# Asignar los colores a cada línea
axis.colors = o3d.utility.Vector3dVector(axis_colors)
# ---------------------------------------------------- # 


# o3d.visualization.draw_geometries([pointcloud, axis])








# ----- Crear cubo 3d ----- # 
# # Definir una caja delimitadora (en este ejemplo, se define manualmente)
# min_bound = [0, 0, 0]  # Límites mínimos de la caja
# max_bound = [640.0, 360.0, 900]     # Límites máximos de la caja
# bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
# # Crear una malla para visualizar la caja delimitadora
# bbox_mesh = o3d.geometry.TriangleMesh.create_box(width=max_bound[0]-min_bound[0],
#                                                   height=max_bound[1]-min_bound[1],
#                                                   depth=max_bound[2]-min_bound[2])
# bbox_mesh.compute_vertex_normals()
# # Cambiar el color de la caja a rojo
# bbox_mesh.paint_uniform_color([1, 0, 0])  # Color rojo
# ----------------------------# 

# ----- Crear un cubo solo lineas en Open3D ----- #

# width = pixel_max[0] - pixel_min[0]
# height = pixel_max[1] - pixel_min[1]
# # resolucion de la cam
# width = resolution[1]
# height = resolution[0]

# camera_mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=900)
# # Mover la caja delimitadora a una nueva posición
# new_pos = np.array([-width/2, -height/2, 0])  # Nueva posición deseada
# camera_mesh.translate(new_pos)
# # Configurar la visualización para mostrar solo las aristas
# camera_mesh.paint_uniform_color([1, 1, 0])  # Color del cubo
# line_camera_mesh = o3d.geometry.LineSet.create_from_triangle_mesh(camera_mesh)
# line_camera_mesh.paint_uniform_color([0, 0, 0])  # Color de las aristas
# # line_camera_mesh.line_width = 5  # Grosor de las aristas
# ------------------------------------- # 

# ----- Cortar nube de puntos ----- #
points_filt_2 = np.asarray(pointcloud.points)
colors_filt_2 = np.asarray(pointcloud.colors)


# o3d.io.write_point_cloud('practice/point_clouds/clouds/square_lab_2_crop.ply', pointcloud)



# # Definir el umbrales
# umbral_x_max = 30
# umbral_x_min = -40
# umbral_y_max = -10
# umbral_y_min = -90

umbral_z_max = 490
umbral_z_min = 420



# # Filtrar los puntos basados en la coordenada x
# points_filt = points[(points[:, 0] <= umbral_x_max) & (points[:, 0] >= umbral_x_min)]
# colors_filt = colors[(points[:, 0] <= umbral_x_max) & (points[:, 0] >= umbral_x_min)]

# #filtro y
# points_filt_2 = points_filt[(points_filt[:, 1] <= umbral_y_max) & (points_filt[:, 1] >= umbral_y_min)]
# colors_filt_2 = colors_filt[(points_filt[:, 1] <= umbral_y_max) & (points_filt[:, 1] >= umbral_y_min)]

#filtro z
points_filt_3 = points_filt_2[(points_filt_2[:, 2] <= umbral_z_max) & (points_filt_2[:, 2] >= umbral_z_min)]
colors_filt_3 = colors_filt_2[(points_filt_2[:, 2] <= umbral_z_max) & (points_filt_2[:, 2] >= umbral_z_min)]

# Crear una nueva nube de puntos con los puntos filtrados
pointcloud_filt = o3d.geometry.PointCloud()
pointcloud_filt.points = o3d.utility.Vector3dVector(points_filt_3)
pointcloud_filt.colors = o3d.utility.Vector3dVector(colors_filt_3)

#hay muchos menos puntos
print('Puntos cortados: ', np.asanyarray(pointcloud_filt.points).size)

# ----- VISUALIZACIONES ----- #  
# o3d.visualization.draw_geometries_with_editing([pointcloud_filt])
# o3d.visualization.draw_geometries([pointcloud, axis, line_camera_mesh, bbox_mesh])
o3d.visualization.draw_geometries([pointcloud_filt, axis])
cv2.destroyAllWindows()