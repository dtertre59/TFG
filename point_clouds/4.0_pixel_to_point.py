import open3d as o3d
import cv2
import numpy as np

from pathlib import Path

name = 'square_lab_2'


# ----- imagen ----- # 
img = cv2.imread(str(Path(__file__).resolve().parent / 'assets' / f'{name}.png'), cv2.IMREAD_COLOR)

# Obtener la resolución de la imagen
resolution = img.shape[:2]  # Solo se obtienen los dos primeros elementos de la tupla (altura y ancho)
print('Resolucion: ', resolution) # 720x1080

cv2.imshow('imagen', img)
cv2.waitKey(0)
# cv2.destroyAllWindows()
# ------------------- #


# ----- import point cloud ----- # 
pointcloud = o3d.io.read_point_cloud(str(Path(__file__).resolve().parent / 'clouds' / f'{name}.ply'))
points = np.asarray(pointcloud.points)




# ----- filtrado por pixel ----- # 
pixel = [580, 410]

# Filtrar los puntos de la nube de puntos cuya coordenada x sea mayor o igual que la coordenada x del pixel
points_filt = points[points[:, 1] <= pixel[0]]

# Crear una nueva nube de puntos con los puntos filtrados
pointcloud_filt = o3d.geometry.PointCloud()
pointcloud_filt.points = o3d.utility.Vector3dVector(points_filt)



# ----- eje de coordenadas para la visualizacion ----- # 
axis_points = np.array([[0, 0, 0],  # Origen
                    [100, 0, 0],  # Punto en el eje x
                    [0, 100, 0],  # Punto en el eje y
                    [0, 0, 100]]) # Punto en el eje z
# Crear las líneas que representan los ejes
lines = [[0, 1], [0, 2], [0, 3]]  # Conexiones entre los puntos (origen, x) (origen, y) (origen, z)
# Crear el objeto LineSet
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(axis_points)
line_set.lines = o3d.utility.Vector2iVector(lines)
# ---------------------------------------------------- # 

# o3d.visualization.draw_geometries_with_editing([pointcloud])
o3d.visualization.draw_geometries([pointcloud_filt, line_set])

cv2.destroyAllWindows()