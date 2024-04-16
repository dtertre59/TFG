import open3d as o3d
import cv2
import numpy as np

from pathlib import Path

name = 'square_lab_2'

# import point cloud
pointcloud = o3d.io.read_point_cloud(str(Path(__file__).resolve().parent / 'clouds' / f'{name}.ply'))
img = cv2.imread(str(Path(__file__).resolve().parent / 'assets' / f'{name}.png'), cv2.IMREAD_COLOR)

cv2.imshow('imagen', img)
cv2.waitKey(0)
# cv2.destroyAllWindows()

points = np.array([[0, 0, 0],  # Origen
                    [100, 0, 0],  # Punto en el eje x
                    [0, 100, 0],  # Punto en el eje y
                    [0, 0, 100]]) # Punto en el eje z

# Crear las líneas que representan los ejes
lines = [[0, 1], [0, 2], [0, 3]]  # Conexiones entre los puntos (origen, x) (origen, y) (origen, z)

# Crear el objeto LineSet
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines = o3d.utility.Vector2iVector(lines)


# o3d.visualization.draw_geometries_with_editing([pointcloud])
o3d.visualization.draw_geometries([pointcloud, line_set])

cv2.destroyAllWindows()