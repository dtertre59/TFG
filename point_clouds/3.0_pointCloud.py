import open3d as o3d
import numpy as np
import time
from pathlib import Path

# import point cloud
pointcloud = o3d.io.read_point_cloud(str(Path(__file__).resolve().parent / 'clouds' / 'square_lab_1.ply'))

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
o3d.visualization.draw_geometries_with_editing([pointcloud])
