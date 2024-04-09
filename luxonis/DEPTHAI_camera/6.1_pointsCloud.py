import open3d as o3d
import numpy as np


# Generar una nube de puntos aleatoria
num_points = 1000
points = np.random.rand(num_points, 3)

# Crear un objeto PointCloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Visualizar la nube de puntos
o3d.visualization.draw_geometries([point_cloud])
