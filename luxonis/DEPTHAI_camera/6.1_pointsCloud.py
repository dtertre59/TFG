import open3d as o3d
import numpy as np


# Generar una nube de puntos aleatoria
num_points = 1000
points = np.random.rand(num_points, 3)

# Crear un objeto PointCloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Guardar la nube de puntos en formato PLY
o3d.io.write_point_cloud("pointsCloud_files/output_cloud_rand.ply", point_cloud)
# Visualizar la nube de puntos
# o3d.visualization.draw_geometries([point_cloud])
o3d.visualization.draw_geometries_with_editing([point_cloud])


# Importar el archivo PLY
ply_path = "pointsCloud_files/output_cloud_rand.ply"  # Ruta al archivo PLY
pcd = o3d.io.read_point_cloud(ply_path)

# Visualizar el objeto PointCloud
o3d.visualization.draw_geometries([pcd])
