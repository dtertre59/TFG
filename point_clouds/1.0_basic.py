import cv2
import open3d as o3d
import numpy as np
import time
from pathlib import Path

# Obtener la ubicación del script actual
script_dir = Path(__file__).resolve().parent

filename = script_dir / 'clouds' / 'rand_0.ply'


# Generar una nube de puntos aleatoria
num_points = 1000
points = np.random.rand(num_points, 3)

def create_pointcloud(points) -> o3d.geometry.PointCloud: 
    # Crear un objeto PointCloud
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    return pointcloud

def create_axis(normalized: bool = True):
    if normalized:
        # Crear el eje de coordenadas
        axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        return axis_mesh
    else:
         # Crear los puntos de los extremos de los ejes
        points = np.array([[0, 0, 0],  # Origen
                        [1, 0, 0],  # Punto en el eje x
                        [0, 1, 0],  # Punto en el eje y
                        [0, 0, 1]]) # Punto en el eje z

        # Crear las líneas que representan los ejes
        lines = [[0, 1], [0, 2], [0, 3]]  # Conexiones entre los puntos (origen, x) (origen, y) (origen, z)

        # Crear el objeto LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
    
        return line_set



def export_pointcloud(filename: str, pointcloud):
    # Guardar la nube de puntos en formato PLY
    o3d.io.write_point_cloud(filename, pointcloud)


def import_pointcloud(filename: str) -> o3d.geometry.PointCloud:
    # Importar el archivo PLY
    pointcloud = o3d.io.read_point_cloud(str(filename))
    return pointcloud


def visualization(geometries: list, edit: bool = False):
    # Visualizar el objeto PointCloud
    if edit:
        o3d.visualization.draw_geometries_with_editing(geometries)
    else:
        o3d.visualization.draw_geometries(geometries)



def create_visualizer(geometries: list) -> o3d.visualization.Visualizer:
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


## FALTA EL DESTRY WINDOWS CUANDO TPCAS UNA TECLAS


if __name__ == '__main__':
    pointcloud = create_pointcloud(points=points)
    axis = create_axis(normalized=False)

    # Visualizar la nube de puntos y el eje de coordenadas
    # visualization([pointcloud, axis], edit=False)

    visualizer = create_visualizer([pointcloud, axis])
    while True:
        update_visualizer(visualizer, [pointcloud])
        R = np.array([[np.cos(0.1), -np.sin(0.1), 0],
                    [np.sin(0.1), np.cos(0.1), 0],
                    [0, 0, 1]])
        points = np.dot(points, R.T)
        
        # Actualizar las coordenadas de los puntos
        pointcloud.points = o3d.utility.Vector3dVector(points)
        time.sleep(0.1)
    
    visualizer.destroyAllWindows()
    # Cerrar la visualización al salir del bucle
    visualizer.destroy_window()

        



