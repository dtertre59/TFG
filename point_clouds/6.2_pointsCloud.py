import open3d as o3d
import numpy as np
import time

# Configuración inicial
num_points = 300
radius = 1.0
omega = 0.1  # Velocidad angular

# Generar puntos en un círculo
angles = np.linspace(0, 2*np.pi, num_points)
x = radius * np.cos(angles)
y = radius * np.sin(angles)
z = np.zeros_like(x)
points = np.column_stack((x, y, z))


# Generar colores que van de rojo a azul
hue_values = np.linspace(0, 240, num_points)  # Tonos en el rango de 0 a 240 (rojo a azul)
colors = np.zeros((num_points, 3))
colors[:, 0] = hue_values  # Asignar tonos a los canales de color rojo

# Crear un objeto PointCloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors/255)

# Visualizar la nube de puntos
o3d.visualization.draw_geometries([point_cloud])


# Visualizar la nube de puntos
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(point_cloud)


# Loop para rotar los puntos en círculos
while True:

    
    # Aplicar rotación a los puntos
    R = np.array([[np.cos(omega), -np.sin(omega), 0],
                  [np.sin(omega), np.cos(omega), 0],
                  [0, 0, 1]])
    points = np.dot(points, R.T)
    
    # Actualizar las coordenadas de los puntos
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    # Actualizar la visualización
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    
    # Esperar un tiempo breve para simular movimiento
    time.sleep(0.2)

# Cerrar la visualización al salir del bucle
vis.destroy_window()
