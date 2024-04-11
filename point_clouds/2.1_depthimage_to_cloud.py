from pathlib import Path
import cv2
import numpy as np
import open3d as o3d


directory = Path(__file__).resolve().parent
image_directory = directory / 'assets'

def import_image(filename: str):
    frame = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    frame

# Cargar las imágenes de disparidad y confianza
imagen_disparidad = cv2.imread(str(image_directory / 'disparity_1.png'), cv2.IMREAD_GRAYSCALE)
imagen_confianza = cv2.imread(str(image_directory / 'conf_1.png'), cv2.IMREAD_GRAYSCALE)

# Convertir la imagen de disparidad a una matriz de profundidad
factor_escala = 1.0  # Ajusta el factor de escala según sea necesario
matriz_disparidad = imagen_disparidad.astype(np.float32) * factor_escala

# Crear una nube de puntos a partir de la matriz de profundidad
h, w = matriz_disparidad.shape
y, x = np.mgrid[0:h, 0:w]
puntos = np.dstack((x, y, matriz_disparidad)).reshape(-1, 3)

# Invertir los puntos en el eje x
puntos[:, 0] = -puntos[:, 0]
# Rotar la nube de puntos 180 grados en el eje z
puntos[:, :2] = -puntos[:, :2]

# Aplicar la imagen de confianza para filtrar los puntos menos confiables
umbral_confianza = 128  # Ajusta el umbral según sea necesario
mascara_puntos_validos = imagen_confianza > umbral_confianza
puntos_filtrados = puntos[mascara_puntos_validos.flatten()]

# Crear un objeto PointCloud de Open3D
nube_puntos = o3d.geometry.PointCloud()
nube_puntos.points = o3d.utility.Vector3dVector(puntos_filtrados)


# Visualizar la nube de puntos en Open3D
o3d.visualization.draw_geometries_with_editing([nube_puntos])




# ENCONTRAR COORDENADAS DE UN PIXEL DE LA IMAGEN

#  Convertir la nube de puntos a un array numpy para facilitar el acceso
points_array = np.asarray(nube_puntos.points)

# Coordenadas del píxel en la imagen de disparidad
x_pixel = 300
y_pixel = 300

# Encontrar el punto en la nube de puntos que corresponde al píxel
point_index = y_pixel * matriz_disparidad.shape[1] + x_pixel
point = points_array[point_index]

print("Información del punto en la nube de puntos:")
print("Coordenadas (x, y, z):", point)
