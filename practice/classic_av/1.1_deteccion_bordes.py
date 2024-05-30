import numpy as np
import cv2
from matplotlib import pyplot as plt
 
from sklearn.cluster import DBSCAN



def crop_frame(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:

    new_frame = frame[corners[0][1]:corners[1][1], corners[0][0]:corners[1][0]]

    return new_frame

# En escala de grises la imagen
def process_frame_wb(frame):
    # Verificar el número de canales en la imagen
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        # La imagen es en color (BGR), convertir a escala de grises
        frame_wb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif len(frame.shape) == 2:
        # La imagen ya está en escala de grises
        print('imagen ya en escala de grises')
        frame_wb = frame
    else:
        raise ValueError("Formato de imagen no soportado")

    return frame_wb


# filtros
def filters(frame: np.ndarray) -> np.ndarray:
    # blanco y negro recortada
    # Convertir a escala de grises
    frame_wb = process_frame_wb(frame)

    # Preprocesamiento: aplicar un filtro de desenfoque
    # frame_wb = cv2.GaussianBlur(frame_wb, (5, 5), 0)

    # edge_frame = cv2.Canny(frame,80,200)
    return frame_wb


def point_agrup(points: np.ndarray):
    # Ahora, vamos a aplicar DBSCAN para agrupar las esquinas
    epsilon = 10  # Radio de la vecindad
    min_samples = 5  # Número mínimo de puntos para formar un cluster
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    dbscan.fit(points)
    labels = dbscan.labels_
    # Encontrar los índices de los puntos que pertenecen a cada grupo
    unique_labels = np.unique(labels)
    # print('unique labels: ',unique_labels)

    grouped_points = [points[labels == label] for label in unique_labels if label != -1]  # Excluir el ruido (-1)
    # print('grouped points: ', grouped_points)

    # Calcular los centroides de cada grupo
    group_centroids = [np.mean(group, axis=0) for group in grouped_points]
    print('Centroides: ', group_centroids)
    return group_centroids



def detect_corners_harris(frame: np.ndarray, block_size: float = 7, ksize: float = 7, k: float = 0.01) -> np.ndarray:

    block_size = 5  # Tamaño del vecindario
    ksize = 5   # Tamaño del kernel de Sobel para derivadas
    k = 0.01  # Parámetro libre del detector de Harris

    # Aplicar el detector de esquinas de Harris; obtenemos matriz de respuesta de esquina
    dst = cv2.cornerHarris(frame_wb, block_size,ksize, k)
    # Dilatar el resultado para marcar mejor las esquinas (opcional)
    dst = cv2.dilate(dst,None)
    
    # frame[dst>0.01*dst.max()] = 0

    umbral = 0.01*dst.max()

    # posicion de los pixel corners hay muchos. hay que agruparlos en zonas
    corners = np.argwhere(dst > umbral)
    print('corners: ', len(corners))

    # frame_with_detections = frame.copy()
    # for corner in corners:
    #     # Invertir las coordenadas para que coincidan con (x, y)
    #     x, y = corner[::-1]
    #     cv2.circle(frame_with_detections, (x,y), 3, 0, 1)
    # return frame_with_detections

    corns = point_agrup(corners)
    print(corns)

    # Dibujar círculos en los centroides de los grupos de esquinas
    frame_with_detections = frame.copy()

    for corner in corns:
        corner = corner.astype(int)
        # Invertir las coordenadas para que coincidan con (x, y)
        x, y = corner[::-1]

        cv2.circle(frame_with_detections, (x,y), 3, 0, -1)


    return frame_with_detections


def detect_corners_shi_tomasi(frame):
    # Parámetros para Shi-Tomasi
    max_corners = 6
    quality_level = 0.005
    min_distance = 10
    block_size = 7

    # Detectar esquinas con Shi-Tomasi
    corners = cv2.goodFeaturesToTrack(frame, max_corners, quality_level, min_distance, blockSize=block_size)
    corners = np.int0(corners)

    # Dibujar las esquinas detectadas
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    return frame


def detect_contorns_1(frame: np.ndarray):
    # Aplicar umbralización para segmentar la cara superior del prisma
    _, umbral = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY)

    # Encontrar contornos en la imagen umbralizada
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Seleccionar el contorno más grande (la cara superior del prisma)
    contorno_superior = max(contornos, key=cv2.contourArea)

    # Calcular el centroide del contorno
    momentos = cv2.moments(contorno_superior)
    centro_x = int(momentos["m10"] / momentos["m00"])
    centro_y = int(momentos["m01"] / momentos["m00"])

    frame_with_detections = frame.copy()

    # Calcular los centroides de todos los contornos

    cv2.drawContours(frame_with_detections, [contorno_superior], -1, (0, 255, 0), 2)
    cv2.circle(frame_with_detections, (centro_x, centro_y), 5, (0, 0, 255), -1)
    

    return frame_with_detections


def detect_contorns_2(frame: np.ndarray):

    # Encontrar contornos en la imagen
    edges_cropped = cv2.Canny(frame,80,200)
    contornos, _ = cv2.findContours(edges_cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    # Dibujar los contornos en una copia de la imagen original
    imagen_con_contornos = frame.copy()
    for contorn in contornos:
        cv2.drawContours(imagen_con_contornos, [contorn], 0, (0,255,255), 2)
    return imagen_con_contornos




img = cv2.imread('practice/classic_av/todas.png', cv2.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"



square = np.array([[1070, 430], [1265, 720]])
hexagon = np.array([[768,516], [910,765]])
circle = np.array([[1346,511],[1544,767]])

frame = crop_frame(img, square)

frame_wb = filters(frame)




# 1. mejor
frame_with_detections = detect_corners_harris(frame_wb)


# HACER EL DEL CIRCULLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO (CENTRO)

# frame_with_detections = detect_corners_shi_tomasi(frame_wb)
# frame_with_detections = detect_contorns_1(frame_wb)

# frame_with_detections = detect_contorns_2(frame_wb)

# Mostrar la imagen resultante
cv2.imshow('Imagen con filtro', frame_with_detections)
cv2.waitKey(0)
cv2.destroyAllWindows()
