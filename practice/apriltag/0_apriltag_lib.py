import apriltag
import cv2
from pathlib import Path
import numpy as np

width = 1280
height = 720

# extraccion de la imagen
img = cv2.imread(str(Path(__file__).resolve().parent / 'assets' / 'apriltags_1.png'), cv2.IMREAD_COLOR)
# Redimensiona la imagen utilizando la interpolación de área de OpenCV
img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# parametros de la camara
# focal
fx = 3008.92857 #pixeles. 0.00337 # 3.37mm
fy = fx
# center (resolution/2) -> 1920x1080
cx = width/2
cy = height/2

camera_params = [fx, fy, cx, cy]

tag_size = 0.015



# GET detections
def get_apriltrag_detections(frame) -> list:
    # detector
    detector = apriltag.Detector()

    # opciones del detector
    # print(detector.options.__dict__)

    # detecciones apriltags de la imagen
    detections = detector.detect(frame)
    return detector, detections

# GET detection POSE
def get_detection_pose(detector: apriltag.Detector, detection, camera_params, tag_size):
    # conseguimos matriz de transformacion
    transformation_matrix, initial_error, final_error = detector.detection_pose(detection, camera_params=camera_params, tag_size=tag_size)
    center = detection.center.astype(int)
    corners = detection.corners.astype(int)
    return  transformation_matrix, center, corners


# Paint axis
def paint_apriltag_axis(frame, center, corners):
    # print(detections)
    color_white = (255, 255, 255)
    color_black = (0,0,0)
    color_red = (0, 0, 255)
    color_green = (0, 255, 0)

    # Dibujar el recuadro del AprilTag
    cv2.line(frame, tuple(corners[0]), tuple(corners[1]), color_white, 2, cv2.LINE_AA, 0)
    cv2.line(frame, tuple(corners[1]), tuple(corners[2]), color_white, 2, cv2.LINE_AA, 0)
    cv2.line(frame, tuple(corners[2]), tuple(corners[3]), color_white, 2, cv2.LINE_AA, 0)
    cv2.line(frame, tuple(corners[3]), tuple(corners[0]), color_white, 2, cv2.LINE_AA, 0)
    

    # dibujar ejes de coordenadas
    x_axis = np.array(((corners[1] + corners[2])/2), dtype=int)
    y_axis = np.array(((corners[0] + corners[1])/2), dtype=int)
    # print(x_axis)
    cv2.line(frame, tuple(center), x_axis, color_red, 2, cv2.LINE_AA, 0)
    cv2.line(frame, tuple(center), y_axis, color_green, 2, cv2.LINE_AA, 0)

    #  Dibujar centro en la imagen
    cv2.circle(frame, tuple(center), 3, color_black, -1)

    


# PRUEBAS

# extraccion de la imagen
name = 'apriltags_6.png'
img = cv2.imread(str(Path(__file__).resolve().parent / 'assets'/ name), cv2.IMREAD_COLOR)
# Redimensiona la imagen utilizando la interpolación de área de OpenCV
img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


detector, detections = get_apriltrag_detections(frame = img_grayscale)

ts = []

for detection in detections:
    t, center, corners = get_detection_pose(detector, detection, camera_params, tag_size)
    ts.append(t)
    paint_apriltag_axis(img, center, corners)


# Mostrar la imagen con el rectángulo y el centro marcados
cv2.imshow('AprilTag', img)

cv2.waitKey()
cv2.destroyAllWindows()

# ----- REPRESENTACION EN 3D ----- # 
import rep_3d

axis_scale = 0.5
fig, ax, t_camera = rep_3d.init_3d_rep()
rep_3d.print_3d_rep(ax, t_camera,axis_scale, c='c', pointname='camera', ax_ref=False)

# t_prueba = np.array([[1, 0, 0, 0],
#                     [0, 1, 0, 0],
#                     [0, 0, 1 ,1],
#                     [0, 0, 0, 1]])

# rep_3d.print_3d_rep(ax, t_prueba, scale, 'r')
rep_3d.print_3d_rep(ax, ts[0], axis_scale, 'r',ax_ref=False)
# for t in ts:
#     # Invertir la matriz de transformación
#     # i_t= np.linalg.inv(t)

#     rep_3d.print_3d_rep(ax, t, axis_scale, 'r',ax_ref=False)

rep_3d.show_3d_rep(fig, ax, 'Sistema de Referencia: Camara')

# --------------------------------- # 
# ----- CALCULO DE DISTANCIAS ----- # 

cam, a1, a2 = rep_3d.distance(ts[0], ts[1])

fig2, ax2, t= rep_3d.init_3d_rep()

rep_3d.print_point(ax2, cam, 'camera', 'b')
rep_3d.print_point(ax2, a1, 'a1', 'r')
rep_3d.print_point(ax2, a2, 'a2', 'g')

rep_3d.print_line(ax2, a1, a2)

distance = rep_3d.points_distance(a1, a2)
print('distancia entre puntos: ', distance)

rep_3d.show_3d_rep(fig2, ax2, '3-1')






