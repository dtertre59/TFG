
import cv2

import pupil_apriltags
from pathlib import Path
import numpy as np

width = 1280
height = 720


# parametros de la camara
# focal
fx = 3008.92857 #pixeles. 0.00337 # 3.37mm
fy = fx
# center (resolution/2) -> 1920x1080
cx = width/2
cy = height/2

camera_params = [fx, fy, cx, cy]

tag_size = 0.085

# GET detection POSE
def get_detections(detector: pupil_apriltags.Detector, frame, camera_params, tag_size):
    # conseguimos matriz de transformacion
    detections = detector.detect(frame, True, camera_params=camera_params, tag_size=tag_size)
    return detections


def get_center_corners(detection):
    center = detection.center.astype(int)
    corners = detection.corners.astype(int)
    return center, corners

def get_transformation_matrix(detection):
    # Combinar matriz de rotación y vector de traslación en una matriz de transformación homogénea
    pose_matrix = np.hstack((detection.pose_R, detection.pose_t))
    pose_matrix = np.vstack((pose_matrix, [0, 0, 0, 1]))
    return pose_matrix

# Paint axis
def paint_apriltag_axis(frame, center, corners):
    # print(detections)
    color_white = (255, 255, 255)
    color_black = (0,0,0)
    color_red = (0, 0, 255)
    color_green = (0, 255, 0)
    color_blue = (255, 0, 0)

    # Dibujar el recuadro del AprilTag
    cv2.line(frame, tuple(corners[0]), tuple(corners[1]), color_white, 2, cv2.LINE_AA, 0)
    cv2.line(frame, tuple(corners[1]), tuple(corners[2]), color_white, 2, cv2.LINE_AA, 0)
    cv2.line(frame, tuple(corners[2]), tuple(corners[3]), color_white, 2, cv2.LINE_AA, 0)
    cv2.line(frame, tuple(corners[3]), tuple(corners[0]), color_white, 2, cv2.LINE_AA, 0)
    

    # dibujar ejes de coordenadas
    x_axis = np.array(((corners[1] + corners[2])/2), dtype=int)
    y_axis = np.array(((corners[2] + corners[3])/2), dtype=int)


    # print(x_axis)
    cv2.line(frame, tuple(center), x_axis, color_red, 2, cv2.LINE_AA, 0)
    cv2.line(frame, tuple(center), y_axis, color_green, 2, cv2.LINE_AA, 0)

    #  Dibujar centro del tag
    cv2.circle(frame, tuple(center), 3, color_black, -1)

    # dibujar centro de la imagen
    ccenter = tuple((int(width/2), int(height/2)))
    cv2.circle(frame, ccenter, 3, color_black, -1)


    # Mostrar la imagen con el rectángulo y el centro marcados
    cv2.imshow('AprilTag', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# extraccion de la imagen
img = cv2.imread(str(Path(__file__).resolve().parent / 'assets/apriltag_2.png'), cv2.IMREAD_COLOR)
# Redimensiona la imagen utilizando la interpolación de área de OpenCV
img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


detector = pupil_apriltags.Detector()
detections = get_detections(detector, img_grayscale, camera_params, tag_size)

T = get_transformation_matrix(detection=detections[0])
center, corners = get_center_corners(detection=detections[0])

print(T)

paint_apriltag_axis(img, center, corners)

# # Verificar si se detectó algún AprilTag y ver resultados
# if detection:

#     # Obtener las coordenadas de los vértices y el centro del primer AprilTag detectado
#     corners = detection.corners.astype(int)
#     center = detection.center.astype(int)


#     color = (0, 255, 0)

#     # Dibujar el recuadro del AprilTag y el centro en la imagen
#     cv2.line(img, tuple(corners[0]), tuple(corners[1]), color, 2, cv2.LINE_AA, 0)
#     cv2.line(img, tuple(corners[1]), tuple(corners[2]), color, 2, cv2.LINE_AA, 0)
#     cv2.line(img, tuple(corners[2]), tuple(corners[3]), color, 2, cv2.LINE_AA, 0)
#     cv2.line(img, tuple(corners[3]), tuple(corners[0]), color, 2, cv2.LINE_AA, 0)

#     cv2.circle(img, tuple(center), 5, (0, 0, 255), -1)

#     # Mostrar la imagen con el rectángulo y el centro marcados
#     cv2.imshow('AprilTag', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("No se detectaron AprilTags en la imagen.")
