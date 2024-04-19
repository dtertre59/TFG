import apriltag
import cv2
from pathlib import Path
import numpy as np

width = 1280
height = 720

directory = Path(__file__).resolve().parent.parent


# parametros de la camara
# focal
fx = 0.00337 # 3.37mm
fy = fx
# center (resolution/2) -> 1920x1080
cx = width/2
cy = height/2

camera_params = [fx, fy, cx, cy]

tag_size = 0.0085

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

    # Mostrar la imagen con el rectángulo y el centro marcados
    cv2.imshow('AprilTag', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





# PRUEBAS

# extraccion de la imagen
img = cv2.imread(str(directory / 'assets'/'apriltag_1.png'), cv2.IMREAD_COLOR)
# Redimensiona la imagen utilizando la interpolación de área de OpenCV
img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


detector, detections = get_apriltrag_detections(frame = img_grayscale)

t, center, corners = get_detection_pose(detector, detections[0], camera_params, tag_size)

paint_apriltag_axis(img, center, corners)
