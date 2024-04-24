# import apriltag
import cv2
from pathlib import Path
import numpy as np

import pupil_apriltags as apriltag

from models.camera import CameraConfig, ApriltagConfig, Apriltag


# INIT detector
def init_detector(families: str = "tag36h11") -> apriltag.Detector:
    return apriltag.Detector(families=families)

# GET detections
def get_detections(detector: apriltag.Detector, frame: np.ndarray, camera_config: CameraConfig, apriltag_config: ApriltagConfig) -> list[apriltag.Detection]:
    # parametros de la camara
    # camera_params = [3156.71852, 3129.52243, 359.097908, 239.736909]
    camera_params = [camera_config.f.x, camera_config.f.y, camera_config.c.x, camera_config.c.y]
    # transformamos imagen a escala de grises
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detecciones
    detections = detector.detect(frame_grayscale, True, camera_params=camera_params, tag_size=apriltag_config.size)
    return detections

# GET apriltag center and corners
def get_center_corners(detection: apriltag.Detection) -> tuple:
    center = detection.center.astype(int)
    corners = detection.corners.astype(int)
    return center, corners

# GET apriltag center and x, y axis
def get_center_x_y_axis(detection: apriltag.Detection) -> tuple:
    center, corners = get_center_corners(detection)
    x_axis = np.array(((corners[1] + corners[2])/2), dtype=int)
    y_axis = np.array(((corners[2] + corners[3])/2), dtype=int)
    
    return center, x_axis, y_axis

# GET Transformation matrix
def get_transformation_matrix(detection: apriltag.Detection) -> np.ndarray:
    # Combinar matriz de rotación y vector de traslación en una matriz de transformación homogénea
    T = np.hstack((detection.pose_R, detection.pose_t))
    T = np.vstack((T, [0, 0, 0, 1]))
    # rotamos 180 sobre el eje x para que qede ajustada (problema de la libreria)
    # rot = np.array([[1, 0, 0, 0],
    #                 [0, -1, 0, 0],
    #                 [0, 0, -1, 0],
    #                 [0, 0, 0, 1]])
    # T = np.dot(T, rot)
    return T

# Paint axis
def paint_apriltag(frame: np.ndarray, detection: apriltag.Detection) -> None:
    # print(detections)
    color_white = (255, 255, 255)
    color_black = (0,0,0)
    color_red = (0, 0, 255)
    color_green = (0, 255, 0)

    id = detection.tag_id
    center, corners = get_center_corners(detection) 

    # Dibujar el recuadro del AprilTag
    cv2.line(frame, tuple(corners[0]), tuple(corners[1]), color_white, 2, cv2.LINE_AA, 0)
    cv2.line(frame, tuple(corners[1]), tuple(corners[2]), color_white, 2, cv2.LINE_AA, 0)
    cv2.line(frame, tuple(corners[2]), tuple(corners[3]), color_white, 2, cv2.LINE_AA, 0)
    cv2.line(frame, tuple(corners[3]), tuple(corners[0]), color_white, 2, cv2.LINE_AA, 0)
    
    # dibujar ejes de coordenadas
    center, x_axis, y_axis = get_center_x_y_axis(detection)
    # print(x_axis)
    cv2.line(frame, tuple(center), x_axis, color_red, 2, cv2.LINE_AA, 0)
    cv2.line(frame, tuple(center), y_axis, color_green, 2, cv2.LINE_AA, 0)

    #  Dibujar centro en la imagen
    cv2.circle(frame, tuple(center), 3, color_black, -1)

    # Escribir el número Id del taf
    cv2.putText(frame, str(id), tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return


# GET APRIL CLASS STRUCTURE SIN ACABAR
def get_april(detection: apriltag.Detection, apriltag_config: ApriltagConfig) -> Apriltag:
    center, corners = get_center_corners(detection)
    T = get_transformation_matrix(detection)
    # ap = Apriltag(detection.tag_id, family=detection.tag_family, size=apriltag_config.size, c)
    return






# -------------------- PRUEBAS --------------------------------

# # CONFIG
# camera_config = CameraConfig(width=1280, height=720, fx= 3008.92857, fy=3008.92857)
# apriltag_config = ApriltagConfig(family='tag36h11', size=0.015)

# # ADD IMAGE
# img = cv2.imread(str(Path(__file__).resolve().parent.parent / 'assets'/'apriltags_6.png'), cv2.IMREAD_COLOR)
# # Redimensiona la imagen utilizando la interpolación de área de OpenCV
# img = cv2.resize(img, (camera_config.resolution.x, camera_config.resolution.y), interpolation=cv2.INTER_AREA)
# img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # DETECION
# detector = init_detector(families=apriltag_config.family)
# detections = get_detections(detector, img_grayscale, camera_config, apriltag_config)

# # PAINT IMAGE
# for detection in detections:
#     center, corners = get_center_corners(detection)
#     paint_apriltag(img, detection)


# print(get_transformation_matrix(detections[1]))

# # Mostrar la imagen con el rectángulo y el centro marcados
# cv2.imshow('AprilTag', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()