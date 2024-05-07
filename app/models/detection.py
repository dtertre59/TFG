"""
        detection.py

    Manejo de detecciones

"""

# -------------------- PACKAGES ------------------------------------------------------------------------------------------ #

import cv2
import numpy as np
from abc import ABC, abstractmethod
# apriltags
import pupil_apriltags
# neuronal network
from ultralytics import YOLO

from models.myCamera import MyCameraConfig, MyCamera

from models.vectors import Vector2D


# -------------------- VARIABLES ----------------------------------------------------------------------------------------- #

objects_colors = {
        'circle': (0,0,255),
        'hexagon': (0,255,0),
        'scuare': (255,0,0) # va con q pero hay que cambiarlo en la red neuronal
    }

# -------------------- FUNCTIONS ----------------------------------------------------------------------------------------- #

class PieceBase():
    def __init__(self, name: str, corners: np.ndarray) -> None:
        self.name = name
        self.corners = corners
        


class Piece:
    def __init__(self, name: str, coordinates: np.ndarray, color: tuple):
        self.name = name
        self.coordinates = coordinates
        self.color = color


# -------------------- APRILTAG ------------------------------------------------------------------------------------------ #
class ApriltagConfig():
    """Configuracion del Apriltag"""
    def __init__(self, family: str, size: float) -> None:
        self.family = family  # Familia del AprilTag
        self.size = size  # Tamaño del AprilTag

        
class Apriltag(ApriltagConfig):
    """Apriltag completo. configuracion mas funcionalidades"""
    def __init__(self, family: str, size: float) -> None:
        super().__init__(family, size)

        self.detector = pupil_apriltags.Detector(families=family)

        self.detections = None

    def paint(self, frame: np.ndarray, index: int) -> np.ndarray:
        """Pintamos los ejes del apriltag detectado en la imagen"""
        detection = self.detections[index]
        color_white = (255, 255, 255)
        color_black = (0,0,0)
        color_red = (0, 0, 255)
        color_green = (0, 255, 0)

        id = detection.tag_id
        center = detection.center.astype(int)
        corners = detection.corners.astype(int)

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

        #  Dibujar centro en la imagen
        cv2.circle(frame, tuple(center), 3, color_black, -1)

        # Escribir el número Id del taf
        cv2.putText(frame, str(id), tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return frame

    def detect(self, frame: np.ndarray, camera_params: list):
        # 1. frame to grayscale
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 2. detections
        self.detections = self.detector.detect(frame_grayscale, True, camera_params=camera_params, tag_size=self.size)
        return


# -------------------- NEURONAL NETWORKS --------------------------------------------------------------------------------- #

class YoloBaseModel(ABC):
    def __init__(self, filename: str) -> None:
        self.model = YOLO(filename)
        self.detections = None

    # @abstractmethod
    # def paint():
    #     pass
    # @abstractmethod
    # def detect():
    #     pass



class YoloObjectDetection(YoloBaseModel):
    """Deteccion de objetos en una imagen con YOLOv8 OBJECT DETECTION"""
    def __init__(self, filename: str) -> None:
        super().__init__(filename)

    # repasar  IDEX, DETECTIONS Y OBJECTS
    def paint(self, frame: np.ndarray, index: int) -> np.ndarray:
        """Pintar bounding box de la denteccion de la pieza"""
        detection = self.detections[index]
        objects = detection.boxes.cls.numpy().tolist()  
        names = detection.names

        if objects:
            for index, object in enumerate(objects):
                coordinates = detection.boxes.xyxy[index].numpy() # .tolist()
                # nombre del objeto detectado
                object_name = names[object]
                # color asociado
                object_color = objects_colors[object_name]

                # Escribir el nombre encima del rectángulo
                cv2.putText(frame, object_name, (int(coordinates[0]), int(coordinates[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, object_color, 2)
                # Dibujar el rectángulo en la imagen
                cv2.rectangle(img=frame, pt1=(int(coordinates[0]), int(coordinates[1])),
                            pt2=(int(coordinates[2]), int(coordinates[3])), color=object_color, thickness=2)

        return frame

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        """Deteccion con red neuronal"""
        self.detections = list(self.model(frame, stream=True))
        return

        
class YoloPoseEstimation(YoloBaseModel):
    def __init__(self, filename: str) -> None:
        super().__init__(filename)



# -------------------- DETECTIONS COORDINATOR ---------------------------------------------------------------------------- #

class DetectionsCoordinator():

    @staticmethod
    def apriltag_detections(frame, camera: MyCamera, apriltag: Apriltag):
        # 1. Camera params
        camera_params = [camera.f.x, camera.f.y, camera.c.x, camera.c.y]
        # 2. deteccion
        apriltag.detect(frame, camera_params)
        # 3. verificacion y paint
        if apriltag.detections:
            for i, detection in enumerate(apriltag.detections):
                frame = apriltag.paint(frame, i)
            return frame, True
        else:
            return frame, False

      
    @staticmethod
    def nn_object_detections(frame, camera: MyCamera, nn_model: YoloObjectDetection):
        # 1. deteccion
        nn_model.detect(frame)
        # 2. verificacion y paint
        if nn_model.detections:
            for i, detection in enumerate(nn_model.detections):
                if detection.obb == None:
                    return frame, False
                else:
                    frame = nn_model.paint(frame, i)
                    return frame, True
            
        else:
            return frame, False


    @staticmethod
    def nn_poseEstimation_detections(frame, camera: MyCamera, nn_model: YoloPoseEstimation):
        pass