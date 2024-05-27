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

from models.camera import CameraConfig, Camera
from models.vectors import Vector2D
from models.constants import ColorBGR
from models.piece import BoundingBox, PieceA, PieceN, PieceN2, Piece
from models.robot import Robot

# -------------------- VARIABLES ----------------------------------------------------------------------------------------- #

objects_colors = {
        'circle': (0,0,255),
        'hexagon': (0,255,0),
        'scuare': (255,0,0) # va con q pero hay que cambiarlo en la red neuronal
    }

# -------------------- FUNCTIONS ----------------------------------------------------------------------------------------- #



# -------------------- APRILTAG ------------------------------------------------------------------------------------------ #

class ApriltagConfig():
    """Configuracion del Apriltag"""
    def __init__(self, family: str, size: float) -> None:
        self.family = family  # Familia del AprilTag
        self.size = size  # Tamaño del AprilTag

        
class Apriltag(ApriltagConfig):
    """Apriltag completo. configuracion mas funcionalidades
    formado por piezas de tipo A"""
    def __init__(self, family: str, size: float) -> None:
        super().__init__(family, size)

        self.detector = pupil_apriltags.Detector(families=family)

        self.detections = None
        self.pieces = []

    def detect(self, frame: np.ndarray, camera_params: list):
        # 1. frame to grayscale
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 2. detections
        self.detections = self.detector.detect(frame_grayscale, True, camera_params=camera_params, tag_size=self.size)
        # 3. instancias de piezas
        self.pieces = []
        for detection in self.detections:
            id = detection.tag_id
            center = detection.center.astype(int)
            cors = detection.corners.astype(int)
            corners = []
            for corner in cors:
                corners.append(Vector2D(corner))
            # transformation matrix
            T = np.hstack((detection.pose_R, detection.pose_t))
            T = np.vstack((T, [0, 0, 0, 1]))
            # Rotate 180 degrees over the x-axis to get it properly aligned (library issue)
            rot = np.array([[1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]])
            T = np.dot(T, rot)

            piece = PieceA(name=str(id), color=(0,0,0), center=Vector2D(center), corners=corners, T=T)
            self.pieces.append(piece)

        return

    def paint(self, frame: np.ndarray) -> None:
        """Pintamos los ejes del apriltag detectado en la imagen"""
        for piece in self.pieces:
            piece.paint(frame)
        return


# -------------------- NEURONAL NETWORKS --------------------------------------------------------------------------------- #

class YoloBaseModel(ABC): # el abs es para el abstract
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
    """Deteccion de objetos en una imagen con YOLOv8 OBJECT DETECTION
    formado de piezas de tipoN"""
    def __init__(self, filename: str) -> None:
        super().__init__(filename)
        self.pieces = []


    def detect(self, frame: np.ndarray) -> None:
        """Deteccion con red neuronal"""
        # 1. deteccion
        self.detections = list(self.model(frame, stream=True))
        # 2. instancias de las piezas detectadas
        self.pieces = []
        for detection in self.detections:
            # identificación de objetos
            objects = detection.boxes.cls.numpy().tolist()
            # diccionario de nombres de la red
            names = detection.names     
            if objects:
                for index, object in enumerate(objects):
                    # coodenadas de cada objeto
                    coordinates = detection.boxes.xyxy[index].numpy()
                    # nombre de la pieza detectad
                    piece_name = names[object]
                    # color asociado
                    piece_color = ColorBGR.get_piece_color(name=piece_name)
                    # creamos instancia de la pieza
                    bbox = BoundingBox(p1 = np.array([int(coordinates[0]), int(coordinates[1])]), p2=np.array([int(coordinates[2]), int(coordinates[3])]))
                    print('Nombre: ', piece_name)
                    print(bbox)
                    piece = PieceN(name=piece_name, color=piece_color, bbox=bbox)
                    # print(piece)
                    self.pieces.append(piece)

        return


    def paint(self, frame: np.ndarray) -> None:
        if self.pieces:
            for piece in self.pieces:
                piece.paint(frame)
        return 

     
class YoloPoseEstimation(YoloBaseModel):
    def __init__(self, filename: str) -> None:
        super().__init__(filename)
        self.pieces = []
        return

    def detect(self, frame: np.ndarray) -> None:
        # 1. detecciones
        self.detections = list(self.model(frame, stream=True))
        # 2. instancias de las piezas detectadas
        self.pieces = []

        for detection in self.detections:
            # identificación de objetos
            objects = detection.boxes.cls.numpy().tolist()
            # diccionario de nombres de la red
            names = detection.names
            if objects:
                for index, ob in enumerate(objects):
                    # coodenadas de cada objeto
                    piece_name = names[ob]
                    # color asociado
                    piece_color = ColorBGR.get_piece_color(name=piece_name)
                    coordinates = detection.boxes.xyxy[index].numpy()
                    bbox = BoundingBox(p1 = np.array([int(coordinates[0]), int(coordinates[1])]), p2=np.array([int(coordinates[2]), int(coordinates[3])]))                    
                    # center = detection.keypoints.xy[index][-1].int().tolist()
                    # corners = detection.keypoints.xy[index][:-1].int().tolist()
                    keypoints = detection.keypoints.xy[index].int().tolist()
                    piece = PieceN2(name=piece_name, color=piece_color, bbox=bbox, keypoints=keypoints)
                    self.pieces.append(piece)
        return
    
    def paint(self, frame: np.ndarray) -> None:
        if self.pieces:
            for piece in self.pieces:
                piece.paint(frame)
        return 




    

