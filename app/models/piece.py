"""
        piece.py

    Piezas

"""

# -------------------- PACKAGES ------------------------------------------------------------------------------------------ #
import numpy as np
import cv2

from models.vectors import Vector2D, Vector3D, Vector6D
from functions import helper_functions as hf


# -------------------- VARIABLES ----------------------------------------------------------------------------------------- #


# -------------------- CLASSES ------------------------------------------------------------------------------------------- #

class BoundingBox():
    def __init__(self, p1: np.ndarray, p2: Vector2D) -> None:
        self.p1 = Vector2D(p1[0], p1[1])
        self.p2 = Vector2D(p2[0], p2[1])
    
    def __str__(self) -> str:
        text = f"""    - Bounding Box: [p1: {self.p1}, p2: {self.p2}]"""
        return text
    
    def paint(self, frame: np.ndarray, color: tuple = (0, 255, 0)) -> None:
        # Dibujar el rectángulo en la imagen
        cv2.rectangle(img=frame, pt1=(int(self.p1.x), int(self.p1.y)),
                    pt2=(int(self.p2.x), int(self.p2.y)), color=color, thickness=2)
        return


class PieceBase():
    def __init__(self, name: str, color: tuple = (0,255,0)) -> None:
        self.name = name
        self.color = color

    def __str__(self) -> str:
        text = f"""Pieza:
    - Nombre: {self.name}
    - Color: {self.color}"""
        return text



class PieceA(PieceBase):
    def __init__(self, name: str, color: tuple, center: Vector2D, corners: list[Vector2D], T: np.ndarray) -> None:
        super().__init__(name, color)
        self.center= center
        self.corners = corners
        self.T = T

    def __str__(self) -> str:
        textb = super().__str__()
        textcen = f"""    - Center: {self.center}"""
        textc = """    - Corners: ["""
        for corner in self.corners:
            textc += corner.__str__() + ' ,'
        textc = textc[:-2] + ']'
        textT = f'    - Transformation matrix: {self.T}'
        text = f'{textb}\n{textcen}\n{textc}\n{textT}'
        return text
    
    def paint(self, frame) -> None:

        color_white = (255, 255, 255)
        color_black = (0,0,0)
        color_red = (0, 0, 255)
        color_green = (0, 255, 0)

        # Dibujar el recuadro del AprilTag
        cv2.line(frame, (self.corners[0].x, self.corners[0].y), (self.corners[1].x, self.corners[1].y), color_white, 2, cv2.LINE_AA, 0)
        cv2.line(frame, (self.corners[1].x, self.corners[1].y), (self.corners[2].x, self.corners[2].y), color_white, 2, cv2.LINE_AA, 0)
        cv2.line(frame, (self.corners[2].x, self.corners[2].y), (self.corners[3].x, self.corners[3].y), color_white, 2, cv2.LINE_AA, 0)
        cv2.line(frame, (self.corners[3].x, self.corners[3].y), (self.corners[0].x, self.corners[0].y), color_white, 2, cv2.LINE_AA, 0)
        
        # dibujar ejes de coordenadas
        x_axis = np.array(((np.array([self.corners[1].x, self.corners[1].y]) + np.array([self.corners[2].x, self.corners[2].y]))/2), dtype=int)
        y_axis = np.array(((np.array([self.corners[2].x, self.corners[2].y]) + np.array([self.corners[3].x, self.corners[3].y]))/2), dtype=int)

        # print(x_axis)
        cv2.line(frame, (self.center.x, self.center.y), x_axis, color_red, 2, cv2.LINE_AA, 0)
        cv2.line(frame, (self.center.x, self.center.y), y_axis, color_green, 2, cv2.LINE_AA, 0)

        #  Dibujar centro en la imagen
        cv2.circle(frame, (self.center.x, self.center.y), 3, color_black, -1)

        # Escribir el número Id del taf solo si es el de referencia
        if self.name == '4':
            cv2.putText(frame, self.name, (self.center.x, self.center.y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)



class PieceN(PieceBase):
    """La bounding box es el cuadrado delimitador de la pieza. Lo sacamos con la red neuronal object detection"""
    def __init__(self, name: str, color: tuple, bbox: BoundingBox) -> None:
        super().__init__(name, color)
        self.bbox = bbox

    def __str__(self) -> str:
        textb = super().__str__()
        textbbox = self.bbox.__str__()
        text = textb + '\n' + textbbox
        return text
    
    def paint(self, frame: np.ndarray) -> None:
        # Escribir el nombre encima de la pieza
        cv2.putText(frame, self.name, (int(self.bbox.p1.x), int(self.bbox.p1.y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.color, 2)
        # Cuadrado boundingbox
        self.bbox.paint(frame, self.color)
        return
    

class Piece(PieceBase):
    def __init__(self, pieceA: PieceA, pieceN: PieceN) -> None:
        # 1. el nombre y color lo sacamos de la red neuronal que nos diferencia el objeto
        self.pieceA = pieceA
        self.pieceN = pieceN

        super().__init__(pieceN.name, pieceN.color)
        self.bbox = pieceN.bbox

        self.center= pieceA.center
        self.corners = pieceA.corners
        self.T = pieceA.T

        self.point3d = None
        self.pose = None
  
    def __str__(self) -> str:
        textN = self.pieceN.__str__()

        textcen =f'    - Center: {self.center}'
        textc = '    - Corners: ['
        for corner in self.corners:
            textc += corner.__str__() + ' ,'
        textc = textc[:-2] + ']'
        textT = f'    - Transformation matrix: {self.T}'
        textA = f'{textcen}\n{textc}\n{textT}'

        text = f'{textN}\n{textA}'
        return text

    def paint(self, frame) -> None:
        self.pieceA.paint(frame)
        self.pieceN.paint(frame)
        return

    def validate(self) -> bool:
        if (self.bbox.p1.x < self.center.x) and (self.bbox.p1.y < self.center.y) and (self.bbox.p2.x > self.center.x) and (self.bbox.p2.y > self.center.y):
            return True
        else:
            return False
        

    def calculatePose(self, ref: PieceA, t_ref_to_robot: np.ndarray = np.eye(4)):
        # print(t_ref_to_robot)
        t_ref_to_cam = ref.T
        t_piece_to_cam = self.T

        # 1. metodo
        # puntos de origen de los sistemas de coordenadas
        pcam_cam = pref_ref = ppieze_pieze = prob_rob = np.array([0 ,0, 0])

        # puntos respecto de la camara (ref, pieze )
        pref_cam = hf.point_tansf(t_ref_to_cam, pref_ref)
        ppieze_cam = hf.point_tansf(t_piece_to_cam, ppieze_pieze)
        
        # puntos respecto de ref (cam -> ref)
        pcam_ref = hf.point_tansf(np.linalg.inv(t_ref_to_cam), pcam_cam)
        ppieze_ref = hf.point_tansf(np.linalg.inv(t_ref_to_cam), ppieze_cam)

        # puntos respecto a la base del robot
        t_ref_to_robot = t_ref_to_robot
        
        pcam_rob = hf.point_tansf(t_ref_to_robot, pcam_ref)
        pref_rob = hf.point_tansf(t_ref_to_robot, pref_ref)
        ppieze_rob = hf.point_tansf(t_ref_to_robot, ppieze_ref)

        # 2. metodo
        t_piece_to_ref = np.dot(np.linalg.inv(t_ref_to_cam), t_piece_to_cam)
        t_piece_to_robot = np.dot(t_ref_to_robot,t_piece_to_ref)
        t_cam_to_robot = np.dot(t_ref_to_robot, np.linalg.inv(t_ref_to_cam))

        ppiece_robot = hf.point_tansf(t_piece_to_robot, np.array([0,0,0]))

        pose_robot = hf.pose_transf(t_piece_to_robot, np.array([0,0,0]))
        print('Matriz de transicion de la pieza al robot: ', t_piece_to_robot)
        print('Pose de la pieza respecto del robot: ', pose_robot)
        input()
        self.point3d = Vector3D(ppiece_robot[0], ppiece_robot[1], ppiece_robot[2]) # =  Vector6D(pose_robot[0], pose_robot[1], pose_robot[2])

        self.pose = Vector6D(pose_robot[0], pose_robot[1], pose_robot[2], pose_robot[3], pose_robot[4], pose_robot[5])

        # 3D representation
        size = 0.2
        robot_axes = np.array([[size, 0, 0],
                            [0, size, 0],
                            [0, 0, size]])
        ref_axes = np.dot(t_ref_to_robot[:3, :3], robot_axes.T).T
        piece_axes = np.dot(t_piece_to_robot[:3, :3], robot_axes.T).T
        cam_axes = np.dot(t_cam_to_robot[:3, :3], robot_axes.T).T

        fig, ax = hf.init_mat3d()
        hf.add_point_with_axes(ax, prob_rob, robot_axes, 'robot', 'k')
        hf.add_point_with_axes(ax, pref_rob, ref_axes, 'ref', 'r')
        hf.add_point_with_axes(ax, pcam_rob, cam_axes, 'camera', 'b')
        hf.add_point_with_axes(ax, ppiece_robot, piece_axes, 'piece', 'g')

        hf.show_mat3d(fig, ax, 'apriltags representation')
        return 





