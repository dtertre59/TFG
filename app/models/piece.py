"""
        piece.py

    Piezas

"""

# -------------------- PACKAGES ------------------------------------------------------------------------------------------ #
import numpy as np
import cv2

from models.vectors import Vector2D


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
        textcen = """    - Center: {self.center}"""
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
        

    def calculate3dPoint(self, ref: PieceA):
        pass





