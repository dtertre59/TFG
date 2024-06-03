"""
        piece.py

    Piezas

"""

# -------------------- PACKAGES ------------------------------------------------------------------------------------------ #
import numpy as np
import cv2
import open3d as o3d
from typing import overload, Union
import copy

from models.vectors import Vector2D, Vector3D, Vector6D
from models.constants import ColorBGR, CameraCte

from functions import helper_functions as hf


# -------------------- VARIABLES ----------------------------------------------------------------------------------------- #


# -------------------- CLASSES ------------------------------------------------------------------------------------------- #

# -------------------- BOUNDING BOX -------------------------------------------------------------------------------------- #

class BoundingBox():
    # Constructor
    def __init__(self, p1: np.ndarray, p2: np.ndarray) -> None:
        self.p1 = Vector2D(p1)
        self.p2 = Vector2D(p2)
        return
    
    # Str
    def __str__(self) -> str:
        text = f'Bounding Box: [p1: {self.p1}, p2: {self.p2}]'
        return text
    
    # Pintar
    def paint(self, frame: np.ndarray, color: tuple = ColorBGR.GREEN) -> None:
        # Dibujar el rectángulo en la imagen
        cv2.rectangle(img=frame, pt1=self.p1.get_tuple_int(),
                    pt2=self.p2.get_tuple_int(), color=color, thickness=2)
        return
    
    # As array 
    def get_array(self) -> np.ndarray:
        return np.array([self.p1.get_array(), self.p2.get_array()])

    # As array int 
    def get_array_int(self) -> np.ndarray:
        return np.array([self.p1.get_array_int(), self.p2.get_array_int()])
    
    # Ampliar recuadro
    def expand(self, pixels: int) -> None:
        self.p1.x -= pixels
        self.p1.y -= pixels
        self.p2.x += pixels
        self.p2.y += pixels
        return
        

# -------------------- PIECE BASE ---------------------------------------------------------------------------------------- #

class PieceBase():
    # Constructor
    def __init__(self, name: str, color: tuple = ColorBGR.GREEN) -> None:
        self.name = name
        self.color = color
        return

    # Str
    def __str__(self, title: str = 'Pieza Base') -> str:
        text = f"""----- {title} -----
    - Nombre: {self.name}
    - Color: {self.color}"""
        return text


# -------------------- PIECE A - APRILTAG -------------------------------------------------------------------------------- #

class PieceA(PieceBase):
    # Constructor
    def __init__(self, name: str, color: tuple, center: Vector2D, corners: list[Vector2D], T: np.ndarray) -> None:
        super().__init__(name, color)
        self.center= center
        self.corners = corners
        self.T = T
        return

    # Str
    def __str__(self, title: str = 'Pieza tipo A') -> str:
        textbase = super().__str__(title)
        textcen = f"""    - Center: {self.center}"""
        textc = """    - Corners: ["""
        for corner in self.corners:
            textc += corner.__str__() + ' ,'
        textc = textc[:-2] + ']'
        textT = f'    - Transformation matrix: {self.T}'
        text = f'{textbase}\n{textcen}\n{textc}\n{textT}'
        return text
    
    # Pintar
    def paint(self, frame) -> None:
        # Dibujar el recuadro del AprilTag
        cv2.line(frame, self.corners[0].get_tuple_int(), self.corners[1].get_tuple_int(), ColorBGR.WHITE, 2, cv2.LINE_AA, 0)
        cv2.line(frame, self.corners[1].get_tuple_int(), self.corners[2].get_tuple_int(), ColorBGR.WHITE, 2, cv2.LINE_AA, 0)
        cv2.line(frame, self.corners[2].get_tuple_int(), self.corners[3].get_tuple_int(), ColorBGR.WHITE, 2, cv2.LINE_AA, 0)
        cv2.line(frame, self.corners[3].get_tuple_int(), self.corners[0].get_tuple_int(), ColorBGR.WHITE, 2, cv2.LINE_AA, 0)
        
        # Punto ejes de coordenadas
        x_axis = Vector2D((self.corners[1] + self.corners[2]).get_array()/2)
        y_axis = Vector2D((self.corners[2] + self.corners[3]).get_array()/2)

        # Pintar ejes
        cv2.line(frame, self.center.get_tuple_int(), x_axis.get_tuple_int(), ColorBGR.RED, 2, cv2.LINE_AA, 0)
        cv2.line(frame, self.center.get_tuple_int(), y_axis.get_tuple_int(), ColorBGR.GREEN, 2, cv2.LINE_AA, 0)

        #  Dibujar centro en la imagen
        cv2.circle(frame, self.center.get_tuple_int(), 3, ColorBGR.BLACK, -1)

        # Escribir el número Id del tag solo si es el de referencia
        if self.name == '4':
            cv2.putText(frame, self.name, self.center.get_tuple_int(), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return


# -------------------- PIECE N - OBJECT DETECTION ------------------------------------------------------------------------ #

class PieceN(PieceBase):
    """La bounding box es el cuadrado delimitador de la pieza. Lo sacamos con la red neuronal object detection"""
    # Constructor
    def __init__(self, name: str, color: tuple, bbox: BoundingBox) -> None:
        super().__init__(name, color)
        self.bbox = bbox
        return

    # Str
    def __str__(self, title: str = 'Pieza tipo N') -> str:
        textb = super().__str__(title)
        textbbox = self.bbox.__str__()
        text = textb + '\n    - ' + textbbox
        return text
    
    # Pintar
    def paint(self, frame: np.ndarray) -> None:
        # Escribir el nombre encima de la pieza
        cv2.putText(frame, self.name, (int(self.bbox.p1.x), int(self.bbox.p1.y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.color, 2)
        # Cuadrado boundingbox
        self.bbox.paint(frame, self.color)
        return


# -------------------- PIECE N2 - POSE ESTIMATION ------------------------------------------------------------------------ #

class PieceN2(PieceBase):
    # Constructor
    def __init__(self, name: str, color: tuple, bbox: BoundingBox, keypoints) -> None:
        super().__init__(name, color)
        self.bbox = bbox
        self.keypoints = keypoints

        self.center = None
        self.corners = None
        return

    # Str
    def __str__(self, title: str = 'Pieza tipo N2') -> str:
        textb = super().__str__(title)
        textbbox = self.bbox.__str__()
        text = textb + '\n    - ' + textbbox + '\n    - Center: ' + self.center.__str__() 
        return text

    # Paint
    def paint(self, frame: np.ndarray) -> None:
        # Escribir el nombre encima de la pieza
        cv2.putText(frame, self.name, (int(self.bbox.p1.x), int(self.bbox.p1.y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.color, 2)
        # Cuadrado boundingbox
        self.bbox.paint(frame, self.color)

        # keypoints
        for keypoint in self.keypoints:
            # point
            cv2.circle(frame, tuple(keypoint), 3, ColorBGR.RED, -1)
        
        # Dibujar esqueleto de la pieza
        for index, keypoint in enumerate(self.keypoints):
        #    print(index)
           try:
                cv2.line(frame, (self.keypoints[index][0], self.keypoints[index][1]), (self.keypoints[index+1][0], self.keypoints[index+1][1]), ColorBGR.BLUE, 2, cv2.LINE_AA, 0)
           except:
               pass 

        return


# -------------------- PIECE --------------------------------------------------------------------------------------------- #

class Piece(PieceBase):
    @overload
    def __init__(self, name: str, color: tuple, bbox: BoundingBox, center: Vector2D|None = None, corners: list[Vector2D]|None = None) -> None: ...

    @overload
    def __init__(self, pieceA: PieceA, pieceN: PieceN) -> None: ...

    @overload
    def __init__(self, pieceN: PieceN) -> None: ...
    
    @overload
    def __init__(self, pieceN2: PieceN2) -> None: ...
    
    # Contructor
    def __init__(self, *args: Union[str, tuple, BoundingBox, Vector2D, PieceA, PieceN, PieceN2]) -> None:
        # Normal
        if len(args) > 2 and isinstance(args[0], str) and isinstance(args[1], tuple) and isinstance(args[2], BoundingBox):
            self.apriltag = None
            super().__init__(args[0], args[1])
            self.apriltag = None
            self.bbox = args[2]
            if len(args) > 3:
                self.center = args[3]
            else:
                self.center = None
            self.corners = None
            self.T = None
            self.point3d = None
            self.pose = None

        # Union PieceA y PieceN
        elif len(args) == 2 and isinstance(args[0], PieceA) and isinstance(args[1], PieceN):
    
            pieceA = args[0]
            pieceN = args[1]

            # 1. el nombre y color lo sacamos de la red neuronal que nos diferencia el objeto
            self.apriltag = pieceA

            super().__init__(pieceN.name, pieceN.color)

            self.bbox = pieceN.bbox
            self.center= pieceA.center

            # Verificar que el apriltag se encuentra dentro de la boundingbox
            if (self.bbox.p1.x < self.center.x) and (self.bbox.p1.y < self.center.y) and (self.bbox.p2.x > self.center.x) and (self.bbox.p2.y > self.center.y):
                pass
            else:
                raise ValueError("Ariltag no perteneciente a la pieza")
            
            # No tenemos la informacion de las esquinas de la pieza, solo del apriltag
            self.corners = None

            self.T = pieceA.T

            self.point3d = None
            self.pose = None

        # Unicamente PieceN
        elif len(args) == 1 and isinstance(args[0], PieceN):
            pieceN = args[0]

            self.apriltag = None

            super().__init__(pieceN.name, pieceN.color)

            # bbox
            self.bbox = pieceN.bbox

            self.center= None
            self.corners = None

            # pointcloud + algebra
            self.T = None
            self.point3d = None
            self.pose = None
        
        # Unicamente PieceN2
        elif len(args) == 1 and isinstance(args[0], PieceN2):
            pieceN2 = args[0]

            self.apriltag = None

            super().__init__(pieceN2.name, pieceN2.color)

            # bbox
            self.bbox = pieceN2.bbox

            # keypoints
            if pieceN2.name == 'square':
                self.center= Vector2D(pieceN2.keypoints[-1][0], pieceN2.keypoints[-1][1])
                self.corners = pieceN2.keypoints[:-1]
            else:
                self.center= Vector2D(pieceN2.keypoints[-1][0], pieceN2.keypoints[-1][1])
                self.corners = pieceN2.keypoints[:-1]

            # pointcloud + algebra
            self.T = None
            self.point3d = None
            self.pose = None

        else:
            raise TypeError("Invalid arguments")
        
        return

    # Str
    def __str__(self, title: str = 'Pieza') -> str:
        textbase = super().__str__(title)
        textbbox = self.bbox.__str__()
        textcen =f'    - Center: {self.center}'
        if type(self.corners) == np.ndarray:
            textc = '    - Corners: ['
            for corner in self.corners:
                textc += corner.__str__() + ' ,'
            textc = textc[:-2] + ']'
        else:
            textc = f'    - Corners: {self.corners}'
        textT = f'    - Transformation matrix: {self.T}\n    - Point3d: {self.point3d}\n    - Pose: {self.pose}'
        textA = f'{textcen}\n{textc}\n{textT}'

        text = f'{textbase}\n    - {textbbox}\n{textA}'
        return text

    # Pintar
    def paint(self, frame) -> None:
        # Apriltag
        if self.apriltag:
            self.apriltag.paint(frame)

        # ------------------------------------------------------------------------------------------------
        # if self.pieceN:
        #     self.pieceN.paint(frame)
        #     if self.center:
        #         cv2.circle(frame, tuple(self.center.get_array_int()), 3, color=(0,0,255), thickness=-1)
        #     if type(self.corners) == np.ndarray:
        #         for corner in self.corners:
        #             cv2.circle(frame, (corner[0],corner[1]), 3, 0, -1)
                     
        # if self.pieceN2:
        #     self.pieceN2.paint(frame)
        # ------------------------------------------------------------------------------------------------

        # Bounding box
        self.bbox.paint(frame, self.color)

        # Nombre
        cv2.putText(frame, self.name, (int(self.bbox.p1.x), int(self.bbox.p1.y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.color, 2)

        # Centro
        if isinstance(self.center, Vector2D):
            center = self.center.get_array_int()
            cv2.circle(frame, tuple(center), 3, ColorBGR.RED, -1)
        
        # Esquinas / elipse
        if isinstance(self.corners, np.ndarray):
            if self.name == 'square' or self.name == 'hexagon':
                for corner in self.corners:
                    # point
                    cv2.circle(frame, tuple(corner), 3, ColorBGR.BLACK, -1)

            if self.name == 'circle':
                print(self)     
                cv2.ellipse(frame, self.center.get_tuple_int(), (int(self.corners[0]/2), int(self.corners[1]/2)), int(self.corners[2]), 0, 360, (0, 255, 0), 2)

        return


    # Calculate Center and corners/ellipse
    def calculate_center_and_corners(self, frame: np.ndarray) -> bool:
        
        # 1. Ampliamos boundig box del objeto para prevenir el corte de esquinas ajustadas
        expand_pixels = 10
        bbox_modified = copy.deepcopy(self.bbox)
        bbox_modified.expand(expand_pixels)
        # 2. recortamos la imagen por la boundingbox
        crop_frame = hf.crop_frame_2(frame, corners=bbox_modified.get_array_int())
        # 3. Pasamos a escala de grises para la detección
        crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)

        # 4. deteccion. obtenemos corners
        if self.name == 'square':
            corners_crop = hf.detect_corners_harris(crop_frame)
            corners = []
            for corner_crop in corners_crop:
                corners.append(bbox_modified.p1.get_array_int() + corner_crop)
            # seleccionamos los 4 esquinas que estan mas arriba: cara superior
            self.corners = np.array(corners[:4]).astype(int)
            # el centroide de los 4 puntos superiores es el centro de la cara
            center = hf.calculate_centroid(self.corners).astype(int)
            self.center = Vector2D(center)
      
        elif self.name == 'hexagon':
            corners_crop = hf.detect_corners_harris(crop_frame)
            corners = []
            for corner_crop in corners_crop:
                corners.append(bbox_modified.p1.get_array_int() + corner_crop)
            # seleccionamos los 6 esquinas que estan mas arriba: cara superior
            self.corners = np.array(corners[:6]).astype(int)
            # el centroide de los 6 puntos superiores es el centro de la cara
            center = hf.calculate_centroid(self.corners).astype(int)
            self.center = Vector2D(center)

        elif self.name == 'circle':
            (xc, yc), (a, b), theta = hf.detect_ellipse(crop_frame)
            center = bbox_modified.p1.get_array() + np.array([xc, yc])
            self.center = Vector2D(center)
            self.corners = np.array([a, b, theta])

        else:
            return False
        
        return True

    # Calculate pose modo 1
    def calculatePose(self, ref: PieceA, t_ref_to_robot: np.ndarray = np.eye(4), verbose: bool = True, matplot_representation: bool = False) -> None:
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
        # print('Matriz de transicion de la pieza al robot: ', t_piece_to_robot)
        # print('Pose de la pieza respecto del robot: ', pose_robot)
        # input()
        self.point3d = Vector3D(ppiece_robot[0], ppiece_robot[1], ppiece_robot[2]) # =  Vector6D(pose_robot[0], pose_robot[1], pose_robot[2])

        self.pose = Vector6D(pose_robot[0], pose_robot[1], pose_robot[2], pose_robot[3], pose_robot[4], pose_robot[5])

        if verbose:
            print()
            print('Pieza seleccionada: ', self.name)
            print('Pose: ', self.pose)

        # 3D representation
        if matplot_representation:
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
    
    # Calculate pose modo 2
    def calculatePose_v2(self, pointcloud, ref: PieceA, t_ref_to_robot: np.ndarray = np.eye(4),verbose: bool = True, matplot_representation: bool = False):
        if not pointcloud:
            print('Not pointcloud')
            return 
        
        
        # referencia respecto de la camara -----------------------------------------------------------------------------
        t_ref_to_cam = ref.T
        pref_cam = hf.point_tansf(t_ref_to_cam, np.array([0 ,0, 0]))
        # en m
        pref_cam_pointcloud = pref_cam.copy()
        # hacemos espejo por culpa del sistemade ref
        rot = hf.rotation_matrix_z(180)
        pref_cam_pointcloud = hf.point_tansf(rot, pref_cam_pointcloud)
        pref_cam_pointcloud *= 1000 # se pasa a mm

        # pieza respecto de la camara: nos falta esta t_piece_to_cam ----------------------------------------------------------------------------------
        # 1. punto en 3d respecto de la camara
        ppiece_cam_pointcloud= hf.pixel_to_point3d(pointcloud, resolution=np.array([1920, 1080]), pixel=self.center.get_array())
        ppiece_cam = ppiece_cam_pointcloud.copy()
        # 2. rotamoce 180 respecto de z para ajustar sistema de referencia
        rot = hf.rotation_matrix_z(180)
        ppiece_cam = hf.point_tansf(rot, ppiece_cam)
        ppiece_cam /= 1000 # en m

        # print('centro de la pieza: ', ppiece_cam)
        # print('centro del apriltag: ', pref_cam)

        # rotacion de la pieza respecto al de ref --------------------------------------
        p1 = ref.corners[0]
        p2 = ref.corners[1]
        xRef = p2 - p1
        print(xRef)
        print(self.corners[0])
        p11 = Vector2D(np.array(self.corners[0]))
        print(p11)
        p22 = Vector2D(np.array(self.corners[1]))

        xPiece = p22 - p11

        degrees = hf.angle_between_vectors(xRef.get_array(), xPiece.get_array())


        # degrees = 45
        rot_piece_to_ref = hf.rotation_matrix_z(45)# degrees)
        rot_ref_to_cam = t_ref_to_cam[:3,:3].copy()
        
        rot_piece_to_cam = np.dot(rot_ref_to_cam, rot_piece_to_ref)
        
        
        t_piece_to_cam = hf.transformation_matrix(rot_piece_to_cam, ppiece_cam)
        print('pieza a camara: ',t_piece_to_cam)
        print('ref a camara: ',t_ref_to_cam)
        
        # --------------------------------------------------------------------------------------------------
        # 2 metodo (1 en la v1)
        t_piece_to_ref = np.dot(np.linalg.inv(t_ref_to_cam), t_piece_to_cam)
        t_piece_to_robot = np.dot(t_ref_to_robot,t_piece_to_ref)
        t_cam_to_robot = np.dot(t_ref_to_robot, np.linalg.inv(t_ref_to_cam))

        ppiece_robot = hf.point_tansf(t_piece_to_robot, np.array([0,0,0]))
        pcam_robot = hf.point_tansf(t_cam_to_robot, np.array([0,0,0]))
        pref_robot = hf.point_tansf(t_ref_to_robot, np.array([0,0,0]))
        probot_robot = np.array([0,0,0])


        pose_robot = hf.pose_transf(t_piece_to_robot, np.array([0,0,0]))
        # print('Matriz de transicion de la pieza al robot: ', t_piece_to_robot)
        # print('Pose de la pieza respecto del robot: ', pose_robot)
        # input()
        self.point3d = Vector3D(ppiece_robot[0], ppiece_robot[1], ppiece_robot[2]) # =  Vector6D(pose_robot[0], pose_robot[1], pose_robot[2])

        self.pose = Vector6D(pose_robot[0], pose_robot[1], pose_robot[2], pose_robot[3], pose_robot[4], pose_robot[5])

        if verbose:
            print()
            print('Pieza seleccionada: ', self.name)
            print('Pose: ', self.pose)

        # 3D representation --------------------------------------------------
        if matplot_representation:
            size = 0.2
            robot_axes = np.array([[size, 0, 0],
                                [0, size, 0],
                                [0, 0, size]])
            ref_axes = np.dot(t_ref_to_robot[:3, :3], robot_axes.T).T
            piece_axes = np.dot(t_piece_to_robot[:3, :3], robot_axes.T).T
            cam_axes = np.dot(t_cam_to_robot[:3, :3], robot_axes.T).T

            fig, ax = hf.init_mat3d()
            hf.add_point_with_axes(ax, probot_robot, robot_axes, 'robot', 'k')
            hf.add_point_with_axes(ax, pref_robot, ref_axes, 'ref', 'r')
            hf.add_point_with_axes(ax, pcam_robot, cam_axes, 'camera', 'b')
            hf.add_point_with_axes(ax, ppiece_robot, piece_axes, 'piece', 'g')

            hf.show_mat3d(fig, ax, 'apriltags representation')


        # REPRESENTACION CON OPEN3D--------------------------------------------------------------------------
        ppcloud = hf.pixel_to_point3d(pointcloud, resolution=np.array([1920, 1080]), pixel=ref.center.get_array())

        cube = hf.create_cube(point=ppcloud, size = [5,5,5])
        cube2 = hf.create_cube(point=pref_cam_pointcloud, size = [5,5,5], color = np.array([0,0,1]))
        axes = hf.create_axes_with_lineSet()

        print('apriltag lib. center: ', pref_cam_pointcloud)
        print('apriltag pointcloud cent: ', ppcloud)
        hf.o3d_visualization([pointcloud, cube, cube2, axes])

        return

    # Calculate pose modo 3
    def calculatePose_v3(self, pointcloud, ref: PieceA, t_ref_to_robot: np.ndarray = np.eye(4), resolution: Vector2D = Vector2D(1920, 1080),verbose: bool = True, matplot_representation: bool = False):
        if not pointcloud:
            print('Not pointcloud')
            return 
        
        # PIXEL to CLOUD -------------------------------------------------------------------------------------
        pref_cloud = hf.pixel_to_point3d(pointcloud, resolution=resolution.get_array_int(), pixel=ref.center.get_array_int())
        print('center: ',self.center.get_array_int())
        ppiece_cloud = hf.pixel_to_point3d(pointcloud, resolution=resolution.get_array_int(), pixel=self.center.get_array_int())

        # MATRIZ TRANSFORMACION APRIL ----------------------------------------------------------------------
        t_ref_to_cam = ref.T
        pref_good = hf.point_tansf(t_ref_to_cam, np.array([0 ,0, 0])) # en m
        pref_good *= 1000 # se pasa a mm

        ppiece_good_mm = hf.point_tansf(T=CameraCte.T_pointcloud_to_good_pointcloud_2, point=ppiece_cloud)
        ppiece_good_m = ppiece_good_mm/1000

        # PASO AL SISTEMA DE REF DEL ROBOT ----------------------------------------------------------------
        # rotacion de la pieza en comparcacion con la referencia
        if self.name == 'circle':
            angle = 0
        else:
            ref_x = ref.corners[1] - ref.corners[0]
            ref_x = ref_x.get_array()
            piece_x = self.corners[1] - self.corners[0]
            angle = hf.angle_between_vectors(ref_x, piece_x)
        print('Angulo: ', angle)
        
        rot_piece_to_ref = hf.rotation_matrix_z(angle)
        rot_ref_to_cam = t_ref_to_cam[:3,:3]
        rot_piece_to_cam = np.dot(rot_ref_to_cam, rot_piece_to_ref)

        t_piece_to_cam = hf.transformation_matrix(rot_piece_to_cam, ppiece_good_m)
        t_piece_to_ref = np.dot(np.linalg.inv(t_ref_to_cam), t_piece_to_cam)
        t_piece_to_robot = np.dot(t_ref_to_robot,t_piece_to_ref)

        ppiece_robot = hf.point_tansf(t_piece_to_robot, np.array([0,0,0]))
        pose_robot = hf.pose_transf(t_piece_to_robot, np.array([0,0,0]))

        # pose_robot[3:] = [0.028,0,3.318]

        self.point3d = Vector3D(ppiece_robot)
        self.pose = Vector6D(pose_robot)


        # REPRESENTACION CON OPEN3D--------------------------------------------------------------------------

        cube = hf.create_cube(point=pref_cloud, size = [5,5,5], color = np.array([0,0,1]))
        cube2 = hf.create_cube(point=pref_good, size = [5,5,5], color = np.array([0,1,0]))

        cube11 = hf.create_cube(point=ppiece_cloud, size = [5,5,5], color = np.array([0,0,1]))
        cube22 = hf.create_cube(point=ppiece_good_mm, size = [5,5,5], color = np.array([0,1,0]))

        axes = hf.create_axes_with_lineSet()

        hf.o3d_visualization([pointcloud, cube, cube2,cube11, cube22, axes])

        # --------------------------------------------------------------------------------------------------

        return


# -------------------- TRAINNING ----------------------------------------------------------------------------------------- #
