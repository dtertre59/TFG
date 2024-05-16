"""
        coordinator.py

    Coordinador 

"""

# -------------------- PACKAGES ------------------------------------------------------------------------------------------ #

import cv2
import numpy as np

from models.camera import CameraConfig, Camera
from models.vectors import Vector2D
from models.piece import BoundingBox, PieceA, PieceN, Piece
from models.robot import Robot, RobotConstants
from models.detection import Apriltag, YoloObjectDetection, YoloPoseEstimation

# -------------------- VARIABLES ----------------------------------------------------------------------------------------- #

# objects_colors = {
#         'circle': (0,0,255),
#         'hexagon': (0,255,0),
#         'scuare': (255,0,0) # va con q pero hay que cambiarlo en la red neuronal
#     }


# -------------------- CLASSES ------------------------------------------------------------------------------------------- #


class Coordinator():

    @staticmethod
    def get_hole_pose_by_name(name: str) -> np.ndarray|None:
        if name == 'square':
            return RobotConstants.POSE_PUZZLE_SQUARE_20
        elif name == 'circle':
            return RobotConstants.POSE_PUZZLE_CIRCLE_20
        elif name == 'hexagon':
            return RobotConstants.POSE_PUZZLE_HEXAGON_20
        else:
            return

    """ ----- DETECCIONES ----- """
    @staticmethod
    def apriltag_detections(frame, camera: Camera, apriltag: Apriltag) -> tuple[np.ndarray, bool, list[PieceA]]:
        # 1. Camera params
        camera_params = [camera.f.x, camera.f.y, camera.c.x, camera.c.y]
        # 2. deteccion
        apriltag.detect(frame, camera_params)
        # 3. verificacion y paint
        if apriltag.pieces:
            # apriltag.paint(frame)
            return frame, True, apriltag.pieces
        else:
            return frame, False, apriltag.pieces
  
    @staticmethod
    def nn_object_detections(frame, camera: Camera, nn_model: YoloObjectDetection):
        # 1. deteccion
        nn_model.detect(frame)
        # 2. verificacion y paint
        if nn_model.pieces:
            # nn_model.paint(frame)
            return frame, True, nn_model.pieces
        else:
            return frame, False, nn_model.pieces

    @staticmethod
    def nn_poseEstimation_detections(frame, camera: Camera, nn_model: YoloPoseEstimation):
        pass

    @staticmethod
    def combined_pieces_detections(piecesA: list[PieceA], piecesN: list[PieceN]) -> tuple[bool, PieceA|None, list[Piece]]:
        """combinar detecciones en una sola"""
        flag = False
        ref = None
        pieces = []
        
        if not piecesA:
            print('No se ha detectado ningun apriltag')
            return flag, ref, pieces
        for pieceA in piecesA:
            if pieceA.name == '4':
                # 1. apriltag de ref
                ref = pieceA
            else:
                for pieceN in piecesN:
                    piece = Piece(pieceA, pieceN)
                    if piece.validate(): # se valida que el centro de la cara de la pieza se envcuentre dentro de la boundig box
                        # 2. piezas con aprils incluidos
                        pieces.append(piece)
        
        if (ref == None) or (pieces == []):
            return flag, ref, pieces
        flag = True
        return flag, ref, pieces

    @staticmethod
    def detections(frame: np.ndarray, camera: Camera, apriltag: Apriltag, nn_model: YoloObjectDetection|YoloPoseEstimation, paint_frame: bool = True) -> tuple[bool, PieceA|None, list[Piece]]:
        frame, flagA, piecesA = Coordinator.apriltag_detections(frame, camera, apriltag)
        if type(nn_model) == YoloObjectDetection:
            frame, flagN, piecesN = Coordinator.nn_object_detections(frame, camera, nn_model)
        elif type(nn_model) == YoloPoseEstimation:
            print('método no completado todavia')
            return
        if paint_frame:
            for pieceA in piecesA:
                pieceA.paint(frame)
            for pieceN in piecesN:
                pieceN.paint(frame)
        flag, ref, pieces = Coordinator.combined_pieces_detections(piecesA, piecesN)
        
        return flag, ref, pieces

    """ ----- MOVIMIENTOS ----- """
    @staticmethod
    def combinated_movement(robot: Robot, piece: Piece) -> None:
        # 1. posicion de la pieza
        if piece.pose:
            pose = piece.pose.get_array()
        else:
            # Exception?
            return

        # # posicion del hoyo
        # name = piece.name
        # hole_pose = Coordinator.get_hole_pose_by_name(piece.name)
        # # VERIFICAR QUE SE PONE ASI
        # if not hole_pose:
        #     return
        
        # 1. posicion segura
    
        robot.move(RobotConstants.POSE_SAFE_APRILTAG_REF)
        secure_pose = pose
        secure_pose[2] = RobotConstants.SAFE_Z
        robot.move(secure_pose)
        # 2. encima de la pieza + rotation + gripper off
        robot.gripper_control(False)
        # 3. bajar para coger la pieza
        take_pose = pose
        take_pose[2] = RobotConstants.TAKE_PIECE_Z
        robot.move(pose)
        # 4. gripper on
        robot.gripper_control(True)
        # 5. levantar hasta posicion segura (no es la misma que la 2. debe estar al doble de altura para que no choque con ninguna pieza al desplazarla)
        secure_pose[2] = RobotConstants.SAFE_Z_2
        robot.move(secure_pose)
        # 6. ir a la posicion del hoyo ( un poco arriba)
        # 7. posicion del hoyo exacta + soltar gripper
        # robot.move()
        robot.gripper_control(False)

        # 8. posicion segura un poco mas arriba

        # 9. posicion reposo para visualizar piezas


        return
    
    """ ----- PRINCIPAL -----"""
    @staticmethod
    def the_whole_process(robot: Robot, camera: Camera, apriltag: Apriltag, nn_od_model: YoloObjectDetection) -> None:
        print()
        # 1. Movemos robot a la posicion de visualizacion de las piezas
        try:
            print('Movimientos iniciales:')
            robot.gripper_control(True)
            robot.move(RobotConstants.POSE_SAFE_APRILTAG_REF)
            input('aaaaaaaaaa')
            robot.move(RobotConstants.POSE_DISPLAY)
        except Exception as e:
            print(str(e))
            return
        
        # 2. detecciones
        print()
        print('Inicio de detecciones:')
        try:
            frame, ref, pieces = camera.run_with_condition(Coordinator.detections, apriltag, nn_od_model, paint_frame = True)
        except:
            print()
            print('Salida desde camara')
            return
        
        cv2.imshow('Detecciones',cv2.resize(frame, (1280, 720)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 3. calcular pose
        # 3.1 Se elige la primera pieza para continuar el proceso
        piece = pieces[0]
        # 3.2 Calculo de la pose de la pieza respecto al sistema de referencia de la base del robot
        piece.calculatePose(ref, RobotConstants.T_REF_TO_ROBOT, verbose=True, matplot_representation=False)
        
        # 4. movimiento combinado: coger la pieza y dejarla en su respectivo hoyo (posicion conocida)
        try:
            print()
            print('Inicio de movimientos combinados:')
            Coordinator.combinated_movement(robot, piece)
        except Exception as e:
            print(str(e))
            return False

        return True

