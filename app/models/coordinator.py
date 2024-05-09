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


    @staticmethod
    def apriltag_detections(frame, camera: Camera, apriltag: Apriltag):
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
    def combined_detections(piecesA: list[PieceA], piecesN: list[PieceN]) -> tuple[PieceA|None, list[Piece]]:
        """combinar detecciones en una sola"""
        pieces = []
        ref = None
        if not piecesA:
            print('No aprils')
            return ref, pieces
        for pieceA in piecesA:
            if pieceA.name == '4':
                # 1. apriltag de ref
                ref = pieceA
            else:
                for pieceN in piecesN:
                    piece = Piece(pieceA, pieceN)
                    if piece.validate():
                        # 2. piezas con aprils incluidos
                        pieces.append(piece)
        return ref, pieces

    @staticmethod
    def combinated_movement(robot: Robot, piece: Piece) -> None:
        # 1. posicion de la pieza
        if piece.pose:
            pose = piece.pose
        else:
            # Exception?
            return
        # posicion del hoyo
        name = piece.name
        hole_pose = Coordinator.get_hole_pose_by_name(piece.name)
        # VERIFICAR QUE SE PONE ASI
        if not hole_pose:
            return
        
        # 1. posicion segura
        
        # 2. encima de la pieza + rotation + gripper off
        robot.gripper_control(False)

        # 3. bajar para coger la pieza
        robot.move(pose)
        # 4. gripper on
        robot.gripper_control(True)
        # 5. levantar hasta posicion segura (no es la misma que la 2. debe estar al doble de altura para que no choque con ninguna pieza al desplazarla)

        # 6. ir a la posicion del hoyo ( un poco arriba)

        # 7. posicion del hoyo exacta + soltar gripper
        robot.move()
        robot.gripper_control(False)

        # 8. posicion segura un poco mas arriba

        # 9. posicion reposo para visualizar piezas


        return
    
    @staticmethod
    def the_whole_process(robot: Robot, camera: Camera, apriltag: Apriltag, nn_od_model: YoloObjectDetection) -> None:
        # 1. Movemos robot a la posicion de visualizacion de las 
        
        robot.move(RobotConstants.POSE_APRILTAG_REF)
        robot.move(RobotConstants.POSE_DISPLAY)
        
        print('inicio paso 2')
        # 2. detecciones
        while True:
            camera.init_rgb()
            # 2.1 Deteccion de la pieza con red neuronal (object detection)
            frame, piecesN = camera.run_with_condition(Coordinator.nn_object_detections, nn_od_model)
            # 2.2 Deteccion de los apriltags de la imagen en la que se ha encontrado la pieza. Es necesario que se encuentre el apriltag de referencia
            frame, boolean , piecesA = Coordinator.apriltag_detections(frame, camera, apriltag)

            # 2.3. union de las detecciones para sacar la referencia y las piezas
            ref, pieces = Coordinator.combined_detections(piecesA, piecesN)
    
            # 2.4. si no ha encontrado la ref y al menos una pieza vuelve a repetir el proceso
            if (ref != None) and (pieces != []):
                print('Referencia: ', ref)
                for piece in pieces:
                    print('pieza final: ',piece)
                break

        print('inicio paso 4')
        robot.move(RobotConstants.POSE_DISPLAY)

        # 3. Imagen con las piezas y la referencia dibujadas

        ref.paint(frame)
        for piece in pieces:
            # print(piece)
            piece.paint(frame)

        cv2.imshow('a',frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # 4 ubicar centro del april de las piezas como punto 3d respecto a la base del robot (matrices de transferencia). Importante la rotacion de la pieza
        # 4.1 Se elige la primera pieza para continuar el proceso
        piece = pieces[0]
        # 4.2 Calculo de la pose de la pieza respecto al sistema de referencia de la base del robot
        piece.calculatePose(ref, RobotConstants.T_REF_TO_ROBOT)
        new_pose = piece.pose.get_array()[:3]
        new_pose = np.append(new_pose, RobotConstants.POSE_APRILTAG_REF[-3:])
        print('new pose: ', new_pose)
        input()
        robot.move( new_pose)

        # 4. Movimiento del robot para coger la pieza y dejarla en su respectivo hoyo (posicion conocida)
        # Coordinator.combinated_movement(robot, piece)

        return

