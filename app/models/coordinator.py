"""
        coordinator.py

    Coordinador 

"""

# -------------------- PACKAGES ------------------------------------------------------------------------------------------ #

import cv2
import numpy as np

from functions import helper_functions as hf

from models.camera import CameraConfig, Camera
from models.vectors import Vector2D, Vector6D
from models.constants import RobotCte, ColorBGR
from models.piece import BoundingBox, PieceA, PieceN, PieceN2, Piece
from models.robot import Robot
from models.detection import DetectorInterface, Apriltag, YoloObjectDetection, YoloPoseEstimation

# -------------------- VARIABLES ----------------------------------------------------------------------------------------- #



# -------------------- CLASSES ------------------------------------------------------------------------------------------- #


class Coordinator():

    """ ----- COMBINAR PIEZAS ----- """

    @staticmethod
    def combined_pieces(piecesA: list[PieceA]|None, piecesN: list[PieceN]|None, piecesN2: list[PieceN2]|None, combine_pieces: bool = True) -> tuple[bool, PieceA|None|bool, list[Piece]]:
        """combinar detecciones en una sola"""
        flag = False
        ref = None
        pieces = []

        # No combinar piezas
        if combine_pieces == False:
            # 1. adquirimos referencia
            if piecesA != None:
                for pieceA in piecesA:
                    if pieceA.name == '4':
                        # 1. apriltag de ref
                        ref = pieceA
            # 2. adquirimos piezas de piecesN
            if piecesN != None:
                for pieceN in piecesN:
                    pieces.append(Piece(pieceN))
            # 3. adquirimos piezas de pieceN2
            if piecesN2 != None:
                for pieceN2 in piecesN2:
                    pieces.append(Piece(pieceN2))

            # 4. # condicion de bandera
            if (pieces != []) and (ref != None):
                flag = True
            print('Flag: ', flag)
            return flag, ref, pieces


        # Combinar piezas
        if piecesA != None:
            # Modo 1
            if piecesN != None: 
                # print('MODO 1 ----------------------------')
                for pieceA in piecesA:
                    if pieceA.name == '4':
                        # 1. apriltag de ref
                        ref = pieceA
                    else:
                        for pieceN in piecesN:
                            try:
                                piece = Piece(pieceA, pieceN)
                            except:
                                pass
                            else:
                                # 2. piezas con aprils incluidos
                                pieces.append(piece)

                                

            # Modo 2 
            elif piecesN2 != None:
                # print('MODO 2 ----------------------------')
                # 1. apriltag de ref
                for pieceA in piecesA:
                    if pieceA.name == '4':
                        # 1. apriltag de ref
                        ref = pieceA
                    # QUITAR
                    ref = pieceA
                # 2. piezas geretales
                for pieceN2 in piecesN2:
                    pieces.append(Piece(pieceN2))


        else: # Modo 3 -> sin ref april
            # print('MODO 3 ----------------------------')
            # 2. piezas 
            for pieceN2 in piecesN2:
                pieces.append(Piece(pieceN2))
            ref = False
        
        # condicion de bandera
        if (pieces != []) and (ref != None):
            flag = True
        return flag, ref, pieces


    """ ----- DETECCIONES ----- """

    # Detecciones apriltags
    @staticmethod
    def apriltag_detections(frame, camera: Camera, apriltag: Apriltag, paint_frame: bool = True) -> dict:
        # 1. Camera params
        # camera_params = [camera.f.x, camera.f.y, camera.c.x, camera.c.y]
        # 2. deteccion
        apriltag.detect(frame)
        # 3. verificacion y paint
        if apriltag.pieces:
            if paint_frame:
                apriltag.paint(frame)
            flag = True
        else:
            flag = False
        return {'flag': flag, 'pieces': apriltag.pieces}
    
    # Detecciones Yolo object detection
    @staticmethod
    def nn_object_detections(frame, camera: Camera, nn_model: YoloObjectDetection, paint_frame: bool = True) -> dict:
        # 1. deteccion
        nn_model.detect(frame)
        # 2. verificacion y paint
        if nn_model.pieces:
            if paint_frame:
                nn_model.paint(frame)
            flag = True
        else:
            flag = False
        return {'flag': flag, 'pieces': nn_model.pieces}

    # Detecciones Yolo pose estimation
    @staticmethod
    def nn_poseEstimation_detections(frame, camera: Camera, nn_model: YoloPoseEstimation, paint_frame: bool = True) -> dict:
        nn_model.detect(frame)
        # 2. verificacion y paint
        if nn_model.pieces:
            if paint_frame:
                nn_model.paint(frame)
            flag = True
        else:
            flag = False
        return {'flag': flag, 'pieces': nn_model.pieces}
    
    # Detecciones con polimorfismo ------- SIN APLICAR
    @staticmethod
    def general_detections(frame: np.ndarray, camera: Camera, detector: DetectorInterface, paint_frame: bool = True) -> dict:
        # 1. deteccion
        detector.detect(frame)
        # 2. verificacion y paint
        if detector.pieces:
            if paint_frame:
                detector.paint(frame)
            flag = True
        else:
            flag = False
        return {'flag': flag, 'pieces': detector.pieces}

    # Detecciones Totales 
    @staticmethod
    def detections(frame: np.ndarray, camera: Camera, nn_model: YoloObjectDetection|YoloPoseEstimation, apriltag: Apriltag|None = None, combine_pieces: bool = True, paint_frame: bool = True) -> dict:
        at_kwargs = {'pieces': None}
        od_kwargs = {'pieces': None}
        pose_kwargs = {'pieces': None}

        if apriltag:
            at_kwargs = Coordinator.apriltag_detections(frame, camera, apriltag, paint_frame=False)
        if type(nn_model) == YoloObjectDetection:

            od_kwargs = Coordinator.nn_object_detections(frame, camera, nn_model, paint_frame=False)

        elif type(nn_model) == YoloPoseEstimation:
            pose_kwargs = Coordinator.nn_poseEstimation_detections(frame, camera, nn_model, paint_frame=False)

        flag, ref, pieces = Coordinator.combined_pieces(at_kwargs['pieces'], od_kwargs['pieces'], pose_kwargs['pieces'], combine_pieces)

        if paint_frame:
            if ref: # si la ref es False nos encotramos en el modo 4 donde no hay ref (conocemos la pos de la camara)
                ref.paint(frame)
            for piece in pieces:
                piece.paint(frame)
        
        return {'flag': flag, 'ref': ref,'pieces': pieces}


    """ ----- MOVIMIENTOS ----- """

    @staticmethod
    def combinated_movement(robot: Robot, piece: Piece, tolerance: int = 15) -> None:

        # 1. posicion de la pieza
        if isinstance(piece.pose, Vector6D):
            pose = piece.pose.get_array()
        else:
            print('No se conoce la pose de la pieza')
            return

        # 2. posicion del hoyo
        hole_pose = RobotCte.get_hole_pose_by_name(piece.name, tolerance=tolerance)
        # VERIFICAR QUE SE PONE ASI
        if not isinstance(hole_pose, np.ndarray):
            print('No se ha encontrado el hoyo')
            return
        print()
        print('Pose Pieza: ', pose)
        print('Hoyo: ', hole_pose)
        print()
        
        # 1. posicion segura
        robot.move(RobotCte.POSE_STANDAR)
        robot.move(RobotCte.POSE_SAFE_APRILTAG_REF)

        secure_pose = pose.copy()
        secure_pose[2] = RobotCte.SAFE_Z
        robot.move(secure_pose)
        # input(f'en posicion segura: {secure_pose}')
        # 2. encima de la pieza + rotation + gripper off
        robot.gripper_control(False)
        # 3. bajar para coger la pieza
        take_pose = pose.copy()
        take_pose[2] = RobotCte.TAKE_PIECE_Z
        robot.move(take_pose)
        # 4. gripper on
        robot.gripper_control(True)
        # 5. levantar hasta posicion segura (no es la misma que la 2. debe estar al doble de altura para que no choque con ninguna pieza al desplazarla)
        secure_pose[2] = RobotCte.SAFE_Z_2
        robot.move(secure_pose)
        # 6. ir a la posicion del hoyo ( un poco arriba)
        secure_hole_pose = hole_pose.copy()
        secure_hole_pose[2] = RobotCte.SAFE_Z_2
        robot.move(secure_hole_pose)
        a_hole_pose = hole_pose.copy()
        # 7. posicion del hoyo exacta + soltar gripper
        a_hole_pose[2] = RobotCte.TAKE_PIECE_Z
        robot.move(a_hole_pose)
        robot.gripper_control(False)
        # 8. posicion segura un poco mas arriba
        robot.move(secure_hole_pose)
        # 9. posicion reposo para visualizar piezas
        robot.move(RobotCte.POSE_STANDAR)
        return
    

    """ ----- PRINCIPAL -----"""

    @staticmethod
    def the_whole_process(robot: Robot, camera: Camera, apriltag: Apriltag, nn_od_model: YoloObjectDetection, tolerance: int = 15) -> None:
        print()
        # 1. Movemos robot a la posicion de visualizacion de las piezas
        try:
            print('Movimientos iniciales:')
            robot.gripper_control(True)
            robot.move(RobotCte.POSE_DISPLAY)
        except Exception as e:
            print(str(e))
            return
        
        # 2. detecciones
        print()
        print('Inicio de detecciones:')
        try:
            r_kwargs = camera.run_with_condition(Coordinator.detections, nn_model = nn_od_model, apriltag=apriltag, combine_pieces = True, paint_frame = True)
            if not r_kwargs:
                print()
                print('Salida voluntaria desde camara')
                return
        except:
            print()
            print('Salida desde camara')
            return
        
        ref: PieceA = r_kwargs['ref']
        pieces: list[Piece] = r_kwargs['pieces']
        frame: np.ndarray = r_kwargs['frame']

        ref.paint(frame)
        for piece in pieces:
            piece.paint(frame)

        cv2.imshow('Detecciones',cv2.resize(frame, (1280, 720)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 3. calcular pose
        # 3.1 Se elige la primera pieza para continuar el proceso
        piece = pieces[0]
        # 3.2 Calculo de la pose de la pieza respecto al sistema de referencia de la base del robot
        piece.calculatePose(ref, RobotCte.T_REF_TO_ROBOT, verbose=True, matplot_representation=False)
        
        # 4. movimiento combinado: coger la pieza y dejarla en su respectivo hoyo (posicion conocida)
        try:
            print()
            print('Inicio de movimientos combinados:')
            Coordinator.combinated_movement(robot, piece, tolerance=tolerance)
        except Exception as e:
            print(str(e))
            return False

        return True
    
    @staticmethod
    def the_whole_process_2(robot: Robot, camera: Camera, apriltag: Apriltag, nn_pose_model: YoloPoseEstimation) -> None:
        print()
        # 1. Movemos robot a la posicion de visualizacion de las piezas
        try:
            print('Movimientos iniciales:')
            robot.gripper_control(True)
            robot.move(RobotCte.POSE_SAFE_APRILTAG_REF)
            robot.move(RobotCte.POSE_DISPLAY_2)
        except Exception as e:
            print(str(e))
            return
        
        # 2. detecciones
        print()
        print('Inicio de detecciones:')
        try:
            r_kwargs = camera.run_with_pointcloud_with_condition(show3d=False, trigger_func=Coordinator.detections, nn_model=nn_pose_model, apriltag=apriltag, paint_frame=True)
        except Exception as e:
            print()
            print('salida de camara: ',str(e))
            return
        
        ref = r_kwargs['ref']
        pieces = r_kwargs['pieces']
        frame = r_kwargs['frame']
        pointcloud = r_kwargs['pointcloud']

        cv2.imshow('Detecciones',cv2.resize(frame, (1280, 720)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 3. calcular pose
        # 3.1 Se elige la primera pieza para continuar el proceso
        piece: Piece = pieces[0]
        # 3.2 Calculo de la pose de la pieza respecto al sistema de referencia de la base del robot
        # 3.2 Calculo de la pose de la pieza respecto al sistema de referencia de la base del robot
        print(piece)
        piece.calculatePose_v2(pointcloud, ref,t_ref_to_robot=RobotCte.T_REF_TO_ROBOT,  matplot_representation=False)

        # 4. movimiento combinado: coger la pieza y dejarla en su respectivo hoyo (posicion conocida)
        try:
            print()
            print('Inicio de movimientos combinados:')
            Coordinator.combinated_movement(robot, piece)
        except Exception as e:
            print(str(e))
            return False

        return True

    @staticmethod
    def the_whole_process_3(robot: Robot, camera: Camera, apriltag: Apriltag, nn_od_model: YoloObjectDetection) -> None:
        print()
        # 1. Movemos robot a la posicion de visualizacion de las piezas
        try:
            print('Movimientos iniciales:')
            robot.move(RobotCte.POSE_STANDAR)
            robot.move(RobotCte.POSE_DISPLAY_2)
        except Exception as e:
            print(str(e))
            return
        
        # 2. detecciones
        print()
        print('Inicio de detecciones:')
        try:
            r_kwargs = camera.run_with_pointcloud_with_condition(show3d=False, trigger_func=Coordinator.detections, nn_model=nn_od_model, apriltag=apriltag, combine_pieces = False, paint_frame=True)
        except Exception as e:
            print()
            print('salida de camara: ',str(e))
            return
        
        ref = r_kwargs['ref']
        pieces = r_kwargs['pieces']
        frame = r_kwargs['frame']
        pointcloud = r_kwargs['pointcloud']

    
        # 3. calcular pose
        # 3.1 Se elige la primera pieza para continuar el proceso ------------------------------------- 
        piece: Piece = pieces[0]
        # 3.2 Calcular centro con Vision artificial clasica -------------------------------------------
        piece.calculate_center_and_corners(frame)
        piece.paint(frame)
        # añadir criculo estrella e irregular
        # pintar mejor
        # revisar funciones de corners y get arrays Vectors, ect
        # ----------------------------------------------------------------------------------------------

        cv2.imshow('Detecciones',cv2.resize(frame, (1280, 720)))
        # cv2.imshow('Detecciones', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 3.3 Calculo de la pose de la pieza respecto al sistema de referencia de la base del robot teniendo el centro en la imagen
        piece.calculatePose_v3(pointcloud, ref,t_ref_to_robot=RobotCte.T_REF_TO_ROBOT,  matplot_representation=False)
        print(piece)
        # 4. movimiento combinado: coger la pieza y dejarla en su respectivo hoyo (posicion conocida)
        try:
            print()
            print('Inicio de movimientos combinados:')
            Coordinator.combinated_movement(robot, piece)
        except Exception as e:
            print(str(e))
            return False

        return True