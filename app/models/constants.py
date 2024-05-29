"""
        constants.py
    Todas las constantes utilizadas en el programa
    Agrupadas en clases
"""

# -------------------- PACKAGES ------------------------------------------------------------------------------------------ #

import time
import logging
import sys
import numpy as np


# -------------------- VARIABLES ----------------------------------------------------------------------------------------- #


class ColorBGR():

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)

    @staticmethod
    def get_piece_color(name: str) -> tuple:
        if name == 'square':
            color = ColorBGR.RED
        elif name == 'circle':
            color = ColorBGR.GREEN
        elif name == 'hexagon':
            color = ColorBGR.BLUE
        else:
            color = ColorBGR.BLACK
        return color


class CameraCte():
    T_pointcloud_to_good_pointcloud = np.array([[    0.99997,   0.0037492,   -0.007014,      11.377],
                                                [ -0.0034771,     0.99926,    0.038411,     -18.397],
                                                [  0.0071528,   -0.038386,     0.99924,     -18.176],
                                                [          0,           0,           0,           1]])
    
    T_pointcloud_to_good_pointcloud_2 = np.array([[     0.9997,   -0.004218,   -0.024238,       17.05],
                                                  [  0.0038985,     0.99991,   -0.013214,      6.5171],
                                                  [   0.024292,    0.013115,     0.99962,     -15.504],
                                                  [          0,           0,           0,           1]])
    
    T_pointcloud_to_good_pointcloud_3 = np.array([[    0.99942,  -0.0052731,   -0.033546,      24.543],
                                                  [  0.0059128,      0.9998,    0.018999,     -12.003],
                                                  [    0.03344,   -0.019186,     0.99926,     -13.607],
                                                  [          0,           0,           0,           1]])


class RobotCte():
   

    POSE_STANDAR = np.array([-0.128, -0.298, 0.180, 0.025, 0, 2.879])
    POSE_STANDAR_2 = np.array([-0.128, -0.298, 0.180, 0.015, 0, 1.501])
    # Posicion para la visualizacion de las piezas
    POSE_DISPLAY = np.array([-0.125, -0.166, 0.170, 1.454, -1.401, -4.095])
    POSE_DISPLAY_2 = np.array([-0.130, -0.126, 0.313, 1.540, -1.323, -4.294])

    # Posicion segura para el movimiento
    SAFE_Z = 0.075
    SAFE_Z_2 = 0.150

    TAKE_PIECE_Z = 0.040

    # POSE_SAFE = np.array([0.128, -0.298, 0.180, 3.1415, 0.2617, 0])
    # POSE_SAFE = np.array([-0.128, -0.298, 0.180, 0.025, 0, 2.879])


     # Posicion del a base del robot
    POSE_ROBOT_BASE = np.array([0, 0, 0])
    # Posicion del apriltag de referencia
    POSE_APRILTAG_REF = np.array([-0.016, -0.320, 0.003, 0.028, 0, 3.318])
    POSE_SAFE_APRILTAG_REF = np.array([-0.016, -0.320, SAFE_Z, 0.028, 0, 3.318])


    # Matriz de transicion del sistema de referencia del apriltag(ref) al de la base del robot
    T_REF_TO_ROBOT = np.array([[1, 0, 0, POSE_APRILTAG_REF[0]],
                                [0, 1, 0, POSE_APRILTAG_REF[1]],
                                [0, 0, 1, POSE_APRILTAG_REF[2]],
                                [0, 0, 0, 1]])


    # POSICIONES DE LOS HOYOS
    # 5%
    POSE_PUZZLE_CIRCLE_5 = np.ndarray([1,1,1,1,1,1])
    POSE_PUZZLE_SQUARE_5 = np.ndarray([1,1,1,1,1,1])
    POSE_PUZZLE_HEXAGON_5 = np.ndarray([1,1,1,1,1,1])
    # 10%
    POSE_PUZZLE_CIRCLE_10 = np.ndarray([1,1,1,1,1,1])
    POSE_PUZZLE_SQUARE_10 = np.ndarray([1,1,1,1,1,1])
    POSE_PUZZLE_HEXAGON_10 = np.ndarray([1,1,1,1,1,1])
    # 20%
    POSE_PUZZLE_CIRCLE_20 = np.ndarray([1,1,1,1,1,1])
    POSE_PUZZLE_SQUARE_20 = np.ndarray([1,1,1,1,1,1])
    POSE_PUZZLE_HEXAGON_20 = np.ndarray([1,1,1,1,1,1])



    @staticmethod
    def get_hole_pose_by_name(name: str) -> np.ndarray|None:
        if name == 'square':
            hole_pose = RobotCte.POSE_PUZZLE_SQUARE_20
        elif name == 'circle':
            hole_pose= RobotCte.POSE_PUZZLE_CIRCLE_20
        elif name == 'hexagon':
            hole_pose = RobotCte.POSE_PUZZLE_HEXAGON_20
        else:
            return
        return hole_pose
