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


class RobotCte():
   

    POSE_STANDAR = np.array([-0.128, -0.298, 0.180, 0.025, 0, 2.879])
    # Posicion para la visualizacion de las piezas
    POSE_DISPLAY = np.array([-0.125, -0.166, 0.170, 1.454, -1.401, -4.095])

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
