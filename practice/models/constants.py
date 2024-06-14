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


# -------------------- CONSTANTS ----------------------------------------------------------------------------------------- #

# -------------------- COLOR RGB ----------------------------------------------------------------------------------------- #

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


# -------------------- CAMERA -------------------------------------------------------------------------------------------- #

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
    
    T_pointcloud_to_good_pointcloud_4 = np.array([[    0.99934, -0.00093728,   -0.036414,      21.921],
                                                  [  0.0012635,     0.99996,   0.0089371,     -4.5045],
                                                  [   0.036404,  -0.0089772,      0.9993,     -11.619],
                                                  [          0,           0,           0,           1]])


# -------------------- ROBOT --------------------------------------------------------------------------------------------- #

class RobotCte():

    # POSE_STANDAR = np.array([-0.128, -0.298, 0.180, 0.025, 0, 2.879])
    POSE_STANDAR = np.array([-0.134, -0.298, 0.170, 0, 0, 4.438])
    # Posicion para la visualizacion de las piezas
    # POSE_DISPLAY = np.array([-0.125, -0.166, 0.170, 1.454, -1.401, -4.095])
    POSE_DISPLAY = np.array(([-0.1385, -0.1758, 0.2457, 0.921, 1.097, 4.151]))

    POSE_DISPLAY_2 = np.array(([-0.1385, -0.1758, 0.2457, 0.921, 1.097, 4.151]))

    # Posicion segura para el movimiento
    SAFE_Z = 0.075
    SAFE_Z_2 = 0.15

    TAKE_PIECE_Z = 0.050
    LEFT_PIECE_Z = 0.037

    # POSE_SAFE = np.array([0.128, -0.298, 0.180, 3.1415, 0.2617, 0])
    # POSE_SAFE = np.array([-0.128, -0.298, 0.180, 0.025, 0, 2.879])


     # Posicion del a base del robot
    POSE_ROBOT_BASE = np.array([0, 0, 0])
    # Posicion del apriltag de referencia
    POSE_APRILTAG_REF = np.array([-0.016, -0.3203, 0.01, 0.0, 0.0, 6.01])
    POSE_SAFE_APRILTAG_REF = np.array([-0.016, -0.3203, SAFE_Z, 0.0, 0.0, 6.01])


    # Matriz de transicion del sistema de referencia del apriltag(ref) al de la base del robot
    T_REF_TO_ROBOT = np.array([[1, 0, 0, POSE_APRILTAG_REF[0]],
                                [0, 1, 0, POSE_APRILTAG_REF[1]],
                                [0, 0, 1, POSE_APRILTAG_REF[2]],
                                [0, 0, 0, 1]])


    # POSICIONES DE LOS HOYOS
    # 5%
    ROTATION = np.array([0,0,2.925])
    POSE_PUZZLE_SQUARE_5 = np.array([-0.435, -0.0678, SAFE_Z, ROTATION[0], ROTATION[1], ROTATION[2]])
    POSE_PUZZLE_HEXAGON_5 = np.array([-0.433, -0.0185, SAFE_Z, ROTATION[0], ROTATION[1], ROTATION[2]])
    POSE_PUZZLE_CIRCLE_5 = np.array([-0.432, -0.117, SAFE_Z, ROTATION[0], ROTATION[1], ROTATION[2]])
    # 10%
    POSE_PUZZLE_SQUARE_10 = np.array([-0.385, -0.067, SAFE_Z, ROTATION[0], ROTATION[1], ROTATION[2]])
    POSE_PUZZLE_HEXAGON_10 = np.array([-0.3855, -0.0185, SAFE_Z, ROTATION[0], ROTATION[1], ROTATION[2]])
    POSE_PUZZLE_CIRCLE_10 = np.array([-0.385, -0.1168, SAFE_Z, ROTATION[0], ROTATION[1], ROTATION[2]])
    # 15%   
    POSE_PUZZLE_SQUARE_15 = np.array([-0.3338, -0.068, SAFE_Z, ROTATION[0], ROTATION[1], ROTATION[2]])
    POSE_PUZZLE_HEXAGON_15 = np.array([-0.3345, -0.0185, SAFE_Z, ROTATION[0], ROTATION[1], ROTATION[2]])
    POSE_PUZZLE_CIRCLE_15 = np.array([-0.3345, -0.1168, SAFE_Z, ROTATION[0], ROTATION[1], ROTATION[2]])



    @staticmethod
    def get_hole_pose_by_name(name: str, tolerance: int = 15) -> np.ndarray|None:
        hole_pose = None
        if tolerance == 15:
            if name == 'square':
                hole_pose = RobotCte.POSE_PUZZLE_SQUARE_15
            elif name == 'circle':
                hole_pose= RobotCte.POSE_PUZZLE_CIRCLE_15
            elif name == 'hexagon':
                hole_pose = RobotCte.POSE_PUZZLE_HEXAGON_15
        elif tolerance == 10:
            if name == 'square':
                hole_pose = RobotCte.POSE_PUZZLE_SQUARE_10
            elif name == 'circle':
                hole_pose= RobotCte.POSE_PUZZLE_CIRCLE_10
            elif name == 'hexagon':
                hole_pose = RobotCte.POSE_PUZZLE_HEXAGON_10
        elif tolerance == 5:
            if name == 'square':
                hole_pose = RobotCte.POSE_PUZZLE_SQUARE_5
            elif name == 'circle':
                hole_pose= RobotCte.POSE_PUZZLE_CIRCLE_5
            elif name == 'hexagon':
                hole_pose = RobotCte.POSE_PUZZLE_HEXAGON_5

        return hole_pose