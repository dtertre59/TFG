
import os
from pathlib import Path
import time

import rtde.rtde as rtde

import mains.main_robot as robot
import mains.main_nn as nn
from mains.main_nn import Piece


class Positions():

    # TOOL POS para mantener la pinza hacia abajo apuntado tods los ejes bien
    # radianes
    RX = 3.1415 # mirando al lado contrario
    RY = 0.2617  # 15º por la pinza
    RZ = 0


    BORDE_TAKE_X = -0.255
    BORDE_TAKE_Y_1 = -0.427
    BORDE_TAKE_Y_2 = - 0.377

    BORDE_RELEASE_X = -0.050
    BORDE_RELEASE_Y_CIRCLE = -0.395
    BORDE_RELEASE_Y_SQUARE = - 0.345

    MOVE_Z = 0.120
    TAKE_Z = 0.055
    RELEASE_Z = 0.050

    POS_INIT = [0.128, -0.298, 0.180, RX, RY, RZ]

    POS_PIECE_1_I = [BORDE_TAKE_X, BORDE_TAKE_Y_1, MOVE_Z, RX, RY, RZ]
    POS_PIECE_1_F = [BORDE_TAKE_X, BORDE_TAKE_Y_1, TAKE_Z, RX, RY, RZ]

    POS_PIECE_2_I = [BORDE_TAKE_X, BORDE_TAKE_Y_2, MOVE_Z, RX, RY, RZ]
    POS_PIECE_2_F = [BORDE_TAKE_X, BORDE_TAKE_Y_2, TAKE_Z, RX, RY, RZ]


    POS_PUZZLE_CIRCLE_I = [BORDE_RELEASE_X, BORDE_RELEASE_Y_CIRCLE, MOVE_Z, RX, RY, RZ]
    POS_PUZZLE_CIRCLE_F = [BORDE_RELEASE_X, BORDE_RELEASE_Y_CIRCLE, RELEASE_Z, RX, RY, RZ]

    POS_PUZZLE_SQUARE_I = [BORDE_RELEASE_X, BORDE_RELEASE_Y_SQUARE, MOVE_Z, RX, RY, RZ]
    POS_PUZZLE_SQUARE_F = [BORDE_RELEASE_X, BORDE_RELEASE_Y_SQUARE, RELEASE_Z, RX, RY, RZ]
    



def move_robot():
    ROBOT_HOST = '192.168.10.222' # "localhost"
    ROBOT_PORT = 30004
    config_filename = "control_loop_configuration.xml"

    # setp1 = [-0.13787, -0.29821, 0.03, -2.21176, -2.21104, 0.01494]

    con = robot.connect_robot(host=ROBOT_HOST, port=ROBOT_PORT)
    setp, watchdog, gripper = robot.setup_robot(con=con, config_file=config_filename)

    # 0. robot en posicion inicial
    robot.robot_move(con, setp, watchdog, Positions.POS_PIECE_2_I)
    input('movido1')
    robot.robot_move(con, setp, watchdog, Positions.POS_PIECE_2_F)
    input('gripper')
    robot.gripper_control(con,gripper=gripper,gripper_on=True)
    input('move3')
    robot.robot_move(con, setp, watchdog, Positions.POS_PIECE_2_I)
    input('move4')
    robot.robot_move(con, setp, watchdog, Positions.POS_PUZZLE_SQUARE_I)
    input('move5')
    robot.robot_move(con, setp, watchdog, Positions.POS_PUZZLE_SQUARE_F)
    input('gripper')
    robot.gripper_control(con,gripper=gripper,gripper_on=False)
    input('move6')
    robot.robot_move(con, setp, watchdog, Positions.POS_PUZZLE_SQUARE_I)
    input('FIN')


def main():
    # ----- PARTE DE CAMARA -----

    # adquirimos piezas
    pieces = nn.streaming_with__nn()
    if not pieces:
        return 
    # vemos piezas detectadas y en que posiciones se encuentran
    order = 0
    for piece in pieces:
        print(order, ': ',piece.__dict__)
        order += 1
    
    # ------ Sin camara ------
    # piece1 = Piece(name='scuare', coordinates=(2,0,0), color=(255,0,0))
    # piece2 = Piece(name='circle', coordinates=(4,0,0), color=(0,255,0))
    # pieces = [piece1, piece2]

    input('FIN DE LA CAMARA, COMIENZO DE LOS MOVIMIENTOS DEL ROBOT')
    # ----- PARTE DE ROBOT -----
        
    # nos conectamos al robot y le decimos donde estan las piezas para que las coja y las mueva de posicion
     # GOLBAL VARIABLES
    ROBOT_HOST = '192.168.10.222' # "localhost"
    ROBOT_PORT = 30004
    config_filename = "control_loop_configuration.xml"

    # setp1 = [-0.13787, -0.29821, 0.03, -2.21176, -2.21104, 0.01494]

    con = robot.connect_robot(host=ROBOT_HOST, port=ROBOT_PORT)
    setp, watchdog, gripper = robot.setup_robot(con=con, config_file=config_filename)

    # 0. robot en posicion inicial
    robot.robot_move(con, setp, watchdog, Positions.POS_INIT)
    robot.gripper_control(con,gripper=gripper,gripper_on=False)

    order = 0
    for piece in pieces:
        if piece.name == 'scuare':
            piece.name = 'square'

        print('Pieza: ',piece.__dict__)
        
        # vectores de posicion
        print('Position: ', order + 1)
        if order == 0:       
            vector_i = Positions.POS_PIECE_1_I
            vector_f = Positions.POS_PIECE_1_F
        elif order == 1:
            vector_i = Positions.POS_PIECE_2_I
            vector_f = Positions.POS_PIECE_2_F
        

        # vectores de puzzle
        if piece.name == 'circle':
            print('CIRCLE')
            vector_p_i = Positions.POS_PUZZLE_CIRCLE_I
            vector_p_f = Positions.POS_PUZZLE_CIRCLE_F
        elif piece.name == 'square':
            print('SQUARE')
            vector_p_i = Positions.POS_PUZZLE_SQUARE_I
            vector_p_f = Positions.POS_PUZZLE_SQUARE_F
        

        # 1. posicion de coger pieza
        robot.robot_move(con, setp, watchdog, vector_i) # el programa sobrescribe el vector en setp y lo envia para que se mueva a esa posicion
        # 2. abrimos pinza
        robot.gripper_control(con,gripper,False)
        # 3. bajamos
        robot.robot_move(con, setp, watchdog, vector_f)
        # 4. cerramos pinza
        robot.gripper_control(con,gripper,True)
        # 5. subimos
        robot.robot_move(con, setp, watchdog, vector_i) 
        # 6. llevamos a posicion del puzle
        robot.robot_move(con, setp, watchdog, vector_p_i)
        # 7. bajamos la hoyo
        robot.robot_move(con, setp, watchdog, vector_p_f)
        # 8. abrimos pinza
        robot.gripper_control(con,gripper,False)
        # 8. posicion arriba
        robot.robot_move(con, setp, watchdog, vector_p_i)
        
        order += 1

if __name__ == '__main__':
    main()
    # move_robot()


