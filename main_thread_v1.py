
import os
from pathlib import Path
import time

import rtde.rtde as rtde

import main_robot as robot
import main_nn as nn
from main_nn import Piece


class Positions():
    POS_INIT = [0.133, -0.425, 0.478, 1.111, 1.352, -1.342]

    POS_PIECE_1_I = [-0.152, -0.425, 0.560, 1.111, 1.352, -1.342]
    POS_PIECE_1_F = [0.218, -0.425, 0.560, 1.111, 1.352, -1.342]

    POS_PIECE_2_I = [-0.152, -0.425, 0.146, 1.111, 1.352, -1.342]
    POS_PIECE_2_F = [0.218, -0.425, 0.146, 1.111, 1.352, -1.342]

    POS_PUZZLE_SQUARE = [-0.152, -0.425, 0.146, 1.111, 1.352, -1.342]
    POS_PUZZLE_CIRCLE = [0.218, -0.425, 0.146, 1.111, 1.352, -1.342]
    




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
    robot.robot_control_loop(con, setp, watchdog, Positions.POS_INIT)
    robot.gripper_control(con,gripper=gripper,gripper_on=False)

    order = 0
    for piece in pieces:
        print(order)
        if order == 0:
            vector_i = Positions.POS_PIECE_1_I
            vector_f = Positions.POS_PIECE_1_F
        elif order == 1:
            vector_i = Positions.POS_PIECE_2_I
            vector_f = Positions.POS_PIECE_2_F

        # 1. posicion de coger pieza
        robot.robot_control_loop(con, setp, watchdog, vector_i) # el programa sobrescribe el vector en setp y lo envia para que se mueva a esa posicion
        # 2. abrimos pinza
        robot.gripper_control(con,gripper,True)
        # 3. bajamos
        robot.robot_control_loop(con, setp, watchdog, vector_f)
        # 4. cerramos pinza
        robot.gripper_control(con,gripper,False)
        # 5. subimos

        # 6. llevamos a posicion del puzle
        
        # 7. soltamos pieza

        # 8. posicion standar
        robot.robot_control_loop(con, setp, watchdog, Positions.POS_INIT)
        
        order += 1

if __name__ == '__main__':
    main()


