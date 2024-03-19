
import os
from pathlib import Path
import time

import rtde.rtde as rtde

import main_robot as robot
import main_nn as nn
from main_nn import Piece


class Positions():
    POS_INIT = [0.133, -0.425, 0.478, 1.111, 1.352, -1.342]
    POS_PIECE_1 = [-0.152, -0.425, 0.560, 1.111, 1.352, -1.342]
    POS_PIECE_2 = [0.218, -0.425, 0.560, 1.111, 1.352, -1.342]

    POS_PUZZLE_SQUARE = [-0.152, -0.425, 0.146, 1.111, 1.352, -1.342]
    POS_PUZZLE_CIRCLE = [0.218, -0.425, 0.146, 1.111, 1.352, -1.342]
    




def main():
    # adquirimos piezas
    pieces = nn.streaming_with__nn()
    if not pieces:
        return 
    # vemos piezas detectadas y en que posiciones se encuentran
    order = 0
    for piece in pieces:
        print(order, ': ',piece.__dict__)
        order += 1
    
    # nos conectamos al robot y le decimos donde estan las piezas para que las coja y las mueva de posicion
     # GOLBAL VARIABLES
    ROBOT_HOST = '192.168.10.222' # "localhost"
    ROBOT_PORT = 30004
    config_filename = "control_loop_configuration.xml"

    # setp1 = [-0.13787, -0.29821, 0.03, -2.21176, -2.21104, 0.01494]

    con = robot.connect_robot(host=ROBOT_HOST, port=ROBOT_PORT)
    setp, watchdog = robot.setup_robot(con=con, config_file=config_filename)
    vector = []

    order = 0
    for piece in pieces:
        if order == 0:
            vector = Positions.POS_PIECE_1
        elif order == 1:
            vector = Positions.POS_PIECE_2
        else:
            # Salimos del bucle porque no hay mas piezas que coger
            break
        
        # 1. posicion de coger pieza
        robot.robot_control_loop(con, setp, watchdog, vector) # el programa sobrescribe el vector en setp y lo envia para que se mueva a esa posicion
        # 2. abrimos pinza

        # 3. bajamos

        # 4. cerramos pinza

        # 5. subimos

        # 6. llevamos a posicion del puzle
        
        # 7. soltamos pieza

        # 8. posicion standar

        
        order += 1

if __name__ == '__main__':
    main()


