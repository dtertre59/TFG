"""
        ur3e_functions.py

    Este Script contiene las funciones de interacción con el robot ur3e, de universal robots.
    La libreria utilizada es rtde (Real time data exchange), proporcionada por el mismo fabricante.

"""

# -------------------- PACKAGES ------------------------------------------------------------------------------------------ #

import sys
import time


from pathlib import Path

import logging

import rtde.rtde as rtde
import rtde.rtde_config as rtde_config


# -------------------- VARIABLES ----------------------------------------------------------------------------------------- #

p_axis = [-0.1075, -0.3839, 0.0108, 2.035, 2.411, 0]

p_inicial = [0.133, -0.425, 0.478, 1.111, 1.352, -1.342]
p_limite_11 = [-0.152, -0.425, 0.560, 1.111, 1.352, -1.342]
p_limite_12 = [0.218, -0.425, 0.560, 1.111, 1.352, -1.342]
p_limite_21 = [-0.152, -0.425, 0.146, 1.111, 1.352, -1.342]
p_limite_22 = [0.218, -0.425, 0.146, 1.111, 1.352, -1.342]


# -------------------- FUNCTIONS ----------------------------------------------------------------------------------------- #

# robot connection
def connect_robot(host: str, port: str)-> rtde.RTDE:
    con = rtde.RTDE(hostname=host,port=port)

    connection_state = con.connect()
    while connection_state != None:
        print(connection_state)
        time.sleep(1)
        connection_state = con.connect()
    print('connected to the robot')
    return con

# robot setup
def setup_robot(con: rtde.RTDE, config_file: str):

    logging.getLogger().setLevel(logging.INFO)

    conf = rtde_config.ConfigFile(config_file)
    # del xml
    state_names, state_types = conf.get_recipe("state")
    setp_names, setp_types = conf.get_recipe("setp")
    watchdog_names, watchdog_types = conf.get_recipe("watchdog")
    gripper_names, gripper_types = conf.get_recipe("gripper")
    # print(state_names, state_types, setp_names, setp_types, watchdog_names, watchdog_types)
        
    # get controller version
    con.get_controller_version()

    # setup recipes
    con.send_output_setup(state_names, state_types)
    setp = con.send_input_setup(setp_names, setp_types)
    watchdog = con.send_input_setup(watchdog_names, watchdog_types)
    gripper = con.send_input_setup(gripper_names, gripper_types)

    # Setpoints to move the robot to
    # setp1 = [-0.13787, -0.29821, 0.03, -2.21176, -2.21104, 0.01494]

    # reset
    setp.input_double_register_0 = p_axis[0]
    setp.input_double_register_1 = p_axis[1]
    setp.input_double_register_2 = p_axis[2]
    setp.input_double_register_3 = p_axis[3]
    setp.input_double_register_4 = p_axis[4]
    setp.input_double_register_5 = p_axis[5]

    return setp, watchdog, gripper

# init robot
def init_robot():
    ROBOT_HOST = '192.168.10.222' # "localhost"
    ROBOT_PORT = 30004
    config_filename = str(Path(__file__).resolve().parent.parent / 'assets' / 'ur3e' / 'configuration_1.xml')

    # setp1 = [-0.13787, -0.29821, 0.03, -2.21176, -2.21104, 0.01494]

    con = connect_robot(host=ROBOT_HOST, port=ROBOT_PORT)
    setp, watchdog, gripper = setup_robot(con=con, config_file=config_filename)

    return con, setp, watchdog, gripper

# funcion de trasformacion
def setp_to_list(sp):
    sp_list = []
    for i in range(0, 6):
        sp_list.append(sp.__dict__["input_double_register_%i" % i])
    return sp_list

# funcion de transformacion inversa
def list_to_setp(sp, list):
    for i in range(0, 6):
        sp.__dict__["input_double_register_%i" % i] = list[i]
    return sp

# movimiento del robot
def robot_move(con: rtde.RTDE, setp, watchdog, vector):
    """ Si el watchdog == 1 -> el robot se esta moviendo"""
    # start data synchronization
    if not con.send_start():
        sys.exit()
    
    move_completed = False

    # The function "rtde_set_watchdog" in the "rtde_control_loop.urp" creates a 1 Hz watchdog
    watchdog.input_bit_register_127 = 0

    init = 1
    while move_completed == False:
        # receive the current state; recibimos los datos que tenemo en el .xml state
        state = con.receive()
        robot_aviable = state.output_int_register_1
        # print(robot_aviable)

        if state is None:
            print('state None')
            break
        

        # do something...
        if move_completed == False and robot_aviable == 1 and init == 1:
            # print('inicio')
            # print('Programa en funcionamiento')
            list_to_setp(setp, vector) # cambiamos los inputs registers por el vector 6d a donde queremos movernos.
            con.send(setp)
            time.sleep(0.5)
            init = 0
            

        # No se ha movido, esta parado en la misma posicion
        elif init == 0 and robot_aviable == 1:
            # print('salida')
            move_completed = True

        time.sleep(0.2)
        # kick watchdog
        con.send(watchdog)
    return

# control del gripper
def gripper_control(con: rtde.RTDE, gripper, gripper_on: bool):
    # hace falta -> SI pero no se por qué
    if not con.send_start():
        sys.exit()
    
    # The function "rtde_set_watchdog" in the "rtde_control_loop.urp" creates a 1 Hz watchdog
    # watchdog.input_bit_register_127 = True
    gripper.input_bit_register_126 = gripper_on

    while True:
        # receive the current state; recivimos los datos que tenemo en el .xml state
        state = con.receive()
        if state is None:
            print('state None')
            break
        if state.output_int_register_0 == 1:
            print('programa funcionando el local')
            break
        con.send(gripper)
        break
    time.sleep(1)
    return


# -------------------- TRAINNING ----------------------------------------------------------------------------------------- #

def main():
    # GOLBAL VARIABLES
    ROBOT_HOST = '192.168.10.222' # "localhost"
    ROBOT_PORT = 30004
    config_filename = "configuration_1.xml"

    # setp1 = [-0.13787, -0.29821, 0.03, -2.21176, -2.21104, 0.01494]

    con = connect_robot(host=ROBOT_HOST, port=ROBOT_PORT)
    setp, watchdog, gripper = setup_robot(con=con, config_file=config_filename)
    vector = []

    while True:
        try:
            p = int(input('posicion: '))
        except:
            p = 0
        if p == 0:
           break 
        if p == 1:
            vector = p_limite_11
        elif p == 2:
            vector = p_limite_12
        elif p == 3:
            vector = p_limite_21
        elif p == 4:
            vector = p_limite_22
        else:
            vector = p_inicial
        robot_move(con, setp, watchdog, vector)

def main2():
    # GOLBAL VARIABLES
    ROBOT_HOST = '192.168.10.222' # "localhost"
    ROBOT_PORT = 30004
    config_filename = "control_loop_configuration.xml"

    # setp1 = [-0.13787, -0.29821, 0.03, -2.21176, -2.21104, 0.01494]

    con = connect_robot(host=ROBOT_HOST, port=ROBOT_PORT)
    setp, watchdog, gripper = setup_robot(con=con, config_file=config_filename)

    gripper_on = True
    while True:
        input('change')
        gripper_control(con, gripper=gripper, gripper_on=gripper_on)
        time.sleep(1.5)
        if gripper_on == False:
            gripper_on = True
        else:
            gripper_on = False

def main3():
    POS_INIT = [0.128, -0.298, 0.180, 3.1415, 0.2617, 0]
    POS_FIN = [0.128, -0.298, 0.100, 3.1415, 0.2617, 0]


    con, setp, watchdog, gripper = init_robot()

    while True:
        print('aa')
        robot_move(con, setp, watchdog, vector=POS_INIT)
        input('FIN INIT')
        robot_move(con, setp,watchdog, vector=POS_FIN)
        input('FIN FIN')

def main4():
    ROBOT_HOST = '192.168.10.222' # "localhost"
    ROBOT_PORT = 30004
    config_filename = config_filename = str(Path(__file__).resolve().parent.parent / 'assets' / 'ur3e' / 'configuration_1.xml')

    # setp1 = [-0.13787, -0.29821, 0.03, -2.21176, -2.21104, 0.01494]

    con = connect_robot(host=ROBOT_HOST, port=ROBOT_PORT)
    setp, watchdog, gripper = setup_robot(con=con, config_file=config_filename)

    # 0. robot en posicion inicial
    robot_move(con, setp, watchdog, p_axis)
    gripper_control(con,gripper=gripper,gripper_on=False)

    input('FIN')


# main4()