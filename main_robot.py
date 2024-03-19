
import sys
import time

import logging

import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

p_inicial = [0.133, -0.425, 0.478, 1.111, 1.352, -1.342]
p_limite_11 = [-0.152, -0.425, 0.560, 1.111, 1.352, -1.342]
p_limite_12 = [0.218, -0.425, 0.560, 1.111, 1.352, -1.342]
p_limite_21 = [-0.152, -0.425, 0.146, 1.111, 1.352, -1.342]
p_limite_22 = [0.218, -0.425, 0.146, 1.111, 1.352, -1.342]

def connect_robot(host: str, port: str)-> rtde.RTDE:
    con = rtde.RTDE(hostname=host,port=port)

    connection_state = con.connect()
    while connection_state != None:
        print(connection_state)
        time.sleep(1)
        connection_state = con.connect()
    print('connected to the robot')
    return con

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
    setp.input_double_register_0 = 0
    setp.input_double_register_1 = 0
    setp.input_double_register_2 = 0
    setp.input_double_register_3 = 0
    setp.input_double_register_4 = 0
    setp.input_double_register_5 = 0

    return setp, watchdog, gripper


def setp_to_list(sp):
    sp_list = []
    for i in range(0, 6):
        sp_list.append(sp.__dict__["input_double_register_%i" % i])
    return sp_list


def list_to_setp(sp, list):
    for i in range(0, 6):
        sp.__dict__["input_double_register_%i" % i] = list[i]
    return sp


def robot_control_loop(con: rtde.RTDE, setp, watchdog, vector):

    # start data synchronization
    if not con.send_start(): # ES NECESARIO???????????????????????????????????????
        sys.exit()
    # move_completed = True
    # The function "rtde_set_watchdog" in the "rtde_control_loop.urp" creates a 1 Hz watchdog
    watchdog.input_bit_register_127 = True

    while True:
        # receive the current state; recivimos los datos que tenemo en el .xml state
        state = con.receive()
        # print(state.__dict__)
        # input(setp.__dict__)
        # print(state.output_int_register_0)
        if state is None:
            print('state None')
            break
        
        if state.output_int_register_0 == 1:
            print('programa funcionando el local')
            break
        
        # do something...
        if state.output_int_register_0 == 0:
            # new_setp = setp1 if setp_to_list(setp) == setp2 else setp2
            # print("New pose = " + str(setp1))
            # send new setpoint
            list_to_setp(setp, vector) # cambiamos los inputs registers por el vector 6d a donde queremos movernos.
            a = con.send(setp)
            # print(a) # True -> enviado correctamente
        # kick watchdog
        con.send(watchdog)
        break


def gripper_control(con: rtde.RTDE,watchdog, gripper, gripper_on: bool):
    gripper.input_bit_register_126 = True
    if not con.send_start(): # ES NECESARIO???????????????????????????????????????
        sys.exit()
    # move_completed = True
    # The function "rtde_set_watchdog" in the "rtde_control_loop.urp" creates a 1 Hz watchdog
    # watchdog.input_bit_register_127 = True
    
    while True:
        # receive the current state; recivimos los datos que tenemo en el .xml state
        state = con.receive()
        # print(state.__dict__)
        # input(setp.__dict__)
        # print(state.output_int_register_0)
        if state is None:
            print('state None')
            break
        if state.output_int_register_0 == 1:
            print('programa funcionando el local')
            break
        con.send(gripper)
    return



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
        gripper_control(con, watchdog, gripper=gripper, gripper_on=gripper_on)
        break

    con.send_pause()
    con.disconnect()



def main():
    # GOLBAL VARIABLES
    ROBOT_HOST = '192.168.10.222' # "localhost"
    ROBOT_PORT = 30004
    config_filename = "control_loop_configuration.xml"

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
        robot_control_loop(con, setp, watchdog, vector)

main2()