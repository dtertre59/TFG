
import time
import logging
import sys

import rtde.rtde as rtde
import rtde.rtde_config as rtde_config


class RobotException(Exception):
    """Excepcion del robot"""


class Robot():
    # init
    def __init__(self, host: str, port: int, config_filename: str) -> None:
        self.host = host
        self.port = port 
        self.config_filename = config_filename

        self.con = rtde.RTDE(hostname=self.host,port=self.port)

        # cogen valor el setup robot
        self.setp = None
        self.watchdog = None
        self.gripper = None

    # Connection
    def connect(self)-> rtde.RTDE:

        connection_state = self.con.connect()
        while connection_state != None:
            print(connection_state)
            time.sleep(1)
            connection_state = self.con.connect()
        print('connected to the robot')

        return self.con

    # Setup
    def setup(self) -> None:
        logging.getLogger().setLevel(logging.INFO)

        conf = rtde_config.ConfigFile(self.config_filename)

        # del xml
        state_names, state_types = conf.get_recipe("state")
        setp_names, setp_types = conf.get_recipe("setp")
        watchdog_names, watchdog_types = conf.get_recipe("watchdog")
        gripper_names, gripper_types = conf.get_recipe("gripper")
        # print(state_names, state_types, setp_names, setp_types, watchdog_names, watchdog_types)
            
        # get controller version
        self.con.get_controller_version()

        # setup recipes
        self.con.send_output_setup(state_names, state_types)

        self.setp = self.con.send_input_setup(setp_names, setp_types)
        self.watchdog = self.con.send_input_setup(watchdog_names, watchdog_types)
        self.gripper = self.con.send_input_setup(gripper_names, gripper_types)

        # Setpoints to move the robot to
        # setp1 = [-0.13787, -0.29821, 0.03, -2.21176, -2.21104, 0.01494]
        p_axis = [-0.1075, -0.3839, 0.0108, 2.035, 2.411, 0]
        # reset
        self.setp.input_double_register_0 = p_axis[0]
        self.setp.input_double_register_1 = p_axis[1]
        self.setp.input_double_register_2 = p_axis[2]
        self.setp.input_double_register_3 = p_axis[3]
        self.setp.input_double_register_4 = p_axis[4]
        self.setp.input_double_register_5 = p_axis[5]

        return
        
    # movimiento del robot
    def move(self, vector):
        """ Si el watchdog == 1 -> el robot se esta moviendo"""

        # start data synchronization
        if not self.con.send_start():
            raise RobotException("robot send_start failed")
        
        move_completed = False

        # The function "rtde_set_watchdog" in the "rtde_control_loop.urp" creates a 1 Hz watchdog
        self.watchdog.input_bit_register_127 = 0

        init = 1
        while move_completed == False:
            # receive the current state; recibimos los datos que tenemo en el .xml state
            state = self.con.receive()
            robot_aviable = state.output_int_register_1
            # print(robot_aviable)

            if state is None:
                print('state None')
                break
            
            # funcion de transformacion inversa
            def list_to_setp(sp, list):
                for i in range(0, 6):
                    sp.__dict__["input_double_register_%i" % i] = list[i]
                return sp

            # do something...
            if move_completed == False and robot_aviable == 1 and init == 1:
                # print('inicio')
                # print('Programa en funcionamiento')
                list_to_setp(self.setp, vector) # cambiamos los inputs registers por el vector 6d a donde queremos movernos.
                self.con.send(self.setp)
                time.sleep(0.5)
                init = 0
                

            # No se ha movido, esta parado en la misma posicion
            elif init == 0 and robot_aviable == 1:
                # print('salida')
                move_completed = True

            time.sleep(0.2)
            # kick watchdog
            self.con.send(self.watchdog)
        return
    
    # control del gripper del robot
    def gripper_control(self, gripper_on: bool):
        # hace falta -> SI pero no se por qué para iniciar el envio
        if not self.con.send_start():
            raise RobotException("robot send_start failed")
        
        # The function "rtde_set_watchdog" in the "rtde_control_loop.urp" creates a 1 Hz watchdog
        # watchdog.input_bit_register_127 = True
        self.gripper.input_bit_register_126 = gripper_on

        while True:
            # receive the current state; recivimos los datos que tenemo en el .xml state
            state = self.con.receive()
            if state is None:
                print('state None')
                break
            if state.output_int_register_0 == 1:
                print('programa funcionando el local')
                break
            self.con.send(self.gripper)
            break
        time.sleep(1)
        return