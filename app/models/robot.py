"""
        robot.py

    Robot 

    Registros utilizados:

        - Estado (receive):
            Posición objetivo -> name="target_q" type="VECTOR6D"/>
            Velocidad objetivo ->  name="target_qd" type="VECTOR6D"/>
            TCP pose actual -> name="actual_TCP_pose" type="VECTOR6D"/>
            Programa de PolyScope en funcionamiento ->  name="output_int_register_0" type="INT32"/>
            Robot esperando -> name="output_int_register_1" type="INT32"/>
            

        - Pose (send):
            x -> name="input_double_register_0" type="DOUBLE"/>
            y -> name="input_double_register_1" type="DOUBLE"/>
            z -> name="input_double_register_2" type="DOUBLE"/>
            rx -> name="input_double_register_3" type="DOUBLE"/>
            ry -> name="input_double_register_4" type="DOUBLE"/>
            rz -> name="input_double_register_5" type="DOUBLE"/>

        - Control del Gripper (send) -> name="input_bit_register_126" type="BOOL"

        Sin utilidad todavia en el programa de polyscope
        - watchdog (send) -> name="input_bit_register_127" type="BOOL"/>



"""

# -------------------- PACKAGES ------------------------------------------------------------------------------------------ #

import time
import logging
import sys
import numpy as np

import rtde.rtde as rtde
import rtde.rtde_config as rtde_config


# -------------------- VARIABLES ----------------------------------------------------------------------------------------- #


# -------------------- CLASSES ------------------------------------------------------------------------------------------- #

class RobotException(Exception):
    """Excepcion del robot"""
    def __init__(self, msg: str):
        msg = 'Excepcion Robot: ' + msg
        super().__init__(msg)



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

     # funcion de transformacion inversa de la POSE
    
    def __del__(self) -> None:
        print('Destructor de la instancia del robot')
        if not self.con.send_start():
            raise RobotException("Fallo en la instrucción de start")

        self.watchdog.input_bit_register_127 = 0
        self.con.send(self.watchdog)

    # transformacion vector a rgi-stros de pose
    def _list_to_setp(self, list):
        for i in range(0, 6):
            self.setp.__dict__["input_double_register_%i" % i] = list[i]
        return
    
    # Connection
    def connect(self)-> rtde.RTDE:
        try:
            connection_state = self.con.connect()
            while connection_state != None:
                print(connection_state)
                time.sleep(1)
                connection_state = self.con.connect()
            print('Robot conectado')
        except Exception as e:
            raise RobotException("Fallo en la conexion")

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
        # p_axis = [-0.1075, -0.3839, 0.0108, 2.035, 2.411, 0]
        # # reset
        # self.setp.input_double_register_0 = p_axis[0]
        # self.setp.input_double_register_1 = p_axis[1]
        # self.setp.input_double_register_2 = p_axis[2]
        # self.setp.input_double_register_3 = p_axis[3]
        # self.setp.input_double_register_4 = p_axis[4]
        # self.setp.input_double_register_5 = p_axis[5]

        print('Setup del robot completo')

        # escribimos registros para indicar al robot que estamos conectados y seteados
        if not self.con.send_start():
            raise RobotException("Fallo en la instrucción de start")
        # registros iniciales
        self._list_to_setp([-0.128, -0.298, 0.180, 0.015, 0, 1.501]) # cambiamos los inputs registers por el vector 6d a donde queremos movernos.
        self.con.send(self.setp)
        # kick watchdog
        self.watchdog.input_bit_register_127 = 1
        self.con.send(self.watchdog)

        return
        
    # movimiento del robot
    def move(self, vector):
        # start data synchronization
        if not self.con.send_start():
            raise RobotException("Fallo en la instrucción de start")
        
        move_completed = False

        # The function "rtde_set_watchdog" in the "rtde_control_loop.urp" creates a 1 Hz watchdog
        # self.watchdog.input_bit_register_127 = 0

        init = 1
        # bucle de movimiento del robot
        while move_completed == False:
            # receive the current state; recibimos los datos que tenemo en el .xml state
            state = self.con.receive()
            if state is None:
                raise RobotException('No se recibe el estado')
            
            # list_to_setp(self.setp, vector) # cambiamos los inputs registers por el vector 6d a donde queremos movernos.
            # self.con.send(self.setp)


            # comprobamos que el programa del robot esta en funcionamiento
            program_running = state.output_int_register_0
            if not program_running:
                raise RobotException('No está activado el programa en PolyScope')


            # comprobamos estado del robot (parado=1, en movimiento = 0)
            robot_aviable = state.output_int_register_1

            # do something...
            if move_completed == False and robot_aviable == 1 and init == 1:
                # print('inicio')
                print('Robot en movimiento a pose: ', vector)
                self._list_to_setp(vector) # cambiamos los inputs registers por el vector 6d a donde queremos movernos.
                self.con.send(self.setp)
                time.sleep(0.5)
                init = 0
                

            # No se ha movido, esta parado en la misma posicion
            elif init == 0 and robot_aviable == 1:
                print('Fin del movimiento del robot')
                move_completed = True

            time.sleep(0.2)
        return
    
    # control del gripper del robot
    def gripper_control(self, gripper_on: bool):
        # hace falta -> SI pero no se por qué para iniciar el envio
        if not self.con.send_start():
            raise RobotException("Fallo en la instrucción de start")
        
        # The function "rtde_set_watchdog" in the "rtde_control_loop.urp" creates a 1 Hz watchdog
        # watchdog.input_bit_register_127 = True
        self.gripper.input_bit_register_126 = gripper_on

        while True:
            # receive the current state; recibimos los datos que tenemo en el .xml state
            state = self.con.receive()
            if state is None:
                raise RobotException('No se recibe el estado')

            # comprobamos que el programa del robot esta en funcionamiento
            program_running = state.output_int_register_0
            if not program_running:
                raise RobotException('No está activado el programa en PolyScope')
            
            print('Gripper ON: ', gripper_on)
            self.con.send(self.gripper)
            break
        time.sleep(1)
        return
    


# pruebas
# from pathlib import Path
# ROBOT_HOST = '192.168.10.222' # "localhost"
# ROBOT_PORT = 30004
# robot_config_filename = config_filename = str(Path(__file__).resolve().parent.parent / 'assets' / 'ur3e' / 'configuration_1.xml')
# robot = Robot(ROBOT_HOST, ROBOT_PORT, robot_config_filename)

# robot.connect()
# robot.setup()

# POSE_STANDAR = np.array([-0.128, -0.298, 0.180, 0.025, 0, 2.879])
# POSE_DISPLAY = np.array([-0.125, -0.166, 0.270, 1.454, -1.401, -4.095])
# robot.move(POSE_STANDAR)
# robot.move(POSE_DISPLAY)

# def rx_ry_para(angulo_rotacion):
#     ang_rad = angulo_rotacion * (np.pi/180)
#     rx = np.pi * np.sin(ang_rad/2)
#     ry = np.pi * np.cos(ang_rad/2)
#     return np.array([rx, ry, 0])

# new_rot = RobotConstants.POSE_DISPLAY[-3:]

# new_rot = np.array([0,3.14,0])
# new_pose = np.append(RobotConstants.POSE_DISPLAY[:3], new_rot)
# print('rotamos 0º', new_pose)
# # input(new_pose)
# robot.move(new_pose)

# # new_rot = np.array([np.pi*np.sin(np.pi/4),np.pi*np.sin(np.pi/4),0])
# # new_rot = np.array([3.14,0,0])
# new_rot = rx_ry_para(90)
# new_pose = np.append(RobotConstants.POSE_DISPLAY[:3], new_rot)
# print('rotamos 90º', new_pose)
# robot.move(new_pose)

# new_rot = rx_ry_para(45)
# new_pose = np.append(RobotConstants.POSE_DISPLAY[:3], new_rot)
# print('rotamos 45º', new_pose)
# robot.move(new_pose)

# new_rot = rx_ry_para(30)
# new_pose = np.append(RobotConstants.POSE_DISPLAY[:3], new_rot)
# print('rotamos 30º', new_pose)
# robot.move(new_pose)

# new_rot = rx_ry_para(135)
# new_pose = np.append(RobotConstants.POSE_DISPLAY[:3], new_rot)
# print('rotamos 135º', new_pose)
# robot.move(new_pose)