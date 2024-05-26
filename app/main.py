"""
        main.py

    Main script

"""

# -------------------- PACKAGES ------------------------------------------------------------------------------------------ #

import cv2
from pathlib import Path
import numpy as np

from functions import main_functions as mf
from functions import helper_functions as hf

# from models.robot import Robot
from models.camera import Camera
from models.robot import Robot
from models.detection import Apriltag, YoloObjectDetection, YoloPoseEstimation
from models.coordinator import Coordinator


# -------------------- VARIABLES ----------------------------------------------------------------------------------------- #

ROBOT_HOST = '192.168.10.222' # "localhost"
ROBOT_PORT = 30004
robot_config_filename = config_filename = str(Path(__file__).resolve().parent / 'assets' / 'ur3e' / 'configuration_1.xml')


# -------------------- FUNCTIONS ----------------------------------------------------------------------------------------- #

# main para testing de camera
def main_camera():
    camera = Camera(width=1920, height=1080, fx= 1498.367322, fy=1497.377563)
    # camera = Camera(width=1280, height=720, fx= 1498.367322, fy=1497.377563)

    camera.init_rgb()
    # camera.init_rgb_and_pointcloud()

    camera.run_with_options(name='irregular', crop_size=640)
    # camera.run_with_pointcloud(show3d=False)


# PRUEBAS DE DETECCIONES
def main_camera_detect():
    camera = Camera(width=1920, height=1080, fx= 1498.367322, fy=1497.377563)
    apriltag = Apriltag(family='tag36h11', size=0.015)
    nn_od_model = YoloObjectDetection(filename=str(Path(__file__).resolve().parent / 'assets' / 'nn_models' /'yolov8n_od_v1.pt'))
    nn_pose_model = YoloPoseEstimation(filename=str(Path(__file__).resolve().parent / 'assets' / 'nn_models' /'yolov8s_pose_irr_v1.pt'))
    
    
    camera.init_rgb()
    # camera.init_rgb_and_pointcloud()


    # camera.run_with_condition(Coordinator.apriltag_detections, apriltag, paint_frame = True)
    # camera.run_with_condition(Coordinator.nn_object_detections,  nn_od_model, paint_frame = True)
    camera.run_with_condition(Coordinator.nn_poseEstimation_detections,  nn_pose_model, paint_frame = True)

    # camera.run_with_pointcloud_with_condition(show3d=False, trigger_func=Coordinator.apriltag_detections, apriltag=apriltag, paint_frame = True)
    
    
    # try:
    #     # camera.run_with_condition(Coordinator.detections, nn_od_model, apriltag, paint_frame = True)
    #     r_kwargs = camera.run_with_pointcloud_with_condition(show3d=False, trigger_func=Coordinator.detections, nn_model=nn_pose_model, apriltag=apriltag, paint_frame = True)
    # except Exception as e:
    #     print('salida de camara: ',str(e))
    #     return
    
    # # 1. para el modo 2 encesitamos sacar la pointcloud. o por lo menos el punto 3d del centro de la pieza

    # ref = r_kwargs['ref']
    # piece = r_kwargs['pieces'][0]
    # frame = r_kwargs['frame']
    # pointcloud = r_kwargs['pointcloud']

    # hf.o3d_visualization([pointcloud])





# MAIN V1. Con red neuronal-object detection y apriltags
def main():
    print()
    print(' ----- Init ----- ')
    print()

    # 1. Instancias
    robot = Robot(ROBOT_HOST, ROBOT_PORT, robot_config_filename)
    # camera = Camera(width=3840, height=2160, fx= 2996.7346441158315, fy=2994.755126405525) 
    camera = Camera(width=1920, height=1080, fx= 1498.367322, fy=1497.377563) 
    # camera = Camera(width=1280, height=720, fx= 998.911548, fy=998.2517088)
    apriltag = Apriltag(family='tag36h11', size=0.015)
    nn_od_model = YoloObjectDetection(filename=str(Path(__file__).resolve().parent / 'assets' / 'nn_models' /'yolov8n_od_v1.pt'))  

    # 2. Init
    print('Inicio: ')
    try:
        # 2.1 robot
        robot.connect()
        robot.setup()
        # 2.2 camera
        camera.init_rgb()
        flag = True
    except Exception as e:
        print(str(e))
        flag = False

    
    # 3. Proceso completo de una pieza:
    while flag:
        flag = Coordinator.the_whole_process(robot, camera, apriltag, nn_od_model)
        if not flag:
            break

    print()
    print(' ----- End----- ')
    print()

    return



# MAIN V2. Con red neuronal-pose y pointcloud y apriltag para ref
def main2():
    print()
    print(' ----- Init ----- ')
    print()

    # 1. Instancias
    # robot = Robot(ROBOT_HOST, ROBOT_PORT, robot_config_filename)
    # camera = Camera(width=3840, height=2160, fx= 2996.7346441158315, fy=2994.755126405525) 
    camera = Camera(width=1920, height=1080, fx= 1498.367322, fy=1497.377563) 
    # camera = Camera(width=1280, height=720, fx= 998.911548, fy=998.2517088)
    apriltag = Apriltag(family='tag36h11', size=0.015)
    nn_pose_model = YoloPoseEstimation(filename=str(Path(__file__).resolve().parent / 'assets' / 'nn_models' /'yolov8s_pose_v5.pt'))  

    # 2. Init
    print('Inicio: ')
    try:
        # 2.1 robot
        # robot.connect()
        # robot.setup()
        # 2.2 camera
        camera.init_rgb_and_pointcloud()
        flag = True
    except Exception as e:
        print(str(e))
        flag = False

    
    # 3. Proceso completo de una pieza:
    while flag:
        flag = Coordinator.the_whole_process_2(camera, apriltag, nn_pose_model)
        if not flag:
            break

    print()
    print(' ----- End----- ')
    print()

    return














    # 1. conexion con el robot (verificar con algun registro)

    # 1.1 mover robot a posicion inicial o de reposo

    # 2. (thread) visualizar con la camara el area donde se encuentran las piezas (el robot en reposo ya apunta a este area)

    # 2.1 adquirimos frame (es necesario que se vea el
    #  apriltag de ref)

    # 2.2 deteccion de la pieza con red neuronal. Pose estimation con la red. por lo que concocemos el centro de la cara superior de la pieza

    # 2.3 pasamos pixel (centro) a punto en 3d de la nube de puntos (respecto de la camara)

    # 2.4 pasamos el punto 3d respecto de la camara al sistema de ref del robot utilizando el apriltag de referencia

    # 3. movimiento del robot para conger la pieza y dejarla en su respectivo hoyo (posicion conocida)
    
    # 5. (opcional) revisar que los hoyos no esten ocupados para poder mover la pieza a su hoyo

    # 6 .(opcional) poner un apriltag en el madero de los hoyos y asi no es necesario saber la posicion exacta de cada hoyo, solo la relativa respecto al april2

    return



if __name__ == '__main__':
    main_camera()
    # main_camera_detect()
    # main()
    # main2()
