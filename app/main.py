"""
        main.py

    Main script

"""

# -------------------- PACKAGES ------------------------------------------------------------------------------------------ #

import cv2
from pathlib import Path

from functions import main_functions as mf

from functions.models.robot import Robot
from functions.models.myCamera import MyCamera
from functions.models.myApriltag import MyApriltag
from functions.models.detection import DetectionsCoordinator




# -------------------- VARIABLES ----------------------------------------------------------------------------------------- #

ROBOT_HOST = '192.168.10.222' # "localhost"
ROBOT_PORT = 30004
robot_config_filename = config_filename = str(Path(__file__).resolve().parent / 'assets' / 'ur3e' / 'configuration_1.xml')

p_init = [0.128, -0.298, 0.180, 3.1415, 0.2617, 0]

# -------------------- FUNCTIONS ----------------------------------------------------------------------------------------- #

# Con apriltags
def main():
    # # 1. conexion con el robot (verificar con algun registro)
    # robot = Robot(ROBOT_HOST, ROBOT_PORT, robot_config_filename)
    # robot.connect()
    # robot.setup()
    # # 1.1 mover robot a posicion inicial o de reposo (verificar que el prorama esta en funcionamiento en la tablet)
    # robot.move(p_init)

    # 2. (thread) visualizar con la camara el area donde se encuentran las piezas (el robot en reposo ya apunta a este area)
    apriltag = MyApriltag(family='tag36h11', size=0.015)

    camera = MyCamera(width=1280, height=720, fx= 998.911548, fy=998.2517088)
    camera.init_rgb()
    

    # 2.1 adquirimos frame (es necesario que se vea el apriltag de ref) -> podriamos que devolviera directamente la deteccion para no tener que analizar la imagen otra vez
    frame = camera.run_with_condition(DetectionsCoordinator.apriltag_detections, apriltag)

    cv2.imshow('a',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
    # 2.2 deteccion de apriltags con las funciones de la libreria apriltags. Sabemos la pieza que es porque tiene un tag_id conocido.

    # 2.3 ubicar centro del april de las piezas como punto 3d respecto a la base del robot (matrices de transferencia). Importante la rotacion de la pieza

    new_pose = [0.128, -0.298, 0.180, 3.1415, 0.2617, 0]
    # 3. movimiento del robot para conger la pieza y dejarla en su respectivo hoyo (posicion conocida)

    # 4. repetimos en bucle hasta que no haya mas piezas

    # 5. (opcional) revisar que los hoyos no esten ocupados para poder mover la pieza a su hoyo

    # 6 .(opcional) poner un apriltag en el madero de los hoyos y asi no es necesario saber la posicion exacta de cada hoyo, solo la relativa respecto al april2

    return


    

# Con red neuronal
def main2():
    # 1. conexion con el robot (verificar con algun registro)

    # 1.1 mover robot a posicion inicial o de reposo

    # 2. (thread) visualizar con la camara el area donde se encuentran las piezas (el robot en reposo ya apunta a este area)

    # 2.1 adquirimos frame (es necesario que se vea el apriltag de ref)

    # 2.2 deteccion de la pieza con red neuronal. Pose estimation con la red. por lo que concocemos el centro de la cara superior de la pieza

    # 2.3 pasamos pixel (centro) a punto en 3d de la nube de puntos (respecto de la camara)

    # 2.4 pasamos el punto 3d respecto de la camara al sistema de ref del robot utilizando el apriltag de referencia

    # 3. movimiento del robot para conger la pieza y dejarla en su respectivo hoyo (posicion conocida)
    

    return



if __name__ == '__main__':
    main()
    # main2()