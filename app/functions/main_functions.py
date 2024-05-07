"""
        main_functions.py

    Este Script contiene una union de los conceptos basicos en funcciones más especificas.
    Teniendo en cuenta los parametros que tenemos en nuestro proyecto en específico.

    Parámetros de la camara OAK-D LITE:
        - Resoluciones: 
            . 13MP = 4208x3120 -> (fx = , fy = )
            . 4K = 3840x2160 -> (fx = 2996.7346441158315, fy = 2994.755126405525)
            . FULL HD = 1920x1080 -> (fx = 1498.367322, fy = 1497.377563)
            . 720P = 1280x720 -> (fx = 998.911548, fy = 998.2517088)
"""

# -------------------- PACKAGES ------------------------------------------------------------------------------------------ #
import os
import sys
from pathlib import Path
# Obtener la ruta del directorio del script principal
dir_principal = Path(__file__).resolve().parent.parent
# Añadir el directorio del script principal al sys.path
sys.path.append(str(dir_principal))

import numpy as np
from pathlib import Path
import cv2

import functions.depthai_functions as daif
import functions.helper_functions as hf
import functions.ur3e_functions as ur3f

import functions.apriltag_functions as atf

from models.myCamera import MyCameraConfig
from models.detection import ApriltagConfig



# -------------------- VARIABLES ----------------------------------------------------------------------------------------- #

# CAMERA Config
# camera_config = CameraConfig(width=3840, height=2160, fx= 2996.7346441158315, fy=2994.755126405525) 
# camera_config = CameraConfig(width=1920, height=1080, fx= 1498.367322, fy=1497.377563) 
camera_config = MyCameraConfig(width=1280, height=720, fx= 998.911548, fy=998.2517088)

apriltag_config = ApriltagConfig(family='tag36h11', size=0.015)


ROBOT_HOST = '192.168.10.222' # "localhost"
ROBOT_PORT = 30004
robot_config_filename = config_filename = str(Path(__file__).resolve().parent.parent / 'assets' / 'ur3e' / 'configuration_1.xml')

# ROBOT POSE -> Vector6D [X, Y, Z, RX, RY, RZ] # mm, rad
ROBOT_BASE = np.array([0, 0, 0])
APRILTAG_POSE = np.array([-0.016, -0.320, 0, 2.099, 2.355, -0.017])

# PIEZAS
PIEZE_WIDTH = 30
PIEZE_HEIGHT = 59.86 # mm



PIEZE_POSE = np.array([-0.109, -0.408, 0.070, 2.099, 2.355, -0.017])



# -------------------- FUNCTIONS ----------------------------------------------------------------------------------------- #



#FUNCIONA BIEN ->  CAMERA CON DETECCION APRILTAGS
def init_camera_and_visualize(trigger_frame_func = None):
    pipeline, device = daif.init_camera_rgb()
    detector = atf.init_detector()
    frame = daif.run_camera(pipeline, device, trigger_frame_func, detector)
    return frame

    





# HAY QUE3 HACERLO BIEN. DEVUELVE UNA DETECCION -> center, corners, transformation matrix
# def init_camera_with_detections(trigger_frame_func = None):
#     pipeline, device = daif.init_camera_rgb()
#     detection = daif.run_camera(pipeline, device, trigger_frame_func)
#     return detection


# def init_camera_visualize_trigger_detection(frame):
#     # ESTUDIAR DECORADORES
#     def trigger_function_nn(frame):
#         pass
#     def trigger_function_apriltags():
#         pass
#     # 1. es necesario enviar una funcion que se ejecute en la otra funcion analizando cada frame y si se cumple que lance un trigger o devuelva el frame detectado
#     frame = init_camera_and_visualize(trigger_function_nn)
    
#     pass



def detections_with_apriltags(detector, frame: np.ndarray):
    # Aqui tenemos imagen sin analizar
    # Redimensiona la imagen utilizando la interpolación de área de OpenCV
    frame = cv2.resize(frame, (camera_config.resolution.x, camera_config.resolution.y), interpolation=cv2.INTER_AREA) 
    # DETECION
    # detector = atf.init_detector(families=apriltag_config.family)
    detections = atf.get_detections(detector, frame, camera_config, apriltag_config)

    if not detections:
        print('No apriltags detections')
        return frame, False
    
    # paint detectins
    for detection in detections:
        atf.paint_apriltag(frame, detection)
    
    return frame, True




def frame_to_apriltag_detections(frame: np.ndarray):
    # Aqui tenemos imagen sin analizar
    # Redimensiona la imagen utilizando la interpolación de área de OpenCV
    frame = cv2.resize(frame, (camera_config.resolution.x, camera_config.resolution.y), interpolation=cv2.INTER_AREA) 
    # DETECION
    detector = atf.init_detector(families=apriltag_config.family)
    detections = atf.get_detections(detector, frame, camera_config, apriltag_config)
    if not detections:
        print('No apriltags detections')
        return
    return detections


# imagen -> matrices de transformacion
def frame_to_pos(frame: np.ndarray) -> list[np.ndarray]:
    
    detections = frame_to_apriltag_detections(frame)
    
    # detections
    reference_apriltag = None
    for detection in detections:
        # encontramos apriltag de referencia con el robot: tag_id = 1 (conocemos su posicion respecto a la base del robot)
        if detection.tag_id == 4:
            reference_apriltag = detection
        # paint apriltags in the frame
        atf.paint_apriltag(frame, detection)

    # Mostrar la imagen con el rectángulo y el centro marcados
    cv2.imshow('AprilTag', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if reference_apriltag is None:
        print ('No apriltag reference')
        return

    reference_apriltag_t = atf.get_transformation_matrix(reference_apriltag)
    pieze_t = atf.get_transformation_matrix(detections[-1])

    return reference_apriltag_t, pieze_t


# matrizes de transformacion de los puntos a la camara -> puntos respecto del rovbot
def pos_to_camera_points(ref_detection, pieze_detection):
    t_ref_to_cam = atf.get_transformation_matrix(ref_detection)
    t_pieze_to_cam = atf.get_transformation_matrix(pieze_detection)


    # puntos de origen de los sistemas de coordenadas
    pcam_cam = pref_ref = ppieze_pieze = prob_rob = np.array([0 ,0, 0])

    # puntos respecto de la camara (ref, pieze )
    pref_cam = hf.point_tansf(t_ref_to_cam, pref_ref)
    ppieze_cam = hf.point_tansf(t_pieze_to_cam, ppieze_pieze)
    
    # puntos respecto de ref (cam -> ref)
    pcam_ref = hf.point_tansf(np.linalg.inv(t_ref_to_cam), pcam_cam)
    ppieze_ref = hf.point_tansf(np.linalg.inv(t_ref_to_cam), ppieze_cam)

    # puntos respecto a la base del robot
    t_ref_to_robot = np.array([[1, 0, 0, APRILTAG_POSE[0]],
                               [0, 1, 0, APRILTAG_POSE[1]],
                               [0, 0, 1, APRILTAG_POSE[2]],
                               [0, 0, 0, 1]])
    
    pcam_rob = hf.point_tansf(t_ref_to_robot, pcam_ref)
    pref_rob = hf.point_tansf(t_ref_to_robot, pref_ref)
    ppieze_rob = hf.point_tansf(t_ref_to_robot, ppieze_ref)

    prob_ref = hf.point_tansf(np.linalg.inv(t_ref_to_robot), prob_rob)
    prob_cam = hf.point_tansf(t_ref_to_cam, prob_ref)

    return pref_cam, ppieze_cam, prob_cam, pcam_cam, t_ref_to_cam, t_pieze_to_cam, t_ref_to_robot

# sacar el sistema de ref del april
def pos_to_ref_points(t_ref_to_cam, t_pieze_to_cam):
    # puntos de origen de los sistemas de coordenadas
    pcam_cam = pref_ref = ppieze_pieze = prob_rob = np.array([0 ,0, 0])

    # puntos respecto de la camara (ref, pieze )
    pref_cam = hf.point_tansf(t_ref_to_cam, pref_ref)
    ppieze_cam = hf.point_tansf(t_pieze_to_cam, ppieze_pieze)
    
    # puntos respecto de ref (cam -> ref)
    pcam_ref = hf.point_tansf(np.linalg.inv(t_ref_to_cam), pcam_cam)
    ppieze_ref = hf.point_tansf(np.linalg.inv(t_ref_to_cam), ppieze_cam)

    # puntos respecto a la base del robot
    t_ref_to_robot = np.array([[1, 0, 0, APRILTAG_POSE[0]],
                               [0, 1, 0, APRILTAG_POSE[1]],
                               [0, 0, 1, APRILTAG_POSE[2]],
                               [0, 0, 0, 1]])
    
    pcam_rob = hf.point_tansf(t_ref_to_robot, pcam_ref)
    pref_rob = hf.point_tansf(t_ref_to_robot, pref_ref)
    ppieze_rob = hf.point_tansf(t_ref_to_robot, ppieze_ref)

    prob_ref = hf.point_tansf(np.linalg.inv(t_ref_to_robot), prob_rob)

    return pref_ref, ppieze_ref, prob_ref, pcam_ref, t_ref_to_cam, t_pieze_to_cam, t_ref_to_robot

def pieze_point_to_robot_ref(t_ref_to_cam, t_pieze_to_cam):
    # puntos de origen de los sistemas de coordenadas
    pcam_cam = pref_ref = ppieze_pieze = prob_rob = np.array([0 ,0, 0])

    # puntos respecto de la camara (ref, pieze )
    pref_cam = hf.point_tansf(t_ref_to_cam, pref_ref)
    ppieze_cam = hf.point_tansf(t_pieze_to_cam, ppieze_pieze)
    
    # puntos respecto de ref (cam -> ref)
    pcam_ref = hf.point_tansf(np.linalg.inv(t_ref_to_cam), pcam_cam)
    ppieze_ref = hf.point_tansf(np.linalg.inv(t_ref_to_cam), ppieze_cam)

    # puntos respecto a la base del robot
    t_ref_to_robot = np.array([[1, 0, 0, APRILTAG_POSE[0]],
                               [0, 1, 0, APRILTAG_POSE[1]],
                               [0, 0, 1, APRILTAG_POSE[2]],
                               [0, 0, 0, 1]])
    
    pcam_rob = hf.point_tansf(t_ref_to_robot, pcam_ref)
    pref_rob = hf.point_tansf(t_ref_to_robot, pref_ref)
    ppieze_rob = hf.point_tansf(t_ref_to_robot, ppieze_ref)

    prob_ref = hf.point_tansf(np.linalg.inv(t_ref_to_robot), prob_rob)


    # t_pieze_to_cam -> t_cam_to_ref -> t_ref_to_robot
    # t_pieze_to_ref = np.dot(np.linalg.inv(t_ref_to_cam), t_pieze_to_cam)

    t_pieze_to_ref = np.dot(np.linalg.inv(t_ref_to_cam), t_pieze_to_cam)
    t_pieze_to_robot = np.dot(t_ref_to_robot,t_pieze_to_ref)

    return t_pieze_to_robot


# matrizes de transformacion de los puntos a la camara -> puntos respecto del rovbot
def pos_to_robot_points(t_ref_to_cam, t_pieze_to_cam):
    # puntos de origen de los sistemas de coordenadas
    pcam_cam = pref_ref = ppieze_pieze = prob_rob = np.array([0 ,0, 0])

    # puntos respecto de la camara (ref, pieze )
    pref_cam = hf.point_tansf(t_ref_to_cam, pref_ref)
    ppieze_cam = hf.point_tansf(t_pieze_to_cam, ppieze_pieze)
    
    # puntos respecto de ref (cam -> ref)
    pcam_ref = hf.point_tansf(np.linalg.inv(t_ref_to_cam), pcam_cam)
    ppieze_ref = hf.point_tansf(np.linalg.inv(t_ref_to_cam), ppieze_cam)

    # puntos respecto a la base del robot
    t_ref_to_robot = np.array([[1, 0, 0, APRILTAG_POSE[0]],
                               [0, 1, 0, APRILTAG_POSE[1]],
                               [0, 0, 1, APRILTAG_POSE[2]],
                               [0, 0, 0, 1]])
    
    pcam_rob = hf.point_tansf(t_ref_to_robot, pcam_ref)
    pref_rob = hf.point_tansf(t_ref_to_robot, pref_ref)
    ppieze_rob = hf.point_tansf(t_ref_to_robot, ppieze_ref)

    return prob_rob, pcam_rob, pref_rob, ppieze_rob



# Movimientos del robot para llegar al punto de coger la pieza
def move_robot_to_point(point: np.ndarray):
    con = ur3f.connect_robot(host=ROBOT_HOST, port=ROBOT_PORT)
    setp, watchdog, gripper = ur3f.setup_robot(con=con, config_file=config_filename)

    init_pose = APRILTAG_POSE + np.array([0,0,0.120,0,0,0])
    print('moviendo robot posicion inicial... ', init_pose)
    # 0. robot en posicion inicial
    ur3f.robot_move(con, setp, watchdog, init_pose)
    ur3f.gripper_control(con,gripper=gripper,gripper_on=False)

    april_pose = APRILTAG_POSE + np.array([0,0,0.005,0,0,0])
    print('move to Apriltag pose... ', april_pose)
    ur3f.robot_move(con, setp, watchdog, april_pose)

    
    ur3f.robot_move(con, setp, watchdog, init_pose)
    ppieze_pose = np.append(point, init_pose[3:])
    ppieze_pose_i = ppieze_pose
    ppieze_pose_i[2] = init_pose[2]
    print('move to cuadrado... ', ppieze_pose_i)
    ur3f.robot_move(con, setp, watchdog, ppieze_pose_i)
    
    return 


# -------------------- TRAINNING ----------------------------------------------------------------------------------------- #

def main():
    
    # 1. adquirimos frame de la camara
    p, q = daif.init_camera_rgb()
    frame = daif.get_camera_frame(p, q)
    if frame is None:
        print('No frame')
        return 
    
    # 1. adquirimos imagen descargada si no estamos utilizando la camara
    # frame = cv2.imread(str(Path(__file__).resolve().parent.parent / 'assets' / 'apriltags_1.png'), cv2.IMREAD_COLOR)


    # 2. matrices de transformacion
    reference_apriltag_t, pieze_t = frame_to_pos(frame)

    # 3. puntos respecto al robot
    prob, pcam, pref, ppieze = pos_to_robot_points(reference_apriltag_t, pieze_t)

    # 4. mostramos puntos segun el sistema ref del robot
    print(prob)
    print(pcam)
    print(pref)
    print(ppieze)
    fig, ax, i = hf.init_3d_rep()
    scale = 0.5
    axis = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])  # Ejes unitarios
    hf.print_point_with_axis(ax, prob, axis , 'base robot', 'k')
    # axis_r = np.dot(t_april1_to_robot[:3, :3], axis.T).T
    hf.print_point(ax, pref, 'april ref', 'g')
    # axis_c = np.dot(t_april1_to_camera[:3,:3], axis.T)
    hf.print_point(ax, pcam, 'camara', 'b')
    # axis_pieza = np.dot(t_camera_to_april2[:3, :3], axis_c.T).T
    hf.print_point(ax, ppieze, 'pieza', 'c')
    hf.show_3d_rep(fig, ax, 'Sistema de Referencia: Robot')

    # 5. movemos robot al punto de la pieza
    move_robot_to_point(ppieze)


def main_simple_april_camref():
    name = 'april_square_2_4'

    # 1. frame and pointcloud
    frame = cv2.imread(str(Path(__file__).resolve().parent.parent / 'assets' / 'pictures' / f'{name}.png'))

    # ----- APRILTAG DETECTIONS + OPENCV + FRAME ----- # 

    #2.  detections
    detections = frame_to_apriltag_detections(frame) 
    if not detections:
        print('No detections')
        return
    
    # 2.1 paint frame
    reference_apriltag = None
    for detection in detections:
        # encontramos apriltag de referencia con el robot: tag_id = 4 (conocemos su posicion respecto a la base del robot)
        if detection.tag_id == 4:
            reference_apriltag = detection
        else:
            square_apriltag = detection
        # paint apriltags in the frame
        atf.paint_apriltag(frame, detection)

    # reference_apriltag = square_apriltag

    # Show frame
    cv2.imshow('AprilTag', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Points REFEREMCE APRIL SISTEM --------------------------------
    
    t_ref_to_cam = atf.get_transformation_matrix(reference_apriltag)

    pref_cam = hf.point_tansf(t_ref_to_cam, [0,0,0])
    pcam_cam = np.array([0,0,0])

    # cam_axes = np.eye(3, 3) # matriz identidad
    size = 0.2
    cam_axes = np.array([[size, 0, 0],
                         [0, size, 0],
                         [0, 0, size]])
    ref_axes = np.dot(t_ref_to_cam[:3, :3], cam_axes.T).T


    # ----- MATPLOT ------------------ # 

    fig, ax = hf.init_mat3d()

    hf.add_point_with_axes(ax, pcam_cam, cam_axes, 'camera', 'b')
    hf.add_point_with_axes(ax, pref_cam, ref_axes, 'ref', 'r')





    hf.show_mat3d(fig, ax, 'apriltags representation')

    return 

def main_simple_april_ref():
    name = 'april_square_2_4'

    # 1. frame and pointcloud
    frame = cv2.imread(str(Path(__file__).resolve().parent.parent / 'assets' / 'pictures' / f'{name}.png'))

    # ----- APRILTAG DETECTIONS + OPENCV + FRAME ----- # 

    #2.  detections
    detections = frame_to_apriltag_detections(frame) 
    if not detections:
        print('No detections')
        return
    
    # 2.1 paint frame
    reference_apriltag = None
    for detection in detections:
        # encontramos apriltag de referencia con el robot: tag_id = 4 (conocemos su posicion respecto a la base del robot)
        if detection.tag_id == 4:
            reference_apriltag = detection
        else:
            square_apriltag = detection
        # paint apriltags in the frame
        atf.paint_apriltag(frame, detection)

    # reference_apriltag = square_apriltag

    # Show frame
    cv2.imshow('AprilTag', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Points REFEREMCE APRIL SISTEM --------------------------------
    
    t_ref_to_cam = atf.get_transformation_matrix(reference_apriltag)
    t_cam_to_ref = np.linalg.inv(t_ref_to_cam)

    pref_ref = np.array([0,0,0])
    pcam_ref = hf.point_tansf(t_cam_to_ref, [0,0,0])

    # cam_axes = np.eye(3, 3) # matriz identidad
    size = 0.2
    ref_axes = np.array([[size, 0, 0],
                         [0, size, 0],
                         [0, 0, size]])
    cam_axes = np.dot(t_cam_to_ref[:3, :3], ref_axes.T).T

    # ----- MATPLOT ------------------ # 

    fig, ax = hf.init_mat3d()

    hf.add_point_with_axes(ax, pcam_ref, cam_axes, 'camera', 'b')
    hf.add_point_with_axes(ax, pref_ref, ref_axes, 'ref', 'r')

    hf.show_mat3d(fig, ax, 'apriltags representation')

    return 

def main_simple_april_camref_ref_pluspieze():
    name = 'april_square_2_4'

    # 1. frame and pointcloud
    frame = cv2.imread(str(Path(__file__).resolve().parent.parent / 'assets' / 'pictures' / f'{name}.png'))

    # ----- APRILTAG DETECTIONS + OPENCV + FRAME ----- # 

    #2.  detections
    detections = frame_to_apriltag_detections(frame) 
    if not detections:
        print('No detections')
        return
    
    # 2.1 paint frame
    reference_apriltag = None
    for detection in detections:
        # encontramos apriltag de referencia con el robot: tag_id = 4 (conocemos su posicion respecto a la base del robot)
        if detection.tag_id == 4:
            reference_apriltag = detection
        else:
            square_apriltag = detection
        # paint apriltags in the frame
        atf.paint_apriltag(frame, detection)


    # Show frame
    cv2.imshow('AprilTag', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Points REFEREMCE APRIL SISTEM --------------------------------

    t_ref_to_cam = atf.get_transformation_matrix(reference_apriltag)
    t_cam_to_ref = np.linalg.inv(t_ref_to_cam)

    t_pieze_to_cam = atf.get_transformation_matrix(square_apriltag)
    t_cam_to_pieze = np.linalg.inv(t_pieze_to_cam)

    t_pieze_to_ref = np.dot(t_cam_to_ref, t_pieze_to_cam)
    t_ref_to_pieze = np.linalg.inv(t_pieze_to_ref)

    pcam_cam = np.array([0,0,0])
    pref_cam = hf.point_tansf(t_ref_to_cam, [0,0,0])
    ppieze_cam = hf.point_tansf(t_pieze_to_cam, [0,0,0])

    size = 0.2
    cam_axes = np.array([[size, 0, 0],
                         [0, size, 0],
                         [0, 0, size]])
    ref_axes = np.dot(t_ref_to_cam[:3, :3], cam_axes.T).T
    ppieze_axes = np.dot(t_pieze_to_cam[:3, :3], cam_axes.T).T

    # ----- MATPLOT ------------------ # 

    fig, ax = hf.init_mat3d()

    hf.add_point_with_axes(ax, pcam_cam, cam_axes, 'camera', 'b')
    hf.add_point_with_axes(ax, pref_cam, ref_axes, 'ref', 'r')
    hf.add_point_with_axes(ax, ppieze_cam, ppieze_axes, 'square', 'k')

    hf.show_mat3d(fig, ax, 'apriltags representation')

    return 

# ESTE ES EL BUENO
def main_simple_april_ref_pluspieze():
    name = 'april_square_2_4'

    # 1. frame and pointcloud
    frame = cv2.imread(str(Path(__file__).resolve().parent.parent / 'assets' / 'pictures' / f'{name}.png'))

    # ----- APRILTAG DETECTIONS + OPENCV + FRAME ----- # 

    #2.  detections
    detections = frame_to_apriltag_detections(frame) 
    if not detections:
        print('No detections')
        return
    
    # 2.1 paint frame
    reference_apriltag = None
    for detection in detections:
        # encontramos apriltag de referencia con el robot: tag_id = 4 (conocemos su posicion respecto a la base del robot)
        if detection.tag_id == 4:
            reference_apriltag = detection
        else:
            square_apriltag = detection
        # paint apriltags in the frame
        atf.paint_apriltag(frame, detection)


    # Show frame
    cv2.imshow('AprilTag', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Points REFEREMCE APRIL SISTEM --------------------------------
    
    t_ref_to_cam = atf.get_transformation_matrix(reference_apriltag)
    t_cam_to_ref = np.linalg.inv(t_ref_to_cam)

    t_pieze_to_cam = atf.get_transformation_matrix(square_apriltag)
    t_cam_to_pieze = np.linalg.inv(t_pieze_to_cam)

    t_pieze_to_ref = np.dot(t_cam_to_ref, t_pieze_to_cam)
    t_ref_to_pieze = np.linalg.inv(t_pieze_to_ref)

    pref_ref = np.array([0,0,0])
    pcam_ref = hf.point_tansf(t_cam_to_ref, [0,0,0])

    ppieze_ref = hf.point_tansf(t_pieze_to_ref, [0,0,0])

    # cam_axes = np.eye(3, 3) # matriz identidad
    size = 0.2
    ref_axes = np.array([[size, 0, 0],
                         [0, size, 0],
                         [0, 0, size]])
    cam_axes = np.dot(t_cam_to_ref[:3, :3], ref_axes.T).T

    ppieze_axes = np.dot(t_pieze_to_ref[:3, :3], ref_axes.T).T
    # ----- MATPLOT ------------------ # 

    fig, ax = hf.init_mat3d()

    hf.add_point_with_axes(ax, pcam_ref, cam_axes, 'camera', 'b')
    hf.add_point_with_axes(ax, pref_ref, ref_axes, 'ref', 'r')
    hf.add_point_with_axes(ax, ppieze_ref, ppieze_axes, 'square', 'k')
    

    hf.show_mat3d(fig, ax, 'apriltags representation')

    return 

# BIEN TAMBIEN
def main_april():
    name = 'april_square_2_4'

    # 1. frame and pointcloud
    frame = cv2.imread(str(Path(__file__).resolve().parent.parent / 'assets' / 'pictures' / f'{name}.png'))

    # ----- APRILTAG DETECTIONS + OPENCV + FRAME ----- # 

    #2.  detections
    detections = frame_to_apriltag_detections(frame) 
    if not detections:
        print('No detections')
        return
    
    # 2.1 paint frame
    reference_apriltag = None
    for detection in detections:
        # encontramos apriltag de referencia con el robot: tag_id = 4 (conocemos su posicion respecto a la base del robot)
        if detection.tag_id == 4:
            reference_apriltag = detection
        else:
            square_apriltag = detection
        # paint apriltags in the frame
        atf.paint_apriltag(frame, detection)

    # Show frame
    cv2.imshow('AprilTag', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Points REFEREMCE APRIL SISTEM --------------------------------
    p_ref, p_pieze, p_rob, p_cam, t_ref_to_cam, t_pieze_to_cam, t_ref_to_robot  = pos_to_ref_points(atf.get_transformation_matrix(reference_apriltag), atf.get_transformation_matrix(square_apriltag))

    t_ref_to_cam = t_ref_to_cam

    t_cam_to_pieze = np.linalg.inv(t_pieze_to_cam)

    t_pieze_to_ref = np.dot(np.linalg.inv(t_ref_to_cam), t_pieze_to_cam)


    # cam_axes = np.eye(3, 3) # matriz identidad
    size = 0.2
    ref_axes = np.array([[size, 0, 0],
                         [0, size, 0],
                         [0, 0, size]])
    cam_axes = np.dot(np.linalg.inv(t_ref_to_cam)[:3, :3], ref_axes.T).T
    pieze_axes = np.dot(t_pieze_to_ref[:3,:3], ref_axes.T).T
    rob_axes = np.dot(np.linalg.inv(t_ref_to_robot)[:3,:3],ref_axes.T).T

    # ----- MATPLOT ------------------ # 

    fig, ax = hf.init_mat3d()

    hf.add_point_with_axes(ax, p_cam, cam_axes, 'camera', 'b')
    hf.add_point_with_axes(ax, p_ref, ref_axes, 'ref', 'r')
    hf.add_point_with_axes(ax, p_pieze, pieze_axes, 'pieze', 'g')
    hf.add_point_with_axes(ax, p_rob, rob_axes, 'robot', 'k')




    hf.show_mat3d(fig, ax, 'apriltags representation')

    return 

def main_april_v2():
    name = 'april_square_2_4'

    # 1. frame and pointcloud
    frame = cv2.imread(str(Path(__file__).resolve().parent.parent / 'assets' / 'pictures' / f'{name}.png'))

    # ----- APRILTAG DETECTIONS + OPENCV + FRAME ----- # 

    #2.  detections
    detections = frame_to_apriltag_detections(frame) 
    if not detections:
        print('No detections')
        return
    
    # 2.1 paint frame
    reference_apriltag = None
    for detection in detections:
        # encontramos apriltag de referencia con el robot: tag_id = 4 (conocemos su posicion respecto a la base del robot)
        if detection.tag_id == 4:
            reference_apriltag = detection
        else:
            square_apriltag = detection
        # paint apriltags in the frame
        atf.paint_apriltag(frame, detection)

    # Show frame
    cv2.imshow('AprilTag', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Points REFEREMCE APRIL SISTEM --------------------------------
    t_ref_to_cam = atf.get_transformation_matrix(reference_apriltag)
    t_pieze_to_cam = atf.get_transformation_matrix(square_apriltag)

    t_pieze_to_robot  = pieze_point_to_robot_ref(t_ref_to_cam, t_pieze_to_cam)

    t_cam_to_robot = np.dot( t_pieze_to_robot, np.linalg.inv(t_pieze_to_cam))
    
    t_ref_to_robot = np.dot(t_cam_to_robot, t_ref_to_cam)
    
    ppieze = hf.point_tansf(t_pieze_to_robot, [0,0,0])
    pcam = hf.point_tansf(t_cam_to_robot, [0,0,0])
    pref = hf.point_tansf(t_ref_to_robot, [0,0,0])
    probot = np.array([0,0,0])

    # cam_axes = np.eye(3, 3) # matriz identidad
    size = 0.2
    robot_axes = np.array([[size, 0, 0],
                         [0, size, 0],
                         [0, 0, size]])
    cam_axes = np.dot(t_cam_to_robot[:3, :3], robot_axes.T).T
    pieze_axes = np.dot(t_pieze_to_robot[:3,:3], robot_axes.T).T
    ref_axes = robot_axes

    # ----- MATPLOT ------------------ # 

    fig, ax = hf.init_mat3d()

    hf.add_point_with_axes(ax, pcam, cam_axes, 'camera', 'b')
    hf.add_point_with_axes(ax, pref, ref_axes, 'ref', 'r')
    hf.add_point_with_axes(ax, ppieze, pieze_axes, 'pieze', 'g')
    hf.add_point_with_axes(ax, probot, robot_axes, 'robot', 'k')

    hf.show_mat3d(fig, ax, 'apriltags representation')

    return 


def main_april_and_pointcloud():
    name = 'april_square_2_4'

    # 1. frame and pointcloud
    frame = cv2.imread(str(Path(__file__).resolve().parent.parent / 'assets' / 'pictures' / f'{name}.png'))
    pointcloud = hf.import_pointcloud(str(Path(__file__).resolve().parent.parent / 'assets' / 'pointclouds' / f'{name}.ply'))
    pointcloud = hf.invert_pointcloud(pointcloud)
    # ----- APRILTAG DETECTIONS + OPENCV + FRAME ----- # 

    #2.  detections
    detections = frame_to_apriltag_detections(frame) 
    if not detections:
        print('No detections')
        return
    
    # 2.1 paint frame
    reference_apriltag = None
    for detection in detections:
        # encontramos apriltag de referencia con el robot: tag_id = 4 (conocemos su posicion respecto a la base del robot)
        if detection.tag_id == 4:
            reference_apriltag = detection
        else:
            square_apriltag = detection
        # paint apriltags in the frame
        atf.paint_apriltag(frame, detection)

    # referencia 
    center_ref, x_ref, y_ref = atf.get_center_x_y_axis(reference_apriltag)
    # square
    center_sq, x_sq, y_sq = atf.get_center_x_y_axis(square_apriltag)

    # Show frame
    cv2.imshow('AprilTag', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Points --------------------------------
    p_ref, p_pieze, p_rob, p_cam, t_ref_to_cam, t_pieze_to_cam, t_ref_to_robot  = pos_to_camera_points(reference_apriltag, square_apriltag)

    t_cam_to_ref = np.linalg.inv(t_ref_to_cam)
    t_cam_to_pieze = np.linalg.inv(t_pieze_to_cam)
    t_cam_to_rob = np.dot(t_cam_to_ref, t_ref_to_robot)

    # cam_axes = np.eye(3, 3) # matriz identidad
    size = 0.2
    cam_axes = np.array([[size, 0, 0],
                         [0, size, 0],
                         [0, 0, size]])
    ref_axes = np.dot(t_cam_to_ref[:3, :3], cam_axes.T).T
    pieze_axes = np.dot(t_cam_to_pieze[:3,:3], cam_axes.T).T
    rob_axes = np.dot(t_cam_to_rob[:3,:3], cam_axes.T).T

    # ----- MATPLOT ------------------ # 

    fig, ax = hf.init_mat3d()

    hf.add_point_with_axes(ax, p_cam, cam_axes, 'camera', 'b')
    hf.add_point_with_axes(ax, p_ref, ref_axes, 'ref', 'r')
    hf.add_point_with_axes(ax, p_pieze, pieze_axes, 'pieze', 'g')
    hf.add_point_with_axes(ax, p_rob, rob_axes, 'robot', 'k')




    hf.show_mat3d(fig, ax, 'apriltags representation')


    # ----- OPEN 3D SHOW + CLOUD ----- # 

    # el eje de la camara esta rotado 180 grados respecto a erl de la nube de puntos

    p_center_ref = hf.pixel_to_point3d(pointcloud, resolution=[1280,720], pixel=center_ref)
    p_x_ref = hf.pixel_to_point3d(pointcloud, resolution=[1280,720], pixel=x_ref)
    p_y_ref = hf.pixel_to_point3d(pointcloud, resolution=[1280,720], pixel=y_ref)

    p_center_sq = hf.pixel_to_point3d(pointcloud, resolution=[1280,720], pixel=center_sq)
    p_x_sq = hf.pixel_to_point3d(pointcloud, resolution=[1280,720], pixel=x_sq)
    p_y_sq = hf.pixel_to_point3d(pointcloud, resolution=[1280,720], pixel=y_sq)



    axis = hf.create_axes(size=100)

    ref_axis = hf.create_2D_axes_with_points(p_center_ref, p_x_ref, p_y_ref)
    sq_axis = hf.create_2D_axes_with_points(p_center_sq, p_x_sq, p_y_sq)

    # april_ref_axis = hf.create_2D_axes_with_points()
    # april_pieze_axis = hf.create_2D_axes_with_points()
    
    # Z 180degrees
    rot = np.array([[-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1],])

    p_cam = np.dot(rot, p_cam)
    p_rob = np.dot(rot, p_rob)
    p_ref = np.dot(rot, p_ref)
    p_pieze = np.dot(rot, p_pieze)

    size = np.array([20,20,20])
    robot = hf.create_cube(p_rob*1000, size = size, color=[1,1,0])
    cam = hf.create_cube(p_cam*1000, size = size, color=[0,0,1])
    ref = hf.create_cube(p_ref*1000, size = size)
    square = hf.create_cube(p_pieze*1000, size = size)

    # cube = hf.create_cube(point=[0,0,0], size=[10,10,10], color=[1,1,0])
    # line = hf.create_line(point1=[0,0,0], point2=[75,0,75])

    hf.o3d_visualization([pointcloud, axis, ref_axis, sq_axis, robot, cam, ref, square])

    return



# ----- PRUEBAS ----- # 

# main()
# main_simple_april_camref()
# main_simple_april_ref()
# main_simple_april_camref_ref_pluspieze()
# main_simple_april_ref_pluspieze()
# main_april()
# main_april_v2()
# main_april_and_pointcloud()

