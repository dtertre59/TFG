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

import numpy as np
from pathlib import Path
import cv2

import depthai_functions as daif
import apriltag_functions as atf
import helper_functions as hf
import ur3e_functions as ur3f

from models.camera import CameraConfig, ApriltagConfig



# -------------------- VARIABLES ----------------------------------------------------------------------------------------- #

# CAMERA Config
camera_config = CameraConfig(width=3840, height=2160, fx= 2996.7346441158315, fy=2994.755126405525) 
# camera_config = CameraConfig(width=1920, height=1080, fx= 1498.367322, fy=1497.377563) 
# camera_config = CameraConfig(width=1280, height=720, fx= 998.911548, fy=998.2517088)

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

    return prob_cam, pcam_cam, pref_cam, ppieze_cam


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



def main_april_and_pointcloud():
    name = 'april_square_2_4'

    # 1. imagen y nube de puntos
    frame = cv2.imread(str(Path(__file__).resolve().parent.parent / 'assets' / 'pictures' / f'{name}.png'))
    pointcloud = hf.import_pointcloud(str(Path(__file__).resolve().parent.parent / 'assets' / 'pointclouds' / f'{name}.ply'))

    # ----- APRILTAG DETECTIONS + OPENCV + FRAME ----- # 

    # 2. pixels de las coordenadas de los sistemas de ref
    detections = frame_to_apriltag_detections(frame)

    # detections

    reference_apriltag = None
    for detection in detections:
        # encontramos apriltag de referencia con el robot: tag_id = 1 (conocemos su posicion respecto a la base del robot)
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
    
    p_rob, p_cam, p_ref, p_pieze = pos_to_camera_points(reference_apriltag, square_apriltag)

    distance_ref_pieze = hf.points_distance(p_ref, p_pieze)
    distance_rob_ref = hf.points_distance(p_rob, p_ref)
    distance_cam_ref = hf.points_distance(p_cam, p_ref)

    print('April distances:')
    print('ref to pieze: ', distance_ref_pieze)
    print('rob to ref: ', distance_rob_ref)
    print('camera to ref', distance_cam_ref)
    print()

    # ----- OPEN 3D SHOW + CLOUD ----- # 

    p_center_ref = hf.pixel_to_point3d(pointcloud, resolution=[1280,720], pixel=center_ref)
    p_x_ref = hf.pixel_to_point3d(pointcloud, resolution=[1280,720], pixel=x_ref)
    p_y_ref = hf.pixel_to_point3d(pointcloud, resolution=[1280,720], pixel=y_ref)

    p_center_sq = hf.pixel_to_point3d(pointcloud, resolution=[1280,720], pixel=center_sq)
    p_x_sq = hf.pixel_to_point3d(pointcloud, resolution=[1280,720], pixel=x_sq)
    p_y_sq = hf.pixel_to_point3d(pointcloud, resolution=[1280,720], pixel=y_sq)

    distance_ref_pieze = hf.points_distance(p_center_ref, p_center_sq)
    distance_rob_ref = hf.points_distance(ROBOT_BASE[:3], APRILTAG_POSE[:3])
    distance_cam_ref = hf.points_distance([0,0,0], p_center_ref)

    print('Cloud distances:')
    print('ref to pieze: ', distance_ref_pieze)
    print('rob to ref: ', distance_rob_ref)
    print('camera to ref', distance_cam_ref)
    print()



    ref_axis = hf.create_2D_axes_with_points(p_center_ref, p_x_ref, p_y_ref)
    sq_axis = hf.create_2D_axes_with_points(p_center_sq, p_x_sq, p_y_sq)

    axis = hf.create_axes(normalized=False, size=100)

    # april_ref_axis = hf.create_2D_axes_with_points()
    # april_pieze_axis = hf.create_2D_axes_with_points()


    print()
    print('De las matrices de transformacion de los apriltags:')
    print('puntos respecto de la camara')
    print('Robot point: ', p_rob)
    print('Camera point: ', p_cam)
    print('April ref point: ', p_ref)
    print('square point: ', p_pieze)
    

    robot = hf.create_cube(p_rob, size = [50,50,50], color=[0,0,0])
    cam = hf.create_cube(p_cam, size = [50,50,50], color=[0,0,1])
    ref = hf.create_cube(p_ref, size = [50,50,50])
    square = hf.create_cube(p_pieze, size = [50,50,50])

    # cube = hf.create_cube(point=[0,0,0], size=[10,10,10], color=[1,1,0])
    # line = hf.create_line(point1=[0,0,0], point2=[75,0,75])

    hf.o3d_visualization([pointcloud, axis, ref_axis, sq_axis, robot, cam, ref, square])





# ----- PRUEBAS ----- # 

main()
# main_april_and_pointcloud()

