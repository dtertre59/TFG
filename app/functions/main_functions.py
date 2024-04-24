import numpy as np
from pathlib import Path
import cv2

# try:
#     import depthai_functions as daif
# except:
#     print('no camera conection')

import apriltag_functions as atf
import transformations_functions as trf
import ur3e_functions as ur3f

from models.camera import CameraConfig, ApriltagConfig


# ----- VARIABLES ----- # 

# CONFIG
camera_config = CameraConfig(width=1280, height=720, fx= 3008.92857, fy=3008.92857)
apriltag_config = ApriltagConfig(family='tag36h11', size=0.015)


ROBOT_HOST = '192.168.10.222' # "localhost"
ROBOT_PORT = 30004
robot_config_filename = config_filename = str(Path(__file__).resolve().parent.parent / 'assets' / 'ur3e' / 'configuration_1.xml')

# ROBOT POSE -> Vector6D [X, Y, Z, RX, RY, RZ] # mm, rad
ROBOT_BASE = np.array([0, 0, 0])
APRILTAG_POSE = np.array([-0.016, -0.320, 0.017, 2.099, 2.355, -0.017])
PIEZE_POSE = np.array([-0.109, -0.408, 0.070, 2.099, 2.355, -0.017])



# ----- FUNCTIONS ----- #

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
def pos_to_robot_points(t_ref_to_cam, t_pieze_to_cam):
    # puntos de origen de los sistemas de coordenadas
    pcam_cam = pref_ref = ppieze_pieze = prob_rob = np.array([0 ,0, 0])

    # puntos respecto de la camara (ref, pieze )
    pref_cam = trf.point_tansf(t_ref_to_cam, pref_ref)
    ppieze_cam = trf.point_tansf(t_pieze_to_cam, ppieze_pieze)
    
    # puntos respecto de ref (cam -> ref)
    pcam_ref = trf.point_tansf(np.linalg.inv(t_ref_to_cam), pcam_cam)
    ppieze_ref = trf.point_tansf(np.linalg.inv(t_ref_to_cam), ppieze_cam)

    # puntos respecto a la base del robot
    t_ref_to_robot = np.array([[1, 0, 0, APRILTAG_POSE[0]],
                               [0, 1, 0, APRILTAG_POSE[1]],
                               [0, 0, 1, APRILTAG_POSE[2]],
                               [0, 0, 0, 1]])
    
    pcam_rob = trf.point_tansf(t_ref_to_robot, pcam_ref)
    pref_rob = trf.point_tansf(t_ref_to_robot, pref_ref)
    ppieze_rob = trf.point_tansf(t_ref_to_robot, ppieze_ref)

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

    print('move to Apriltag pose... ', APRILTAG_POSE)
    ur3f.robot_move(con, setp, watchdog, APRILTAG_POSE)

    print('move to cuadrado... ', init_pose)
    ur3f.robot_move(con, setp, watchdog, init_pose)
    ppieze_pose = np.append(point, init_pose[3:])
    ur3f.robot_move(con, setp, watchdog, ppieze_pose)
    
    return 

# ----- CODIGO ----- # 

def main():
    
    # 1. adquirimos frame de la camara

    frame = daif.get_camera_frame()
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
    fig, ax, i = trf.init_3d_rep()
    scale = 0.5
    axis = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])  # Ejes unitarios
    trf.print_point_with_axis(ax, prob, axis , 'base robot', 'k')
    # axis_r = np.dot(t_april1_to_robot[:3, :3], axis.T).T
    trf.print_point(ax, pref, 'april ref', 'g')
    # axis_c = np.dot(t_april1_to_camera[:3,:3], axis.T)
    trf.print_point(ax, pcam, 'camara', 'b')
    # axis_pieza = np.dot(t_camera_to_april2[:3, :3], axis_c.T).T
    trf.print_point(ax, ppieze, 'pieza', 'c')
    trf.show_3d_rep(fig, ax, 'Sistema de Referencia: Robot')

    # 5. movemos robot al punto de la pieza
    # move_robot_to_point(ppieze)



def main_april_and_pointcloud():
    name = 'april_square_2_4'
    directory = fr'app\assets'

    # 1. imagen y nube de puntos
    frame = cv2.imread(fr'app\assets\pictures\april_square_2_4.png')
    pointcloud = trf.import_pointcloud(fr'app\assets\pointclouds\{name}.ply')

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

    p_center_ref = trf.pixel_to_point3d(pointcloud, resolution=[1280,720], pixel=center_ref)
    p_x_ref = trf.pixel_to_point3d(pointcloud, resolution=[1280,720], pixel=x_ref)
    p_y_ref = trf.pixel_to_point3d(pointcloud, resolution=[1280,720], pixel=y_ref)

    p_center_sq = trf.pixel_to_point3d(pointcloud, resolution=[1280,720], pixel=center_sq)
    p_x_sq = trf.pixel_to_point3d(pointcloud, resolution=[1280,720], pixel=x_sq)
    p_y_sq = trf.pixel_to_point3d(pointcloud, resolution=[1280,720], pixel=y_sq)

    ref_axis = trf.create_axis_with_points(p_center_ref, p_x_ref, p_y_ref)
    sq_axis = trf.create_axis_with_points(p_center_sq, p_x_sq, p_y_sq)

    axis = trf.create_axis(normalized=False, size=100)

    # april_ref_axis = trf.create_axis_with_points()
    # april_pieze_axis = trf.create_axis_with_points()


    cube = trf.create_cube(point=[0,0,0], size=10, color=[1,1,0])
    line = trf.create_line(point1=[0,0,0], point2=[75,0,75])
    trf.visualization([pointcloud, axis, cube, line, ref_axis, sq_axis])





# ----- PRUEBAS ----- # 

# main()
main_april_and_pointcloud()

