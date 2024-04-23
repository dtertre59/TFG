import numpy as np
from pathlib import Path
import cv2

import depthai_functions as daif
import apriltag_functions as atf
import transformations_functions as trf

from models.camera import CameraConfig, ApriltagConfig


# ----- VARIABLES ----- # 

# CONFIG
camera_config = CameraConfig(width=1280, height=720, fx= 3008.92857, fy=3008.92857)
apriltag_config = ApriltagConfig(family='tag36h11', size=0.015)



# ROBOT POSE -> Vector6D [X, Y, Z, RX, RY, RZ] # mm, rad
APRILTAG_POSE = np.array([-0.016, -0.320, 0.017, 2.099, 2.355, -0.017])

ROBOT_BASE = np.array([0, 0, 0])


def frame_to_pos() -> list[np.ndarray]:
    frame = daif.get_camera_frame()
    if frame is None:
        print('No frame')
        return 
    
    # Aqui tenemos imagen sin analizar
    # Redimensiona la imagen utilizando la interpolación de área de OpenCV
    frame = cv2.resize(frame, (camera_config.resolution.x, camera_config.resolution.y), interpolation=cv2.INTER_AREA)
    
    # DETECION
    detector = atf.init_detector(families=apriltag_config.family)
    detections = atf.get_detections(detector, frame, camera_config, apriltag_config)

    if not detections:
        print('No apriltags detections')
        return
    
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


def move_robot_to_point(point: np.ndarray):
    pass

# ----- CODIGO ----- # 

def main():
    reference_apriltag_t, pieze_t = frame_to_pos()

    prob, pcam, pref, ppieze = pos_to_robot_points(reference_apriltag_t, pieze_t)

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

main()

