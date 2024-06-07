"""
        main.py

    Main script

"""

# -------------------- PACKAGES ------------------------------------------------------------------------------------------ #

import cv2
from pathlib import Path
import numpy as np
import open3d as o3d


from functions import helper_functions as hf


from models.camera import Camera
from models.robot import Robot
from models.detection import Apriltag, YoloObjectDetection, YoloPoseEstimation
from models.coordinator import Coordinator

# QUITAR
from models.piece import PieceA, Piece
from models.constants import CameraCte, RobotCte


# -------------------- VARIABLES ----------------------------------------------------------------------------------------- #

ROBOT_HOST = '192.168.10.222' # "localhost"
ROBOT_PORT = 30004
robot_config_filename = config_filename = str(Path(__file__).resolve().parent / 'assets' / 'ur3e' / 'configuration_1.xml')


# -------------------- FUNCTIONS ----------------------------------------------------------------------------------------- #

# TEST Camera
def main_camera():
    # camera = Camera(width=3840, height=2160, fx = 2996.7346441158315, fy = 2994.755126405525)
    camera = Camera(width=1920, height=1080, fx= 1498.367322, fy=1497.377563)
    # camera = Camera(width=1280, height=720, fx= 1498.367322, fy=1497.377563)
    
    camera.init_rgb()
    # camera.init_rgb_and_pointcloud()
    
    camera.run_with_options()
    # camera.run_with_options(name='irregular', crop_size=640)
    # camera.run_with_pointcloud(show3d=False)

# TEST Camera and Detections
def main_camera_detect():
    camera = Camera(width=1920, height=1080, fx= 1498.367322, fy=1497.377563)
    apriltag = Apriltag(family='tag36h11', size=0.015)
    nn_od_model = YoloObjectDetection(filename=str(Path(__file__).resolve().parent / 'assets' / 'nn_models' /'yolov8n_od_v1.pt'))
    nn_pose_model = YoloPoseEstimation(filename=str(Path(__file__).resolve().parent / 'assets' / 'nn_models' /'yolov8s_pose_irr_v2.pt'))
    
    
    camera.init_rgb()
    # camera.init_rgb_and_pointcloud()


    # camera.run_with_condition(Coordinator.apriltag_detections, apriltag, paint_frame = True)
    # camera.run_with_condition(Coordinator.nn_object_detections,  nn_od_model, paint_frame = True)
    # camera.run_with_condition(Coordinator.nn_poseEstimation_detections,  nn_pose_model, paint_frame = True)

    # camera.run_with_pointcloud_with_condition(show3d=False, trigger_func=Coordinator.apriltag_detections, apriltag=apriltag, paint_frame = True)
    
    
    try:
        # camera.run_with_condition(Coordinator.detections, nn_od_model, apriltag, paint_frame = True)
        # r_kwargs = camera.run_with_pointcloud_with_condition(show3d=
        # False, trigger_func=Coordinator.detections, nn_model=nn_pose_model, apriltag=apriltag, paint_frame = True)
        r_kwargs = camera.run_with_condition(trigger_func=Coordinator.detections, nn_model=nn_od_model, apriltag=apriltag, combine_pieces=False,paint_frame = True)

    except Exception as e:
        print('salida de camara: ',str(e))
        return
    
    # 1. para el modo 2 encesitamos sacar la pointcloud. o por lo menos el punto 3d del centro de la pieza

    ref = r_kwargs['ref']
    pieces: list[Piece] = r_kwargs['pieces']
    frame = r_kwargs['frame']
    
    square = None
    hexagon = None
    circle = None


    for p in pieces:
        if p.name == 'square':
            square = p
        if p.name == 'hexagon':
            hexagon = p
        if p.name == 'circle':
            circle = p

    # print()
    # print(square)
    # square.calculate_center_and_corners(frame)
    # for corner in square.corners:
    #     cv2.circle(frame, (corner[0],corner[1]), 3, 0, -1)  
    # cv2.circle(frame, (square.center[0],square.center[1]), 3, color=(0,0,255), thickness=-1)

    # print()
    # print(hexagon)
    # hexagon.calculate_center_and_corners(frame)
    # for corner in hexagon.corners:
    #     cv2.circle(frame, (corner[0],corner[1]), 3, 0, -1)  
    # cv2.circle(frame, (hexagon.center[0],hexagon.center[1]), 3, (0,0,255), thickness=-1)
    if square:
        print()
        print(square)
        square.calculate_center_and_corners(frame)
        square.paint(frame)

    if circle:
        print()
        print(circle)
        circle.calculate_center_and_corners(frame)
        circle.paint(frame)



    cv2.imshow('a', cv2.resize(frame, (1280, 720)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # print(ref)
    # for piece in r_kwargs['pieces']:
    #     print(piece)

    # hf.o3d_visualization([pointcloud])


# CALIBRATE pointcloud
def main_camera_calibrate_pointcloud():
    camera = Camera(width=1920, height=1080, fx= 1498.367322, fy=1497.377563)
    
    apriltag = Apriltag(family='tag36h11', size=0.015)

    camera.init_rgb_and_pointcloud()

    r_kwargs = camera.run_with_pointcloud_with_condition(show3d=False, trigger_func=Coordinator.apriltag_detections, apriltag=apriltag, paint_frame=True)

    pointcloud = r_kwargs['pointcloud']
    frame = r_kwargs['frame']

    refs: list[PieceA]= r_kwargs['pieces']
    
    def transf_to_cloud(pieceA: PieceA) -> np.ndarray:
        t_april_to_cam = pieceA.T
        pref_cam = hf.point_tansf(t_april_to_cam, np.array([0 ,0, 0]))
        # en m
        pref_cam_pointcloud = pref_cam.copy()
        # hacemos espejo por culpa del sistemade ref
        rot = hf.rotation_matrix_z(0)
        pref_cam_pointcloud = hf.point_tansf(rot, pref_cam_pointcloud)
        pref_cam_pointcloud *= 1000 # se pasa a mm
        return pref_cam_pointcloud
    

    ps_pointcloud = np.empty((0, 3))
    ps_april = np.empty((0, 3))
    
    for ref in refs:
        ps_pointcloud = np.vstack((ps_pointcloud, hf.pixel_to_point3d(pointcloud=pointcloud, resolution=np.array([1920,1080]), pixel=ref.center.get_array())))
        ps_april = np.vstack((ps_april, transf_to_cloud(ref)))

    cubesm = []
    cubesb = []

    print('ahora: ', ps_pointcloud)
    print('corregido: ', ps_april)

    T = hf.procrustes_method(ps_pointcloud.T, ps_april.T, verbose=True)
    # T = CameraCte.T_pointcloud_to_good_pointcloud

    points = np.asarray(pointcloud.points)
    new_points = [hf.point_tansf(T, point) for point in points]
    pointcloud.points = o3d.utility.Vector3dVector(new_points)

    for p_pointcloud in ps_pointcloud:
        p_pointcloud = hf.point_tansf(T, p_pointcloud)
        cubesm.append(hf.create_cube(point=p_pointcloud, size = [5,5,5], color = np.array([0,0,1])))

    for p_april in ps_april:
        cubesb.append(hf.create_cube(point=p_april, size = [5,5,5], color = np.array([0,1,0])))

    cubes = cubesm + cubesb


    axes = hf.create_axes_with_lineSet()

    geometries = [pointcloud,axes] + cubes

    hf.o3d_visualization(geometries=geometries)


# MAIN V1. Red neuronal object-detection | apriltags
def main():
    print()
    print(' ----- Init ----- ')
    print()

    # 1. Instancias
    robot = Robot(ROBOT_HOST, ROBOT_PORT, robot_config_filename)
    # camera = Camera(width=3840, height=2160, fx= 2996.7346441158315, fy=2994.755126405525) 
    camera = Camera(width=1920, height=1080, fx= 1498.367322, fy=1497.377563) 
    # camera = Camera(width=1280, height=720, fx= 998.911548, fy=998.2517088)
    camera_params = [camera.f.x, camera.f.y, camera.c.x, camera.c.y]
    apriltag = Apriltag(family='tag36h11', size=0.015, camera_params=camera_params)
    nn_od_model = YoloObjectDetection(filename=str(Path(__file__).resolve().parent / 'assets' / 'nn_models' /'yolov8n_od_v1.pt'))
    # Tolerancia
    tolerance = 15

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
        flag = Coordinator.the_whole_process(robot, camera, apriltag, nn_od_model, tolerance=tolerance)
        if not flag:
            break

    print()
    print(' ----- End----- ')
    print()

    return


# MAIN V2. Red neuronal pose-estimation | pointcloud | apriltag (referencia)
def main2():
    print()
    print(' ----- Init ----- ')
    print()

    # 1. Instancias
    robot = Robot(ROBOT_HOST, ROBOT_PORT, robot_config_filename)
    # camera = Camera(width=3840, height=2160, fx= 2996.7346441158315, fy=2994.755126405525) 
    camera = Camera(width=1920, height=1080, fx= 1498.367322, fy=1497.377563) 
    # camera = Camera(width=1280, height=720, fx= 998.911548, fy=998.2517088)
    apriltag = Apriltag(family='tag36h11', size=0.015)
    nn_pose_model = YoloPoseEstimation(filename=str(Path(__file__).resolve().parent / 'assets' / 'nn_models' /'yolov8s_pose_v5.pt'))  

    # 2. Init
    print('Inicio: ')
    try:
        # 2.1 robot
        robot.connect()
        robot.setup()
        # 2.2 camera
        camera.init_rgb_and_pointcloud()
        flag = True
    except Exception as e:
        print(str(e))
        flag = False

    
    # 3. Proceso completo de una pieza:
    while flag:
        flag = Coordinator.the_whole_process_2(robot, camera, apriltag, nn_pose_model)
        if not flag:
            break

    print()
    print(' ----- End----- ')
    print()

    return

    
    # 5. (opcional) revisar que los hoyos no esten ocupados para poder mover la pieza a su hoyo

    # 6 .(opcional) poner un apriltag en el madero de los hoyos y asi no es necesario saber la posicion exacta de cada hoyo, solo la relativa respecto al april2


# MAIN V3. Red neuronal object-detection | vision artificial clásica | pointcloud | apriltag (referencia)
def main3():
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
        robot.move(RobotCte.POSE_STANDAR)
        robot.gripper_control(False)
        # 2.2 camera
        camera.init_rgb_and_pointcloud()
        flag = True
    except Exception as e:
        print(str(e))
        flag = False

    
    # 3. Proceso completo de una pieza:
    while flag:
        flag = Coordinator.the_whole_process_3(robot, camera, apriltag, nn_od_model)
        if not flag:
            break

    print()
    print(' ----- End----- ')
    print()

    return


if __name__ == '__main__':
    # correccion_error_nube()
    # main_camera()
    # main_camera_detect()
    # main_camera_calibrate_pointcloud()
    main()
    # main2()
    # main3()
