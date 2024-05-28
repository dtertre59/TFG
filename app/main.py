"""
        main.py

    Main script

"""

# -------------------- PACKAGES ------------------------------------------------------------------------------------------ #

import cv2
from pathlib import Path
import numpy as np
import open3d as o3d

from functions import main_functions as mf
from functions import helper_functions as hf

# from models.robot import Robot
from models.camera import Camera
from models.robot import Robot
from models.detection import Apriltag, YoloObjectDetection, YoloPoseEstimation
from models.coordinator import Coordinator

# QUITAR
from models.piece import PieceA
from models.constants import CameraCte


# -------------------- VARIABLES ----------------------------------------------------------------------------------------- #

ROBOT_HOST = '192.168.10.222' # "localhost"
ROBOT_PORT = 30004
robot_config_filename = config_filename = str(Path(__file__).resolve().parent / 'assets' / 'ur3e' / 'configuration_1.xml')


# -------------------- FUNCTIONS ----------------------------------------------------------------------------------------- #

def correccion_error_nube(P, P_prime) -> np.ndarray:
    # # Puntos en el sistema de la nube de puntos (source)
    # P = np.array([
    #     [-131.13, 53.491, 456],
    #     [25.217, -52.685, 475],
    #     [-22.446, -96.861, 503]
    # ]).T

    # # Puntos corregidos (destination)
    # P_prime = np.array([
    #     [-118.23, 50.243, 435.08],
    #     [31.449, -52.517, 465.34],
    #     [-14.35, -94.815, 488.31]
    # ]).T


    # Calcular centroides
    C_P = P.mean(axis=1).reshape(-1, 1)
    C_P_prime = P_prime.mean(axis=1).reshape(-1, 1)

    # Centrar los puntos alrededor del origen
    P_cent = P - C_P
    P_prime_cent = P_prime - C_P_prime

    # Calcular la matriz de covarianza
    H = np.dot(P_cent, P_prime_cent.T)

    # Descomposición en valores singulares (SVD)
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Calcular la matriz de rotación
    R = np.dot(V, U.T)

    # Asegurarse de que la rotación es válida
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = np.dot(V, U.T)

    # Calcular el vector de traslación
    t = C_P_prime - np.dot(R, C_P)

    # Crear la matriz de transformación completa
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()

    print("Matriz de transformación:")
    print(T)

    return T




# main para testing de camera
def main_camera():
    # camera = Camera(width=3840, height=2160, fx = 2996.7346441158315, fy = 2994.755126405525)
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
    nn_pose_model = YoloPoseEstimation(filename=str(Path(__file__).resolve().parent / 'assets' / 'nn_models' /'yolov8s_pose_irr_v2.pt'))
    
    
    camera.init_rgb()
    # camera.init_rgb_and_pointcloud()


    # camera.run_with_condition(Coordinator.apriltag_detections, apriltag, paint_frame = True)
    camera.run_with_condition(Coordinator.nn_object_detections,  nn_od_model, paint_frame = True)
    # camera.run_with_condition(Coordinator.nn_poseEstimation_detections,  nn_pose_model, paint_frame = True)

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

# PRUEBAS POINTCLOUD
def main_camera_point():
    camera = Camera(width=1920, height=1080, fx= 1498.367322, fy=1497.377563)
    
    apriltag = Apriltag(family='tag36h11', size=0.015)

    camera.init_rgb_and_pointcloud()

    r_kwargs = camera.run_with_pointcloud_with_condition(show3d=False, trigger_func=Coordinator.apriltag_detections, apriltag=apriltag, paint_frame=True)

    pointcloud = r_kwargs['pointcloud']
    frame = r_kwargs['frame']

    ref1: PieceA = r_kwargs['pieces'][0]
    ref2 : PieceA = r_kwargs['pieces'][1]



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
        # print(ps_pointcloud)
        ps_april = np.vstack((ps_april, transf_to_cloud(ref)))

    cubesm = []
    cubesb = []

    print('ahora: ', ps_pointcloud)
    print('corregido: ', ps_april)

    # T = correccion_error_nube(ps_pointcloud.T, ps_april.T)
    T = CameraCte.T_pointcloud_to_good_pointcloud

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


# MAIN V3 Con apriltags y pointcloud
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
    # main_camera_point()
    # main()
    # main2()
    main3()
