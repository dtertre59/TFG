"""
        camera.py

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
from __future__ import annotations
import numpy as np
import time
import depthai as dai
import cv2
import open3d as o3d
from pathlib import Path
import argparse

from models.vectors import Vector2D
from models.piece import PieceA, Piece

from functions import helper_functions as hf


# -------------------- CLASSES ------------------------------------------------------------------------------------------- #

# -------------------- EXCEPTION ----------------------------------------------------------------------------------------- #

class CameraException(Exception):
    """Excepcion de la camara"""
    def __init__(self, msg: str):
        msg = 'Excepcion Camera: ' + msg
        super().__init__(msg)


# -------------------- CAMERA -------------------------------------------------------------------------------------------- #

class CameraConfig():
    # Constructor
    def __init__(self, width: int, height: int, fx: float, fy: float) -> None:
        """ Camera config
            Resolution: pixels
            Camera center: pixels
            Focal lenth: pixels
            """
        self.resolution = Vector2D(width, height)
        self.f = Vector2D(fx, fy)
        self.c = Vector2D(width/2, height/2)


class Camera(CameraConfig):
    # Constructor
    def __init__(self, width: int, height: int, fx: float, fy: float) -> None:
        super().__init__(width, height, fx, fy)

        self.pipeline = None
        self.device = None

    # Init rgb
    def init_rgb(self):
        # Create pipeline
        self.pipeline = dai.Pipeline()
        # self.device = dai.Device()

        camRgb = self.pipeline.create(dai.node.ColorCamera)

        sync = self.pipeline.create(dai.node.Sync)
        xOut = self.pipeline.create(dai.node.XLinkOut)
        xOut.input.setBlocking(False)


        # Properties RGB CAM
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        if self.resolution.y == 720:
            camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
        elif self.resolution.y == 1080:
            camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        elif self.resolution.y == 2160:
            camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        elif self.resolution.y == 3120:
            camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_13_MP)

        camRgb.setFps(15)
        # if 1: camRgb.setIspScale(2, 3)

        # For now, RGB needs fixed focus to properly align with depth.
        # This value was used during calibration
        try:
            calibData = dai.Device().readCalibration2()
            lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
            if lensPosition:
                camRgb.initialControl.setManualFocus(lensPosition)
        except:
            raise CameraException("Fallo en la calibracion de las lentes")

        camRgb.isp.link(sync.inputs["rgb"])
        sync.out.link(xOut.input)
        xOut.setStreamName("out")

        print('Cámara iniciada')
        return
    
    # Init rgb and pointcloud
    def init_rgb_and_pointcloud(self):

        # Ruta al archivo de calibración JSON
        calibJsonFile = str((Path(__file__).parent.parent.resolve() / 'assets' / 'oakd' / 'depthai_calibration.json'))

        # Parsear argumentos de línea de comandos
        parser = argparse.ArgumentParser()
        parser.add_argument('calibJsonFile', nargs='?', help="Path to calibration file in json", default=calibJsonFile)
        args = parser.parse_args()

        # Cargar datos de calibración
        try:
            calibData = dai.CalibrationHandler(args.calibJsonFile)
        except RuntimeError as e:
            raise RuntimeError(f"Error loading calibrtion data: {e}")
        except Exception as e:
            raise CameraException(str(e))

        # Create pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setCalibrationData(calibrationDataHandler=calibData)

        # Create nodes
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        monoRight = self.pipeline.create(dai.node.MonoCamera)

        depth = self.pipeline.create(dai.node.StereoDepth)
        pointcloud = self.pipeline.create(dai.node.PointCloud)

        sync = self.pipeline.create(dai.node.Sync)
        xOut = self.pipeline.create(dai.node.XLinkOut)
        xOut.input.setBlocking(False)


        # Properties RGB CAM
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        if self.resolution.y == 720:
            camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
        elif self.resolution.y == 1080:
            camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        elif self.resolution.y == 2160:
            camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        elif self.resolution.y == 3120:
            camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_13_MP)
        else:
            raise CameraException("Fallo en el setup de la resolucion")

        camRgb.setFps(10)

        # # For now, RGB needs fixed focus to properly align with depth.
        # # This value was used during calibration
        # try:
        #     calibData = dai.Device().readCalibration2()
        #     lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
        #     if lensPosition:
        #         camRgb.initialControl.setManualFocus(lensPosition)
        # except:
            # raise

    
        # Properties MONO CAMS
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoLeft.setCamera("left")

        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoRight.setCamera("right")


        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

        depth.setLeftRightCheck(True)
        depth.setExtendedDisparity(False)
        depth.setSubpixel(True)
        depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        config = depth.initialConfig.get()
        config.postProcessing.thresholdFilter.minRange = 100
        config.postProcessing.thresholdFilter.maxRange = 1000
        depth.initialConfig.set(config)


        # otras caracteristicas
        monoLeft.out.link(depth.left)
        monoRight.out.link(depth.right)

        depth.depth.link(pointcloud.inputDepth)

        camRgb.isp.link(sync.inputs["rgb"])
        pointcloud.outputPointCloud.link(sync.inputs["pcl"])
        pointcloud.initialConfig.setSparse(False)
        sync.out.link(xOut.input)
        xOut.setStreamName("out")

        # inConfig = self.pipeline.create(dai.node.XLinkIn)
        # inConfig.setStreamName("config")
        # inConfig.out.link(pointcloud.inputConfig)


        # xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        # xoutDepth.setStreamName("depth")
        # depth.disparity.link(xoutDepth.input)

        print('Cámara iniciada')

        return

    # Run with options
    def run_with_options(self, directory: str|None = None, name: str = 'img', crop_size: int|bool = False) -> None|dict:
        with dai.Device(self.pipeline) as self.device:
            print('Camara en funcionamiento')
            
            q = self.device.getOutputQueue(name="out", maxSize=4, blocking=False)

            # descargas
            if not directory:
                if crop_size:
                    directory = Path(__file__).resolve().parent.parent / 'assets' / 'pictures' / 'train_640' / name
                else:
                    directory = Path(__file__).resolve().parent.parent / 'assets' / 'pictures' / name

                
            picture_counter = hf.obtain_last_number(directory, name) + 1

            while self.device.isPipelineRunning():
                
                inMessage = q.get()
                inColor = inMessage["rgb"]

                # ----- in rgb
                if inColor:
                    frame = inColor.getCvFrame()
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # frame = cv2.resize(frame, (1280, 720))
                    if crop_size:
                        frame = hf.crop_frame(frame, crop_size)
                        cv2.imshow("OAK-D-Lite", frame)
                    else:
                        cv2.imshow("OAK-D-Lite", cv2.resize(frame, (1280, 720)))
                    
                    


                # ----- teclas
                key = cv2.waitKey(1)
                
                if key == ord('d') and frame is not None:
                    filename = f'{directory}/{name}_{picture_counter}.png'
                    try:
                        print('Export picture ', filename)
                        cv2.imwrite(filename=str(filename), img=frame)
                        # cv2.imwrite(filename='img.png', img=frameRGB)
                    except Exception as e:
                        raise CameraException(str(e))
                    picture_counter += 1
                if key == ord('q'):
                    break

            cv2.destroyAllWindows()
            return

    # Run with  condition
    def run_with_condition(self, trigger_func = None, *args, **kwargs) -> None|dict:
        start_time = time.time()
        with dai.Device(self.pipeline) as self.device:
            print('Camara en funcionamiento')

            q = self.device.getOutputQueue(name="out", maxSize=4, blocking=False)

            flag = False

            while self.device.isPipelineRunning():
                
                inMessage = q.get()
                inColor = inMessage["rgb"]

                # ----- in rgb
                if inColor:
                    frame = inColor.getCvFrame()

                    if trigger_func:
                            modified_frame = frame.copy()
                            results_kwargs = trigger_func(modified_frame, self, *args, **kwargs)
                            flag = results_kwargs.get('flag')
                            frame_resized = cv2.resize(modified_frame, (1280, 720))
                            cv2.imshow("OAK-D-Lite", frame_resized)

                            if flag and ((time.time()-start_time)>10): # ponemos 8 sergundos de enfoque
                                results_kwargs['frame'] = frame
                                cv2.destroyAllWindows()
                                return results_kwargs
                    

                # ----- teclas
                key = cv2.waitKey(1)

                if key == ord('q'):
                    break

            cv2.destroyAllWindows()
            return

    # Run with pointcloud   
    def run_with_pointcloud(self, show3d:bool = False):
        start_time = time.time()
        with dai.Device(self.pipeline) as self.device:
            print('Camara en funcionamiento')
 
            q = self.device.getOutputQueue(name="out", maxSize=4, blocking=False)

            # pclConfIn = self.device.getInputQueue(name="config", maxSize=4, blocking=False)
            # # depth camara
            # qDepth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            

            pointcloud = o3d.geometry.PointCloud()      
            if show3d:
                # visualizador o3d
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                vis.add_geometry(pointcloud)
                # Eje de coordenadas
                coordinateFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=400, origin=[0,0,0])
                vis.add_geometry(coordinateFrame)

            first = True
            rot = 0
            picture_counter = 0
            flag = False

            while self.device.isPipelineRunning():
                
                inMessage = q.get()
                inColor = inMessage["rgb"]
                inPointCloud = inMessage["pcl"]

                # inDepth = qDepth.tryGet()
                


                # ----- in rgb
                if inColor:
                    frame = inColor.getCvFrame()
                    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame, (1280, 720))
                    cv2.imshow("OAK-D-Lite", frame_resized)

                # ----- in depth
                # if inDepth is not None:
                #     frame = inDepth.getCvFrame()
                #     # pintar frame
                #     cv2.imshow("depth", frame)

                # ----- in pointloud
                if inPointCloud:
                    pass
                    # resolucion pixeles = 1280*720
                    # hei = inPointCloud.getHeight()
                    # wid = inPointCloud.getWidth()
                    # print('width: ', wid, '       height: ', hei)
                    # asignamos puntos
                    points = inPointCloud.getPoints().astype(np.float64)
                    # print(points[7200:14399])
                    pointcloud.points = o3d.utility.Vector3dVector(points)
                    # asignamos colores
                    colors = (frameRGB.reshape(-1, 3) / 255.0).astype(np.float64)
                    pointcloud.colors = o3d.utility.Vector3dVector(colors)
                    
                    if show3d:
                        vis.update_geometry(pointcloud)

                    # # SEGUNDO METODO
                    # pointcloud = o3d.geometry.PointCloud()
                    # pointcloud.points = o3d.utility.Vector3dVector(points)
                    # pointcloud.colors = pcd.colors
                    # # Visualizar la nube de puntos
                    # o3d.visualization.draw_geometries([pointcloud])
                if show3d:
                    vis.poll_events()
                    vis.update_renderer()

                    
                # ----- teclas ----- # 
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
        if show3d:
            vis.destroy_window()
        cv2.destroyAllWindows()
        return
    
    # Run with pointcloud with condition  
    def run_with_pointcloud_with_condition(self, show3d:bool = False, trigger_func = None, *args, **kwargs) -> None|dict:
        start_time = time.time()
        with dai.Device(self.pipeline) as self.device:
            print('Camara en funcionamiento')
 
            q = self.device.getOutputQueue(name="out", maxSize=4, blocking=False)

            # pclConfIn = self.device.getInputQueue(name="config", maxSize=4, blocking=False)
            # # depth camara
            # qDepth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            

            pointcloud = o3d.geometry.PointCloud()      
            if show3d:
                # visualizador o3d
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                vis.add_geometry(pointcloud)
                # Eje de coordenadas
                coordinateFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=400, origin=[0,0,0])
                vis.add_geometry(coordinateFrame)

            first = True
            rot = 0
            picture_counter = 0
            flag = False

            while self.device.isPipelineRunning():
                
                inMessage = q.get()
                inColor = inMessage["rgb"]
                inPointCloud = inMessage["pcl"]

                # inDepth = qDepth.tryGet()
                


                # ----- in rgb
                if inColor:
                    frame = inColor.getCvFrame()
                    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if not trigger_func:
                        frame_resized = cv2.resize(frame, (1280, 720))
                        cv2.imshow("OAK-D-Lite", frame_resized)

                # ----- in depth
                # if inDepth is not None:
                #     frame = inDepth.getCvFrame()
                #     # pintar frame
                #     cv2.imshow("depth", frame)

                # ----- in pointloud
                if inPointCloud:

                    # resolucion pixeles = 1280*720
                    # hei = inPointCloud.getHeight()
                    # wid = inPointCloud.getWidth()
                    # print('width: ', wid, '       height: ', hei)
                    # asignamos puntos
                    points = inPointCloud.getPoints().astype(np.float64)
                    # print(points[7200:14399])
                    pointcloud.points = o3d.utility.Vector3dVector(points)
                    # asignamos colores
                    colors = (frameRGB.reshape(-1, 3) / 255.0).astype(np.float64)
                    pointcloud.colors = o3d.utility.Vector3dVector(colors)
                    
                    if show3d:
                        vis.update_geometry(pointcloud)

                    # # SEGUNDO METODO
                    # pointcloud = o3d.geometry.PointCloud()
                    # pointcloud.points = o3d.utility.Vector3dVector(points)
                    # pointcloud.colors = pcd.colors
                    # # Visualizar la nube de puntos
                    # o3d.visualization.draw_geometries([pointcloud])
                if show3d:
                    vis.poll_events()
                    vis.update_renderer()


                if inColor and inPointCloud:
                     if trigger_func:
                            frame_modified = frame.copy()
                            results_kwargs = trigger_func(frame_modified, self, *args, **kwargs)
                            flag = results_kwargs.get('flag')
                            frame_resized = cv2.resize(frame_modified, (1280, 720))
                            cv2.imshow("OAK-D-Lite", frame_resized)

                            if flag and ((time.time()-start_time)>20): # ponemos 20 sergundos de enfoque
                                # añadir la nube de puntos en el retorno
                                results_kwargs['frame'] = frame
                                results_kwargs['pointcloud'] = hf.invert_pointcloud(pointcloud)
                                cv2.destroyAllWindows()
                                return results_kwargs
                    
                # ----- teclas ----- # 
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
        if show3d:
            vis.destroy_window()
        cv2.destroyAllWindows()
        return


# -------------------- TRAINNING ----------------------------------------------------------------------------------------- #

# camera = Camera(width=1280, height=720, fx= 998.911548, fy=998.2517088)
# camera.init_rgb()
# while 1:
#     camera.run_with_condition()

# TAKE PICTURES
# camera = Camera(width=1920, height=1080, fx= 1498.367322, fy=1497.377563) 
# camera.init_rgb()

# camera.run_with_options(name='square')
