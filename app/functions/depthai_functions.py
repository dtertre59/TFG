"""
        depthai_functions.py

    Este script contiene la agrupación de funciones para la interacción de la camara OAK-D LITE.
    Utilizamos la libreria depthai, proporcionada por el fabricante (luxonis)


"""

# -------------------- PACKAGES ------------------------------------------------------------------------------------------ #

import numpy as np
import sys
from pathlib import Path
import cv2
import open3d as o3d
import depthai as dai


# -------------------- FUNCTIONS ----------------------------------------------------------------------------------------- #

# INIT camera RGB
def init_camera_rgb():
    # Create pipeline
    pipeline = dai.Pipeline()
    device = dai.Device()

    camRgb = pipeline.create(dai.node.ColorCamera)

    sync = pipeline.create(dai.node.Sync)
    xOut = pipeline.create(dai.node.XLinkOut)
    xOut.input.setBlocking(False)


    # Properties RGB CAM
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    camRgb.setFps(15)
    # if 1: camRgb.setIspScale(2, 3)

    # For now, RGB needs fixed focus to properly align with depth.
    # This value was used during calibration
    try:
        calibData = device.readCalibration2()
        lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
        if lensPosition:
            camRgb.initialControl.setManualFocus(lensPosition)
    except:
        raise


    camRgb.isp.link(sync.inputs["rgb"])
    sync.out.link(xOut.input)
    xOut.setStreamName("rgb out")

    return pipeline, device

# INIT pointcloud -------------- MEJORAR --------------- CREO QUE NO VA
def init_pointcloud(pipeline: dai.Pipeline):
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)

    depth = pipeline.create(dai.node.StereoDepth)
    pointcloud = pipeline.create(dai.node.PointCloud)
    sync = pipeline.create(dai.node.Sync)
    xOut = pipeline.create(dai.node.XLinkOut)
    xOut.input.setBlocking(False)


    # Properties MONO CAMS
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    monoLeft.setCamera("left")

    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    monoRight.setCamera("right")


    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

    depth.setLeftRightCheck(True)
    # depth.setExtendedDisparity(False)
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


    pointcloud.outputPointCloud.link(sync.inputs["pcl"])
    pointcloud.initialConfig.setSparse(False)
    sync.out.link(xOut.input)
    xOut.setStreamName("pointcloud out")

    inConfig = pipeline.create(dai.node.XLinkIn)
    inConfig.setStreamName("config")
    inConfig.out.link(pointcloud.inputConfig)


    xoutDepth = pipeline.create(dai.node.XLinkOut)
    xoutDepth.setStreamName("depth")

    depth.disparity.link(xoutDepth.input)



# LA BUENA DE RUN
def run_camera(pipeline, device, trigger_func = None, detector = None) -> np.ndarray|None:

    with device:
        device.startPipeline(pipeline)
        
        q = device.getOutputQueue(name="rgb out", maxSize=4, blocking=False)

        detections_bool = False

        while device.isPipelineRunning():
            
            inMessage = q.get()
            inColor = inMessage["rgb"]
            cvColorFrame = inColor.getCvFrame()
            
            cvRGBFrame = cv2.cvtColor(cvColorFrame, cv2.COLOR_BGR2RGB)

            # ----- in rgb
            if inColor:
                frame = inColor.getCvFrame()
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (1280, 720))

                # trigger function
                if trigger_func:
                        frame, detections_bool = trigger_func(detector, frame)
                
                cv2.imshow("rgb", frame)

                if detections_bool:
                    cv2.destroyAllWindows()
                    return frame
                

            # ----- teclas
            key = cv2.waitKey(10)
            
            if key == ord('q'):

                break

        cv2.destroyAllWindows()
        return




# CON DOWNLOAD Y MAS BOTONES
def get_camera_frame(pipeline, device, framename: str = 'image') -> np.ndarray|None:

    with device:
        device.startPipeline(pipeline)
        
        q = device.getOutputQueue(name="rgb out", maxSize=4, blocking=False)
        # q2 = device.getOutputQueue(name="pointcloud out", maxSize=4, blocking=False)
    
        # qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        first = True
        rot = 0
        picture_counter = 0

        while device.isPipelineRunning():
            
            inMessage = q.get()
            inColor = inMessage["rgb"]
            cvColorFrame = inColor.getCvFrame()
            
            cvRGBFrame = cv2.cvtColor(cvColorFrame, cv2.COLOR_BGR2RGB)

            

            # ----- in rgb
            if inColor:
                frameRGB = inColor.getCvFrame()
                cv2.imshow("rgb", frameRGB)

            # ----- in depth camera
            # inDepth = qDepth.tryGet()
            # # imagen depth
            # if inDepth is not None:
            #     frame = inDepth.getCvFrame()
            #     # pintar frame
            #     cv2.imshow("depth", frame)

            # ----- teclas
            key = cv2.waitKey(1)
            if key == ord('d') and frameRGB is not None:
                print('export picture ')
                directory = Path(__file__).resolve().parent.parent / 'assets' / 'pictures' / 'train' / 'square' / f'{framename}.{picture_counter}.png'
                print(directory)
                try:
                    cv2.imwrite(filename=str(directory), img=frameRGB)
                    # cv2.imwrite(filename='img.png', img=frameRGB)
                except Exception as e:
                    print(str(e))
                picture_counter += 1
            
            if key == ord('q'):

                break

            if key == ord('r'):
                cv2.destroyAllWindows()
                return frameRGB

        cv2.destroyAllWindows()
        return 
    

# -------------------- TRAINNING ----------------------------------------------------------------------------------------- #


# pipeline, device = init_camera_rgb()
# # init_pointcloud(pipeline)

# get_camera_frame(pipeline, device, 'square_1')
    
