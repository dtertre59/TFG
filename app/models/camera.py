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



import numpy as np
import time
import depthai as dai
import cv2

from models.vectors import Vector2D


        

class CameraConfig():
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
    def __init__(self, width: int, height: int, fx: float, fy: float) -> None:
        super().__init__(width, height, fx, fy)

        self.pipeline = None
        self.device = None

    # INIT camera RGB
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
        # try:
        #     calibData = self.device.readCalibration2()
        #     lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
        #     if lensPosition:
        #         camRgb.initialControl.setManualFocus(lensPosition)
        # except:
        #     raise


        camRgb.isp.link(sync.inputs["rgb"])
        sync.out.link(xOut.input)
        xOut.setStreamName("rgb out")

        return
    
    # RUN camera
    def run_with_condition(self, trigger_func = None, detector = None) -> np.ndarray|None:
        start_time = time.time()
        with dai.Device(self.pipeline) as self.device:
            # For now, RGB needs fixed focus to properly align with depth.
            # This value was used during calibration
            # try:
            #     calibData = self.device.readCalibration2()
            #     lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
            #     if lensPosition:
            #         camRgb.initialControl.setManualFocus(lensPosition)
            # except:
            #     raise
            # self.device.startPipeline(self.pipeline) # esta dentro del with
            
            q = self.device.getOutputQueue(name="rgb out", maxSize=4, blocking=False)

            detections_bool = False

            while self.device.isPipelineRunning():
                
                inMessage = q.get()
                inColor = inMessage["rgb"]
                cvColorFrame = inColor.getCvFrame()
                
                cvRGBFrame = cv2.cvtColor(cvColorFrame, cv2.COLOR_BGR2RGB)

                # ----- in rgb
                if inColor:
                    frame = inColor.getCvFrame()
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # frame = cv2.resize(frame, (1280, 720))

                    # trigger function
                    if trigger_func:
                            frame, detections_bool, pieces = trigger_func(frame, self, detector)
                    
                    
                    cv2.imshow("rgb", frame)

                    if detections_bool and ((time.time()-start_time)>8): # ponemos 20 sergundos de enfoque
                        cv2.destroyAllWindows()
                        return frame, pieces
                    

                # ----- teclas
                key = cv2.waitKey(1)
                
                if key == ord('q'):

                    break

            cv2.destroyAllWindows()
            return
        

    

    # PRUEBAS

# camera = Camera(width=1280, height=720, fx= 998.911548, fy=998.2517088)
# camera.init_rgb()
# while 1:
#     camera.run_with_condition()
