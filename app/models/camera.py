
import numpy as np

import depthai as dai
import cv2

from .vectors import Vector2D


        

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
        self.device = dai.Device()

        camRgb = self.pipeline.create(dai.node.ColorCamera)

        sync = self.pipeline.create(dai.node.Sync)
        xOut = self.pipeline.create(dai.node.XLinkOut)
        xOut.input.setBlocking(False)


        # Properties RGB CAM
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

        camRgb.setFps(15)
        # if 1: camRgb.setIspScale(2, 3)

        # For now, RGB needs fixed focus to properly align with depth.
        # This value was used during calibration
        try:
            calibData = self.device.readCalibration2()
            lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
            if lensPosition:
                camRgb.initialControl.setManualFocus(lensPosition)
        except:
            raise


        camRgb.isp.link(sync.inputs["rgb"])
        sync.out.link(xOut.input)
        xOut.setStreamName("rgb out")

        return
    
    # RUN camera
    def run_with_condition(self, trigger_func = None, detector = None) -> np.ndarray|None:

        with self.device:
            self.device.startPipeline(self.pipeline)
            
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
                    frame = cv2.resize(frame, (1280, 720))

                    # trigger function
                    if trigger_func:
                            frame, detections_bool, pieces = trigger_func(frame, self, detector)
                    
                    cv2.imshow("rgb", frame)

                    if detections_bool:
                        cv2.destroyAllWindows()
                        return frame, pieces
                    

                # ----- teclas
                key = cv2.waitKey(10)
                
                if key == ord('q'):

                    break

            cv2.destroyAllWindows()
            return
        

    