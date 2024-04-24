import depthai as dai
import numpy as np
import cv2
import sys
from pathlib import Path

import open3d as o3d


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
camRgb.setFps(30)
if 1: camRgb.setIspScale(2, 3)

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
xOut.setStreamName("out")


def get_camera_frame(framename: str = 'image') -> np.ndarray|None:

    with device:
        device.startPipeline(pipeline)
        
        q = device.getOutputQueue(name="out", maxSize=4, blocking=False)


        first = True
        rot = 0
        picture_counter = 0

        while device.isPipelineRunning():
            
            inMessage = q.get()
            inColor = inMessage["rgb"]
            cvColorFrame = inColor.getCvFrame()
            
            cvRGBFrame = cv2.cvtColor(cvColorFrame, cv2.COLOR_BGR2RGB)

            # in rgb
            if inColor:
                frameRGB = inColor.getCvFrame()
                cv2.imshow("depth", frameRGB)

            # teclas
            key = cv2.waitKey(1)
            if key == ord('d'):
                print('export picture ')
                directory = Path(__file__).resolve().parent.parent
                cv2.imwrite(filename=str(directory / 'assets' /  f'{framename}_{picture_counter}.png'), img=frameRGB)
                picture_counter += 1
            
            if key == ord('q'):

                break

            if key == ord('r'):
                cv2.destroyAllWindows()
                return frameRGB

        cv2.destroyAllWindows()
        return 
    
