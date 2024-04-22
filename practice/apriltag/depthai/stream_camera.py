import cv2
import depthai as dai
import time
from pathlib import Path


# def init_cam():
    # Create pipeline
pipeline = dai.Pipeline()
device = dai.Device()
queueNames = []


# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)

rgbOut = pipeline.create(dai.node.XLinkOut)

rgbOut.setStreamName("rgb")
queueNames.append("rgb")

# Properties
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

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


# Linking
camRgb.isp.link(rgbOut.input)


# return pipeline, device





# device, pipeline = init_cam()




# Connect to device and start pipeline
with device:
    device.startPipeline(pipeline)

    frameRgb = None
    cv2.namedWindow('rgb')

    while True:
        latestPacket = {}
        latestPacket["rgb"] = None

        queueEvents = device.getQueueEvents(("rgb"))
        for queueName in queueEvents:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1] 

        if latestPacket["rgb"] is not None:
            frameRgb = latestPacket["rgb"].getCvFrame()
            cv2.imshow("rgb", frameRgb)

            
            key = cv2.waitKey(1) 
            if key == ord('q'):
                break
            elif key == ord('d'):
                directory = Path(__file__).resolve().parent.parent
                image_directory = directory / 'assets'
                cv2.imwrite(filename=str(image_directory / 'apriltags_6.png'), img=frameRgb)



            frameRgb = None
            frameDisp = None
            frameC = None
