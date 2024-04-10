import os
import sys
from pathlib import Path

# manejo de data
import numpy as np

# camara
import depthai
import cv2



# Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
pipeline = depthai.Pipeline()

# First, we want the Color camera as the output
cam_rgb = pipeline.create(depthai.node.ColorCamera)
# cam_rgb.setPreviewSize(300, 300)  # 300x300 will be the preview frame size, available as 'preview' output of the node
# cam_rgb.setInterleaved(False)
cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setBoardSocket(depthai.CameraBoardSocket.CAM_A)

monoLeft = pipeline.create(depthai.node.MonoCamera)
monoRight = pipeline.create(depthai.node.MonoCamera)

depth = pipeline.create(depthai.node.StereoDepth)

depth.setDefaultProfilePreset(depthai.node.StereoDepth.PresetMode.HIGH_DENSITY)
depth.initialConfig.setMedianFilter(depthai.MedianFilter.MEDIAN_OFF)

# Better handling for occlusions:
depth.setLeftRightCheck(False)
# Closer-in minimum depth, disparity range is doubled:
depth.setExtendedDisparity(True) # True -> mejorar la percepción de objetos cercanos
# # Better accuracy for longer distance, fractional disparity 32-levels:
depth.setSubpixel(False) # -> intermpolacion -> mejora en zonas distantes


# setting node configs
depth.setDefaultProfilePreset(depthai.node.StereoDepth.PresetMode.HIGH_DENSITY)
depth.initialConfig.setMedianFilter(depthai.MedianFilter.MEDIAN_OFF)
# Align depth map to the perspective of RGB camera, on which inference is done
depth.setDepthAlign(depthai.CameraBoardSocket.CAM_A)
depth.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

config = depth.initialConfig.get()
config.postProcessing.thresholdFilter.minRange = 100
config.postProcessing.thresholdFilter.maxRange = 1000
depth.initialConfig.set(config)


monoLeft.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.setCamera("left")

monoRight.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setCamera("right")


# XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
xout_rgb = pipeline.createXLinkOut()
xoutDepth = pipeline.create(depthai.node.XLinkOut)


# For the rgb camera output, we want the XLink stream to be named "rgb"
xout_rgb.setStreamName("rgb")
xoutDepth.setStreamName("depth")




# Linking camera preview to XLink input, so that the frames will be sent to host
cam_rgb.preview.link(xout_rgb.input)

monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(xoutDepth.input)

# Abrir camara
with depthai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb",  maxSize=1, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    frame = None
    # bucle de streaming
    while True:
        in_rgb = q_rgb.tryGet()
        inDepth = qDepth.tryGet()
        # RGB CAM
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
            # pintar frame
            cv2.imshow("preview", frame)
        # DEPTH CAM
        if inDepth is not None:
            frame = inDepth.getCvFrame()

            print('max disparity: ',depth.initialConfig.getMaxDisparity())
            print('confidence threshold: ',depth.initialConfig.getConfidenceThreshold())
            print('disparity [200,200]: ', frame[200,200])
            focal_length_in_pixels = 451.145
            baseline = 7.5
            disparity_in_pixels = frame[200,200]
            depth_cm = focal_length_in_pixels * baseline / disparity_in_pixels
            print('depth: ', depth_cm, ' cm')
            # pintar frame
            cv2.imshow("depth", frame)

            # calibramos el frame con la maxima disparidad
            frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

            disparity_in_pixels = frame[200,200]
            depth_cm = focal_length_in_pixels * baseline / disparity_in_pixels
            print('calibrate depth: ', depth_cm, ' cm')
            # pintar frame
            cv2.imshow("depth max disp", frame)

        # esperar a que se cierre con la tecla q
        if cv2.waitKey(1) == ord('q'):
            # Cerrar todas las ventanas
            cv2.destroyAllWindows()
            break        