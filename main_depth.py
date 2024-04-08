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
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)  # 300x300 will be the preview frame size, available as 'preview' output of the node
cam_rgb.setInterleaved(False)

monoLeft = pipeline.create(depthai.node.MonoCamera)
monoRight = pipeline.create(depthai.node.MonoCamera)

depth = pipeline.create(depthai.node.StereoDepth)

# # Better handling for occlusions:
# depth.setLeftRightCheck(False)
# # Closer-in minimum depth, disparity range is doubled:
# depth.setExtendedDisparity(False)
# # Better accuracy for longer distance, fractional disparity 32-levels:
# depth.setSubpixel(False)

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


# setting node configs
depth.setDefaultProfilePreset(depthai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Align depth map to the perspective of RGB camera, on which inference is done
depth.setDepthAlign(depthai.CameraBoardSocket.CAM_A)
depth.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())


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
            frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)
            frame = inDepth.getCvFrame()
            # pintar frame
            cv2.imshow("depth", frame)
        # esperar a que se cierre con la tecla q
        if cv2.waitKey(1) == ord('q'):
            # Cerrar todas las ventanas
            cv2.destroyAllWindows()
            break        