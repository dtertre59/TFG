#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs
import time

# Create pipeline
pipeline = dai.Pipeline()


# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
# detectionNetwork = pipeline.create(dai.node.MobileNetDetectionNetwork)
# spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
depth = pipeline.create(dai.node.StereoDepth)
# objectTracker = pipeline.create(dai.node.ObjectTracker)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)
# xoutNn = pipeline.create(dai.node.XLinkOut)
# xoutSNn = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)
# xoutTracker = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutLeft.setStreamName("left")
xoutRight.setStreamName("right")
# xoutNn.setStreamName("nn")
# xoutSNn.setStreamName("snn")
xoutDepth.setStreamName("depth")
# xoutTracker.setStreamName("tracker")


# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR) # no se para que sirve

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.setCamera("left")

monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setCamera("right")

# detectionNetwork.setBlobPath(blobconverter.from_zoo(name='face-detection-retail-0004', shaves=6))
# detectionNetwork.setConfidenceThreshold(0.5)

# spatialDetectionNetwork.setBlobPath(blobconverter.from_zoo(name='face-detection-retail-0004', shaves=6))
# spatialDetectionNetwork.setConfidenceThreshold(0.5)
# spatialDetectionNetwork.input.setBlocking(False)
# spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
# spatialDetectionNetwork.setDepthLowerThreshold(100)
# spatialDetectionNetwork.setDepthUpperThreshold(5000)

# setting node configs
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Align depth map to the perspective of RGB camera, on which inference is done
depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)
depth.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

# objectTracker.setDetectionLabelsToTrack([15])  # track only person
# # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
# objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
# objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)


# Link
camRgb.preview.link(xoutRgb.input)
monoLeft.out.link(xoutLeft.input)
monoRight.out.link(xoutRight.input)

# camRgb.preview.link(detectionNetwork.input)
# detectionNetwork.out.link(xoutNn.input)

# camRgb.preview.link(spatialDetectionNetwork.input)

# stereo.depth.link(spatialDetectionNetwork.inputDepth)

monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(xoutDepth.input)

# objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
# objectTracker.out.link(xoutTracker.input)

# spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
# spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
# spatialDetectionNetwork.out.link(objectTracker.inputDetections)
# depth.depth.link(spatialDetectionNetwork.inputDepth)

# helper function to convert these <0..1> values into actual pixel posion
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)



# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the grayscale frames from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)
    # qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    # qTracker = device.getOutputQueue("tracker", 4, False)

    # startTime = time.monotonic()
    # counter = 0
    # fps = 0
    # color = (255, 255, 255)

    # fill up 
    frame = None
    detections = []

    while True:
        # Instead of get (blocking), we use tryGet (non-blocking) which will return the available data or None otherwise
        inRgb = qRgb.tryGet()
        inLeft = qLeft.tryGet()
        inRight = qRight.tryGet()
        # inNn = qNn.tryGet()

        inDepth = qDepth.tryGet()

        # inTracker = qTracker.tryGet()

        # counter+=1
        # current_time = time.monotonic()
        # if (current_time - startTime) > 1 :
        #     fps = counter / (current_time - startTime)
        #     counter = 0
        #     startTime = current_time

        # if inRgb is not None and inTracker is not None:
        #     frame = inRgb.getCvFrame()
        #     trackletsData = inTracker.tracklets

        #     for t in trackletsData:
        #         roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
        #         x1 = int(roi.topLeft().x)
        #         y1 = int(roi.topLeft().y)
        #         x2 = int(roi.bottomRight().x)
        #         y2 = int(roi.bottomRight().y)

        #         try:
        #             label = t.label
        #         except:
        #             label = t.label

        #         cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        #         cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        #         cv2.putText(frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        #         cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        #         cv2.putText(frame, f"X: {int(t.spatialCoordinates.x)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        #         cv2.putText(frame, f"Y: {int(t.spatialCoordinates.y)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        #         cv2.putText(frame, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        #     cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

        #     cv2.imshow("tracker", frame)


        # if inRgb is not None:
        #     frame = inRgb.getCvFrame()
        #     cv2.imshow("rgb", inRgb.getCvFrame())

        if inLeft is not None:
            cv2.imshow("left", inLeft.getCvFrame())

        if inRight is not None:
            cv2.imshow("right", inRight.getCvFrame())

        # if inNn is not None:
        #     detections = inNn.detections
        # if frame is not None:
        #     for detection in detections:
        #         bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        #         cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        #     cv2.imshow("preview", frame)


        if inDepth is not None:
            frame = inDepth.getCvFrame()
        if frame is not None:
            # frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)
            cv2.imshow("depth", frame)
            # otra pantalla con cambio de colores
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            cv2.imshow("depth_2", frame)
            frame = None

        if cv2.waitKey(1) == ord('q'):
            break

