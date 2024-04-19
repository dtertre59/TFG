#!/usr/bin/env python3

import cv2
import depthai as dai
import time

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
aprilTag = pipeline.create(dai.node.AprilTag)
manip = pipeline.create(dai.node.ImageManip)

xoutAprilTag = pipeline.create(dai.node.XLinkOut)
xoutAprilTagImage = pipeline.create(dai.node.XLinkOut)

xoutAprilTag.setStreamName("aprilTagData")
xoutAprilTagImage.setStreamName("aprilTagImage")

# Properties
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

manip.initialConfig.setResize(480, 270)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)

aprilTag.initialConfig.setFamily(dai.AprilTagConfig.Family.TAG_36H11)

# Linking
aprilTag.passthroughInputImage.link(xoutAprilTagImage.input)
camRgb.video.link(manip.inputImage)
manip.out.link(aprilTag.inputImage)
aprilTag.out.link(xoutAprilTag.input)
# always take the latest frame as apriltag detections are slow
aprilTag.inputImage.setBlocking(False)
aprilTag.inputImage.setQueueSize(1)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the mono frames from the outputs defined above
    manipQueue = device.getOutputQueue("aprilTagImage", 8, False)
    aprilTagQueue = device.getOutputQueue("aprilTagData", 8, False)

    color = (100, 100, 0)
    colorx = (255, 0, 0)
    colory = (0, 255, 0)
    colorz = (0, 0, 255)


    startTime = time.monotonic()
    counter = 0
    fps = 0

    while(True):
        inFrame = manipQueue.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        monoFrame = inFrame.getFrame()
        frame = cv2.cvtColor(monoFrame, cv2.COLOR_GRAY2BGR)

        aprilTagData = aprilTagQueue.get().aprilTags
        for aprilTag in aprilTagData:
            # puntos para pintar el sistema de referencia
            # Eje x -> botom left, botom right
            # Eje y -> botom left, top left
            # Eje z -> hacia arriba

            topLeft = aprilTag.topLeft
            topRight = aprilTag.topRight
            bottomRight = aprilTag.bottomRight
            bottomLeft = aprilTag.bottomLeft

            center = (int((topLeft.x + bottomRight.x) / 2), int((topLeft.y + bottomRight.y) / 2))
            x_axis = (int((topRight.x + bottomRight.x)/2), int((topRight.y + bottomRight.y)/2))
            y_axis = (int((topLeft.x + topRight.x)/2), int((topLeft.y + topRight.y)/2))

            # cv2.line(frame, (int(topLeft.x), int(topLeft.y)), (int(topRight.x), int(topRight.y)), colorz, 2, cv2.LINE_AA, 0)
            # cv2.line(frame, (int(topRight.x), int(topRight.y)), (int(bottomRight.x), int(bottomRight.y)), colory, 2, cv2.LINE_AA, 0)
            # Eje x
            colorx = (0, 0, 255) # rojo
            cv2.line(frame, center, x_axis, colorx, 2, cv2.LINE_AA, 0)
            # Eje y
            colory = (0, 255, 0) # verde
            cv2.line(frame, center, y_axis, colory, 2, cv2.LINE_AA, 0)

            idStr = "ID: " + str(aprilTag.id)
            cv2.putText(frame, idStr, center, cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            
            print(f'Coordendas\nOrigen: ({bottomLeft.x},{bottomLeft.y}\nEje x: ({bottomRight.x},{bottomRight.y}\nEje y: ({topLeft.x},{topLeft.y}')
            cv2.imshow('Imagen con eje de coordenadas', frame)

        cv2.putText(frame, "Fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))

        cv2.imshow("April tag frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break