# first, import all necessary modules
from pathlib import Path
import threading
import cv2
import depthai
import numpy as np

from ultralytics import YOLO

# Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
pipeline = depthai.Pipeline()

# First, we want the Color camera as the output
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)  # 300x300 will be the preview frame size, available as 'preview' output of the node
cam_rgb.setInterleaved(False)


# XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
xout_rgb = pipeline.createXLinkOut()
# For the rgb camera output, we want the XLink stream to be named "rgb"
xout_rgb.setStreamName("rgb")
# Linking camera preview to XLink input, so that the frames will be sent to host
cam_rgb.preview.link(xout_rgb.input)


# Load nn model 
nn_model = YOLO('./models/yolov8n_square_v1.pt')  # pretrained YOLOv8n model

# colores para el modelo
objects_colors = {
    'circle': (0,0,255),
    'hexagon': (0,255,0),
    'scuare': (255,0,0) # va con q pero hay que cambiarlo en la red neuronal
}



def process_detections(frame):
    detections = nn_model(frame, stream=True)

    for detection in detections:
        # identificación de objetos
        objects = detection.boxes.cls.numpy().tolist()
        # diccionario de nombres
        names = detection.names
        if objects:
            # print('Nombres: ', names)
            # print('objetos: ', objects)
            # encontramos la posición del cuadrado
            # pos = ol.index(2)
            # print(pos)
            # coordenadas
            for index, object in enumerate(objects):
                # coodenadas de cada objeto
                coordinates = detection.boxes.xyxy[index].numpy().tolist()
                # nombre del objeto detectado
                object_name = names[object]
                # color asociado
                object_color = objects_colors[object_name]
                # Escribir el nombre encima del rectángulo
                cv2.putText(frame, object_name, (int(coordinates[0]), int(coordinates[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, object_color, 2)
                # Dibujar el rectángulo en la imagen
                cv2.rectangle(img=frame, pt1=(int(coordinates[0]), int(coordinates[1])),
                            pt2=(int(coordinates[2]), int(coordinates[3])), color=object_color, thickness=2)
    return frame
            


def main():

    # Pipeline is now finished, and we need to find an available device to run our pipeline
    # we are using context manager here that will dispose the device after we stop using it
    with depthai.Device(pipeline) as device:
        # From this point, the Device will be in "running" mode and will start sending data via XLink

        # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
        q_rgb = device.getOutputQueue("rgb",  maxSize=4, blocking=False)

        # Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
        frame = None

        # Main host-side application loop
        while True:
            # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
            in_rgb = q_rgb.tryGet()

            if in_rgb is not None:
                # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
                frame = in_rgb.getCvFrame()
                # frame_resize = cv2.resize(frame, (300,int(frame.shape[0]*300/frame.shape[1])))

            if frame is not None:
                cv2.imshow("preview1", frame)
                # detección de objetos en la funcion. devuelve el frame pintado
                frame = process_detections(frame)
                # resultado
                cv2.imshow("previewDetected", frame)
            # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
            if cv2.waitKey(1) == ord('q'):
                break


if __name__=='__main__':
    main()