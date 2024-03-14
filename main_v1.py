"""
author: David Tertre
objetos utilizados:
    - Robot Ur3e
    - Piezas hechas con la impresora 3d
    - camara OAK-D LITE
Objetivos:
    1. Visualizar las piezas
    2. Conocer la posicion de las piezas por orden de izquierda a derecha
    3. movimientos del robot para coger las piezas y dejarlas en posiciones conocidas
"""
import os
import sys
from pathlib import Path
import cv2
# import depthai
import numpy as np

from ultralytics import YOLO

# ----- VARIABLES -----
# Obtener la ruta actual
# current_directory = Path.cwd()
current_directory = Path().resolve()
current_directory = os.getcwd()
current_directory = os.path.abspath(path=sys.argv[0])
current_directory = Path(sys.argv[0]).resolve()
current_directory = Path(__file__)

parent_directory = current_directory.parent

print("The current directory is:", current_directory)
print("The parent directory is:", parent_directory)

# Load nn model 
model_path = Path(sys.argv[0]).resolve().parent / 'pretrained_models'/'yolov8n_square_v1.pt'
print("model path: ",model_path)
nn_model = YOLO(model_path)  # pretrained YOLOv8n model
# colores para el modelo
objects_colors = {
    'circle': (0,0,255),
    'hexagon': (0,255,0),
    'scuare': (255,0,0) # va con q pero hay que cambiarlo en la red neuronal
}


# ----- FUNCTIONS -----

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
