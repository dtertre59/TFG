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

# manejo de data
import numpy as np

# camara
import depthai
import cv2

# neuronal network
from ultralytics import YOLO


# ----- MODEL CLASSES -----


class Piece:
    def __init__(self, name: str, coordinates: np.ndarray, color: tuple):
        self.name = name
        self.coordinates = coordinates
        self.color = color



# ----- VARIABLES -----

# Obtener la ruta actual
# current_directory = Path.cwd()
current_directory = Path().resolve()
current_directory = os.getcwd()
current_directory = os.path.abspath(path=sys.argv[0])
current_directory = Path(sys.argv[0]).resolve()
current_directory = Path(__file__)

parent_directory = current_directory.parent

# print("The current directory is:", current_directory)
# print("The parent directory is:", parent_directory)


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


objects_colors = {
        'circle': (0,0,255),
        'hexagon': (0,255,0),
        'scuare': (255,0,0) # va con q pero hay que cambiarlo en la red neuronal
    }



# ----- FUNCTIONS -----

# init Yolo model
def init_YOLO_nn_model() -> YOLO:
    # Load nn model 
    model_path = Path(sys.argv[0]).resolve().parent / 'pretrained_models'/'yolov8n_square_v1.pt'
    print("model path: ",model_path)
    nn_model = YOLO(model_path)  # pretrained YOLOv8n model
    # colores para el modelo
    return nn_model

# detectar bojetos en la imagen
def process_detections(nn_model: YOLO, frame):
    detections = nn_model(frame, stream=True)
    
    for detection in detections:
        # identificación de objetos
        objects = detection.boxes.cls.numpy().tolist()
        # diccionario de nombres
        names = detection.names
        pieces = []
        if objects:
            # print('Nombres: ', names)
            # print('objetos: ', objects)
            # encontramos la posición del cuadrado
            # pos = ol.index(2)
            # print(pos)
            # coordenadas
            for index, object in enumerate(objects):
                # coodenadas de cada objeto
                coordinates = detection.boxes.xyxy[index].numpy() # .tolist()
                # nombre del objeto detectado
                object_name = names[object]
                # color asociado
                object_color = objects_colors[object_name]
                # creamos instancia de la pieza
                piece = Piece(name=object_name, coordinates=coordinates, color=object_color)
                pieces.append(piece)
                # Escribir el nombre encima del rectángulo
                cv2.putText(frame, object_name, (int(coordinates[0]), int(coordinates[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, object_color, 2)
                # Dibujar el rectángulo en la imagen
                cv2.rectangle(img=frame, pt1=(int(coordinates[0]), int(coordinates[1])),
                            pt2=(int(coordinates[2]), int(coordinates[3])), color=object_color, thickness=2)
    return frame, pieces

# capturar imagen de stream
def stream_image():
    # Pipeline is now finished, and we need to find an available device to run our pipeline
    # we are using context manager here that will dispose the device after we stop using it
    with depthai.Device(pipeline) as device: # crea la conexion y la cierra cuando sales del with. hace el close solo. No te tienes que encargar tu de ello
        # From this point, the Device will be in "running" mode and will start sending data via XLink
        # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
        q_rgb = device.getOutputQueue("rgb",  maxSize=4, blocking=False) # COLA DE MAXIMO 4
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
                break
    return frame
        
# pintar frame
def paint_frame(frame):
    # pintar frame
    cv2.imshow("preview", frame)
    # esperar a que se cierre con la tecla q
    while True:
        if cv2.waitKey(1) == ord('q'):
            break
    # Cerrar todas las ventanas
    cv2.destroyAllWindows()

# ordenar piezas
def order_pieces(pieces: list[Piece]) -> list[Piece]:
    o_pieces = sorted(pieces, key=lambda x: x.coordinates[0])
    return o_pieces

# streaming detections
def streaming_with__nn():
    # cargar red neuronal
    nn = init_YOLO_nn_model()
    # Abrir camara
    with depthai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb",  maxSize=1, blocking=False)
        frame = None
        # bucle de streaming
        while True:
            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                # analizamos frame con la red neuronal
                frame, pieces = process_detections(nn_model=nn, frame=frame)
                # ordenamos piezas
                if pieces and len(pieces) >= 2:
                    o_pieces = order_pieces(pieces)
                    # for piece in o_pieces:
                    #     print(piece.__dict__)
                    # enviamos piezas si detectamos 2 o mas
                    return o_pieces
                # pintar frame
                cv2.imshow("preview", frame)
            # esperar a que se cierre con la tecla q
            if cv2.waitKey(1) == ord('q'):
                # Cerrar todas las ventanas
                cv2.destroyAllWindows()
                break         
    return



# main
def main():
    # streaming hasta que detectemos 2 o mas piezas o cerremos el streaming con q
    pieces = streaming_with__nn()
    if not pieces:
        return 
    # vemos piezas detectadas y en que posiciones se encuentran
    order = 0
    for piece in pieces:
        print(order, ': ',piece.__dict__)
        order += 1

    return



# if __name__ == '__main__':
#     main()
