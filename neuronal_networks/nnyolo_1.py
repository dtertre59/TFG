from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

# # Run batched inference on a list of images
# results = model(['bus.jpg'])  # return a list of Results objects

# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     result.show()  # display to screen
#     # result.save(filename='result.jpg')  # save to disk


# Read an image using OpenCV
img = cv2.imread('bus.jpg')
img_resize= cv2.resize(img, (400,int(img.shape[0]*400/img.shape[1])))

# Run inference on the img
# results = model(img)  # list of Results objects
# print(results)

# Run inference on 'bus.jpg' with arguments
# model.predict('bus.jpg', save=True, imgsz=320, conf=0.5)

results = model(img_resize)

def take_box_xyyx(results):
    results.boxes


# datos de la primera persona


# View results
for r in results:
    # print(r.boxes)  # print the Boxes object containing the detection bounding boxes
    # print('masks: ', r.masks)  # print the Masks object containing the detected instance masks
    # print('keypoints: ',r.keypoints)  # print the Keypoints object containing the detected keypoints
    # print('probs: ',r.probs)  # print the Probs object containing the detected class probabilities
    # print('obb: ',r.obb)  # print the OBB object containing the oriented detection bounding boxes
    # print(r.path)
    # print(r.names)
    # print(r.probs)  # print the Probs object containing the detected class probabilities





    # identificacion de objetos
    ol = r.boxes.cls.numpy().tolist() # primero convertimos el tensor a array y luego el array a una lista
    print(ol)
    # encontramos la posicion de la primera persona -> identificador 0 -> lo vemos en el diccionario de la red
    pos = ol.index(0)
    print(pos)
    # coordenadas
    coordenadas = r.boxes.xyxy[pos].numpy().tolist()
    print(coordenadas)

    # con opencv dibujamos recuadro
    cv2.imshow('Iimagen', img_resize)
    # Dibujar el rectángulo en la imagen
    cv2.rectangle(img=img_resize, pt1=(int(coordenadas[0]),int(coordenadas[1])),pt2=(int(coordenadas[2]),int(coordenadas[3])), color=(0,255,0), thickness=2)

    # Mostrar la imagen con el rectángulo
    cv2.imshow('Rectangulo', img_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# # Visualize the results
# for i, r in enumerate(results):
#     # Plot results image
#     im_bgr = r.plot()  # BGR-order numpy array

#     # Si deseas convertirlo a una imagen OpenCV, puedes hacer lo siguiente
#     im_opencv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

#     # Mostrar la imagen en una ventana de OpenCV
#     cv2.imshow('Imagen RGB', im_bgr)

#     # Esperar hasta que se presione una tecla y luego cerrar la ventana
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
