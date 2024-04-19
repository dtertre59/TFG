import cv2
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

# cargar imagen con OpenCV
img_path = './images/face1.png'
img = cv2.imread(filename=img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Mostrar la imagen
# plt.imshow(img_rgb)
# plt.axis('off')
# plt.show()

# Inicializar el detector MTCNN
detector = MTCNN()

# Detectar caras en la imagen
faces = detector.detect_faces(img)

# Dibujar rectángulos alrededor de las caras detectadas
for face in faces:
    x, y, width, height = face['box']
    cv2.rectangle(img_rgb, (x, y), (x+width, y+height), (0, 255, 0), 2)

# Mostrar la imagen con caras detectadas
plt.imshow(img_rgb)
plt.axis('off')
plt.show()