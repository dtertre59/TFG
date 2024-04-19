import cv2
import numpy as np

# Img path
filename = './images/face1.png'

# model architecture
prototxt = './models/face_detection/deploy.prototxt'

# model weights
model = './models/face_detection/res10_300x300_ssd_iter_140000.caffemodel'

# Load model
nn = cv2.dnn.readNetFromCaffe(prototxt, model)

# Read image and preprocessing
img = cv2.imread(filename=filename)  
height, width, _ = img.shape
img_resize = cv2.resize(img, (300,300))

# create a blob
blob = cv2.dnn.blobFromImage(img_resize, 1.0, (300,300), (104,117,123)) # filtro de color, tamaño, etc
blob_to_show = cv2.merge([blob[0][0], blob[0][1], blob[0][2]])

# Detections and predictions
nn.setInput(blob=blob)
detections = nn.forward()

for detection in detections[0][0]:
    if detection[2] > 0.9: # confianza de que ha acertado > 90%
        # print('detection: ',detection)
        box = detection[3:7] * np.array([width, height, width, height])
        # print('box: ',box)
        x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # print(x_start)
        cv2.putText(img=img, text="Conf: {:.2f}".format(detection[2]*100), org=(x_start, y_start-5),fontScale=1 ,fontFace=1, color=(255,255,0))
        cv2.rectangle(img=img, pt1=(x_start, y_start), pt2=(x_end, y_end), color=(255, 0, 0), thickness=2)

cv2.imshow("Image", img)
# cv2.imshow("Blob", blob_to_show)
# cv2.imshow("Image re", img_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()