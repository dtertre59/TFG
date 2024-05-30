import numpy as np
import cv2
from matplotlib import pyplot as plt
 
img = cv2.imread('practice/classic_av/todas.png', cv2.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# create a mask
#Bounding Box: [p1: Vetor2D: [x=1346.0, y=511.0], p2: Vetor2D: [x=1544.0, y=767.0]]
#Bounding Box: [p1: Vetor2D: [x=1072.0, y=434.0], p2: Vetor2D: [x=1256.0, y=714.0]]
#Bounding Box: [p1: Vetor2D: [x=768.0, y=516.0], p2: Vetor2D: [x=910.0, y=765.0]]

image_cropped = img[516:765, 768:910]
image_cropped = img[434:714, 1072:1256]
image_cropped = img[511:767, 1346:1544]



# blanco y negro recortada
edges_cropped = cv2.Canny(image_cropped,80,200)

# no se utiliza el histograma
hist_cropped = cv2.calcHist([image_cropped],[0],None,[200],[0,256])


 
plt.subplot(121),plt.imshow(image_cropped,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges_cropped,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

contours, hierarchy = cv2.findContours(edges_cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img = image_cropped

#cnt = contours[3]
#cv2.drawContours(img, [cnt], 0, (0,255,0), 3)


max=0
i=0
index = 0
for cnt in contours:
    perimeter = cv2.arcLength(cnt,False)
    if perimeter > max:
        index = i    
    print(perimeter)
    i=i+1

print('len: ',len(contours))
cnt = contours[index]
cv2.drawContours(img, [cnt], 0, (0,255,255), 3)
cv2.imshow("contorno", img)
cv2.waitKey(0)
    
