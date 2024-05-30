import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('todas.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# create a mask
#Bounding Box: [p1: Vetor2D: [x=1346.0, y=511.0], p2: Vetor2D: [x=1544.0, y=767.0]]
#Bounding Box: [p1: Vetor2D: [x=1072.0, y=434.0], p2: Vetor2D: [x=1256.0, y=714.0]]
#Bounding Box: [p1: Vetor2D: [x=768.0, y=516.0], p2: Vetor2D: [x=910.0, y=765.0]]

image_cropped = img[511:767, 1346:1544]
image_cropped = img[516:765, 768:910]
image_cropped = img[434:714, 1072:1256]

edges_cropped = cv.Canny(image_cropped,80,200)
hist_cropped = cv.calcHist([image_cropped],[0],None,[200],[0,256])
 
plt.subplot(121),plt.imshow(image_cropped,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges_cropped,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

contours, hierarchy = cv.findContours(edges_cropped, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
img = image_cropped

#cnt = contours[3]
#cv.drawContours(img, [cnt], 0, (0,255,0), 3)
max=0
i=0
index = 0
for cnt in contours:
    perimeter = cv.arcLength(cnt,False)
    if perimeter > max:
        index = i    
    print(perimeter)
    i=i+1

cnt = contours[index]
cv.drawContours(img, [cnt], 0, (0,255,0), 3)
cv.imshow("contorno", img)
cv.waitKey(0)
    
