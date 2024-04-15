
import cv2

import pupil_apriltags

# extraccion de la imagen
img = cv2.imread('apriltag/assets/apriltag_1.png', cv2.IMREAD_COLOR)
img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detectar apriltag de la imagen
detector = pupil_apriltags.Detector()
result = detector.detect(img_grayscale)

print(result)