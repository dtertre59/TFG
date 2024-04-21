import robotpy_apriltag as at
import cv2
from pathlib import Path

width = 1280
height = 720
resolution = (width, height)

# extraccion de la imagen
img = cv2.imread(str(Path(__file__).resolve().parent / 'assets/apriltag_2.png'), cv2.IMREAD_COLOR)
# Redimensiona la imagen utilizando la interpolación de área de OpenCV
img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



detector = at.AprilTagDetector()
# Look for tag36h11, correct 3 error bits
detector.addFamily("tag36h11", 3)

# camera params
# Effective Focal Length 3.37mm
# Pixel size: 1.12µm x 1.12µm
# f = 3370/1.12 = 3008.92857
poseEstConfig = at.AprilTagPoseEstimator.Config(
        tagSize=0.085,
        fx=3008.92857,
        fy=3008.92857, 
        cx=resolution[0]/2,
        cy=resolution[1]/2,
    )
estimator = at.AprilTagPoseEstimator(poseEstConfig)

detections = detector.detect(img_grayscale)

detection = detections[0]

center = detection.getCenter()
corners = []
for i in range(4):
    corner = detection.getCorner(ndx=i) 
    corners.append((int(corner.x),int(corner.y)))

# Determine Tag Pose
pose = estimator.estimate(detection)


print('apriltag center: ',center)
print('apriltag corners: ',corners)
print('apriltag pose: ',pose)

color = (0, 255 ,0)

cv2.circle(img, (int(center.x), int(center.y)), 5, (0, 0, 255), -1)

# Dibujar el recuadro del AprilTag y el centro en la imagen
cv2.line(img, tuple(corners[0]), tuple(corners[1]), color, 2, cv2.LINE_AA, 0)
cv2.line(img, tuple(corners[1]), tuple(corners[2]), color, 2, cv2.LINE_AA, 0)
cv2.line(img, tuple(corners[2]), tuple(corners[3]), color, 2, cv2.LINE_AA, 0)
cv2.line(img, tuple(corners[3]), tuple(corners[0]), color, 2, cv2.LINE_AA, 0)

cv2.imshow('AprilTag', img)
cv2.waitKey(0)
cv2.destroyAllWindows()