
import cv2

import pupil_apriltags


# camera resolution
width = 1920
height = 1080

# parametros de la camara
# focal
fx = 0.00337 # 3.37mm
fy = fx
# center (resolution/2)
cx = width/2
cy = height/2

camera_params = (fx, fy, cx, cy)


# extraccion de la imagen
img = cv2.imread('apriltag/assets/apriltag_1.png', cv2.IMREAD_COLOR)
# Redimensiona la imagen utilizando la interpolación de área de OpenCV
img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detectar apriltag de la imagen
detector = pupil_apriltags.Detector()
detections = detector.detect(img_grayscale, estimate_tag_pose=True, camera_params=camera_params, tag_size=0.085)

detection = detections[0]
print(detection)

# Verificar si se detectó algún AprilTag y ver resultados
if detection:

    # Obtener las coordenadas de los vértices y el centro del primer AprilTag detectado
    corners = detection.corners.astype(int)
    center = detection.center.astype(int)


    color = (0, 255, 0)

    # Dibujar el recuadro del AprilTag y el centro en la imagen
    cv2.line(img, tuple(corners[0]), tuple(corners[1]), color, 2, cv2.LINE_AA, 0)
    cv2.line(img, tuple(corners[1]), tuple(corners[2]), color, 2, cv2.LINE_AA, 0)
    cv2.line(img, tuple(corners[2]), tuple(corners[3]), color, 2, cv2.LINE_AA, 0)
    cv2.line(img, tuple(corners[3]), tuple(corners[0]), color, 2, cv2.LINE_AA, 0)

    cv2.circle(img, tuple(center), 5, (0, 0, 255), -1)

    # Mostrar la imagen con el rectángulo y el centro marcados
    cv2.imshow('AprilTag', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se detectaron AprilTags en la imagen.")
