import apriltag
import cv2

width = 1920
height = 1080

# extraccion de la imagen
img = cv2.imread('apriltag/assets/apriltag_1.png', cv2.IMREAD_COLOR)
# Redimensiona la imagen utilizando la interpolación de área de OpenCV
img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# parametros de la camara
# focal
fx = 0.00337 # 3.37mm
fy = fx
# center (resolution/2) -> 1920x1080
cx = width/2
cy = height/2

camera_params = [fx, fy, cx, cy]

# detector
detector = apriltag.Detector()

# opciones del detector
# print(detector.options.__dict__)

# detecciones apriltags de la imagen
detections = detector.detect(img_grayscale)

# elegimos la primera deteccion
detection = detections[0]

# conseguimos matriz de transformacion
transformation_matrix, initial_error, final_error = detector.detection_pose(detection, camera_params=camera_params, tag_size=0.085)

print('T: ', transformation_matrix)



# Verificar si se detectó algún AprilTag y ver resultados
if detections:
    # Obtener las coordenadas de los vértices y el centro del primer AprilTag detectado
    corners = detections[0].corners.astype(int)
    center = detections[0].center.astype(int)
    
    # print(detections)
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