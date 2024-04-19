import depthai as dai

import numpy as np


# parametros de la camara
# Distancia focal efectiva: 3,37 mm · Tamaño de píxel: 1,12 µm x 1,12 µm
# distancia focal = 0.50

# FOCAL LENGTH CALCULATION
# with dai.Device() as device:
#   calibData = device.readCalibration()
#   intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C)
#   focal_length = intrinsics[0][0]
#   print('Right mono camera focal length in pixels:', focal_length)

# For OAK-D @ 400P mono cameras and disparity of eg. 50 pixels
# depth_cm = focal_length_in_pixels * baseline / disparity_in_pixels
# depth_cm = 451.145 * 7.5 / 50 

# print(451.145*1.12*0.001)

# print(stereo.initialConfig.setConfidenceThreshold())

# Min stereo depth distance
# min_distance = focal_length_in_pixels * baseline / disparity_in_pixels = 882.5 * 7.5cm / 95 = 69.67cm
  
# Max stereo depth distance
# Dm = (baseline/2) * tan((90 - HFOV / HPixels)*pi/180)


# Supongamos que tienes las coordenadas de los puntos en la matriz de disparidad (u1, v1) y (u2, v2)
u1, v1 = [0,0]
u2, v2 = [20,20]

# Supongamos que conoces las coordenadas tridimensionales de los puntos 1 y 2
x1, y1, z1 = [0,0,0]
x2, y2, z2 = [200,200,0]

# Supongamos que conoces las distancias de los píxeles de los puntos 1 y 2
D1 = 10
D2 = 12

# Calcular la disparidad entre los puntos en la imagen de disparidad
disparidad_punto1_punto2 = abs(u2 - u1)

# Calcular la distancia entre los puntos 1 y 3
distancia_P1_P3 = D1 * (z2 - z1) / disparidad_punto1_punto2

# Calcular las coordenadas de P3
x3 = x1 + (x2 - x1) * distancia_P1_P3 / (z2 - z1)
y3 = y1 + (y2 - y1) * distancia_P1_P3 / (z2 - z1)
z3 = z1 + (z2 - z1) * distancia_P1_P3 / (z2 - z1)

print("Coordenadas de P3:", x3, y3, z3)


  
