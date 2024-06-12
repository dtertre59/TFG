# Desarrollo de un sistema de clasificación y colocación de objetos con un brazo robótico colaborativo


## Trabajo de Fin de grado en Ingeniería Electrónica Industrial y Automática


## Autor

David Tertre Boyé


## Resumen

El presente proyecto de final de carrera realiza un sistema integral de clasificación y colocación de objetos con el brazo robótico UR3e. Se han desarrollado dos programas que actúan simultáneamente para cumplir los objetivos establecidos. El primero se ejecuta en el robot, programa de Universal Robots, y el segundo, programa en Python, en el ordenador con sistema operativo Linux.  

El sistema resuelve el problema varias formas diferentes de formas diferentes, haciendo uso de métodos avanzados de procesamiento de imágenes, visión artificial y técnicas de aprendizaje automático para identificar, clasificar y posicionar los objetos presentes en el entorno. Estos métodos incluyen la detección mediante apriltags, el entrenamiento y uso de la red neuronal convolucional YOLOv8, y técnicas clásicas de visión artificial como el filtrado de imágenes y la detección de contornos y esquinas.  
Una vez detectados y clasificados, el brazo robótico ejecuta operaciones de pick and place, recogiendo los objetos y ubicándolos en sus posiciones designadas. 

El sistema ha sido diseñado para funcionar en un entorno controlado, optimizando la iluminación y el contraste entre los objetos y el fondo para mejorar la precisión de los métodos de detección. Además, se ha implementado un manejo robusto de excepciones para garantizar la estabilidad del sistema en caso de desconexiones o mal funcionamiento de los componentes.  

La integración del sistema incluye la coordinación entre el robot, la cámara y los algoritmos de detección, permitiendo una operación fluida y eficiente.


## Objetivos

- Implementar un sistema de comunicación y control de movimiento del brazo robótico de Universal Robots desde Python y Linux.
- Implementar un sistema de visión artificial que permita detectar, clasificar y situar en el espacio tridimensional una serie de objetos conocidos con distinta geometría.
- Implementar un método que coja los objetos que se encuentran en una determinada zona del espacio de trabajo y los coloque en otro punto del espacio de trabajo preestablecido.  


## Componentes

- UR3e, cobot fabricado por Universal Robots.
- Cámara Luxonis OAK-D-Lite.
- Ordenador con sistema operativo Linux o Windows.


## Versiones

- Python==3.11
- Pip==24.0
- Biliotecas del proyecto en el archivo 'requirements txt'

## Bibliotecas más importantes

- Comunicacion con el robot: RTDE
- COmunicacion con la cámara: Depthai
- Red neuronal: ultralytics YOLO
- Manejo de imagenes: opencv
- Manejo de nubes de puntos: open3d
- Manejo de numeros: numpy

## Estructura del repositorio

En la carpeta app se encuentra el main y todas las dependencias y ficheros necesarios para que funcione el sistema.

En la carpeta external_assets se enncuentra el modelo 3d del soporte de la cámara fabricado, el programa URP e instalación del robot, y los resultados obtenidos al probar los sistemas desarollados.

En la carpeta practice se encuentran todos los programas de pruebas, de testeo de librerias, programas modelos, etc.

memoria del TFG en formato word y pdf, 








## Repositorios de interés

1. Libreria de comunicacion con brazo robótico: https://github.com/UniversalRobots/RTDE_Python_Client_Library.git
2. Camara:
    https://github.com/luxonis/depthai.git
    https://github.com/luxonis/depthai-python.git
    https://github.com/luxonis/blobconverter.git

3. deep learning, OpenVINO
    https://github.com/openvinotoolkit/open_model_zoo.git
    
4. procesamientode imagenes, show, modelos ai (sample/dnn)
	https://github.com/opencv/opencv.git

