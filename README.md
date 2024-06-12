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

•	Implementar un sistema de comunicación y control de movimiento del brazo robótico de Universal Robots desde Python y Linux.  
•	Implementar un sistema de visión artificial que permita detectar, clasificar y situar en el espacio tridimensional una serie de objetos conocidos con distinta geometría.  
•	Implementar un método que coja los objetos que se encuentran en una determinada zona del espacio de trabajo y los coloque en otro punto del espacio de trabajo preestablecido.  


## Divisiones de Contenido

1. Control del brazo robótico Ur3e.

2. Control de la camara luxomis OAK-D litte.

3. Implementación de un modelo de Red neuronal generativa para la detección de objetos con la camara.

4. Unión

## Herramientas de trabajo

1. Editor de código: Visual Studio Code + python extension
2. Control de versiones: github
3. Sistema Operativo: Ubuntu version

## Configuración inicial

1. python 3.11
2. pip 24.0
3. librerias en requirements txt
4. entorno virtual venv

## Comandos iniciales python

1. Generación del entorno virtual: python -m venv venv
2. Activacion del entorno virtual: 
    1) Windows: venv\Scripts\activate
    2) Linux: source venv/bin/activate
3) Archivo con info de las librerias del proyecto: pip freeze > requirements.txt

## Repositorios de interés

1. Libreria conexion con brazo robótico: https://github.com/UniversalRobots/RTDE_Python_Client_Library.git
2. Camara:
    https://github.com/luxonis/depthai.git
    https://github.com/luxonis/depthai-python.git
    https://github.com/luxonis/blobconverter.git

3. deep learning, OpenVINO
    https://github.com/openvinotoolkit/open_model_zoo.git
    
4. procesamientode imagenes, show, modelos ai (sample/dnn)
	https://github.com/opencv/opencv.git


## OpenVINO toolkit

OpenVINO™ is an open-source toolkit for optimizing and deploying AI inference.

Open Model Zoo repository: This repository includes optimized deep learning models and a set of demos to expedite development of high-performance deep learning inference applications. Use these free pre-trained models instead of training your own models to speed-up the development and production deployment process.

## Luxonis

High‑resolution cameras with depth vision and on‑chip machine learning.

https://docs.luxonis.com/projects/sdk/en/latest/features/ai_models/

## Redes Neuronales
YOLO_v8 de ukltralitics la que he entrenado con mis propo¡ias fotos
he utilizado roboflow para el etiquetado y exportacion de imagenes
Google colab para ejecutar el script de entrenado de la red


## apriltag 

la libreria apriltag no puede instalarse en windows


# ---------- # 10
# -------------------- # 20
# ------------------------------------------------------------------------------------------------------------------------ # 120

