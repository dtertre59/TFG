# TFG

## Authors

David Tertre Boyé

## Intro

Trabajo de Fin de grado en Ingeniería Electrónica Industrial y Automática

## Objetivos

Control de un brazo robotico utilizando registros más la implementación de una cámara de apoyo al robot en la tarea destinada.

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

