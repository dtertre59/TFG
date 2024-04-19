# TFG

## Authors

David Tertre


## Introducción 

Se va a desarrollar una aplicación en python para la manipulación (control/manejo) del robot ur3e.


## Objetivos


## Pasos a seguir

Utilizaremos la libreria proporcionada por la propia marca del robot (Universal Robots): https://github.com/UniversalRobots/RTDE_Python_Client_Library

1. Conexión con el robot mediante IP. EN la red local (LAN)
2. Extracción de registros: Seleccionados en el archivo record_configuration.xml
3. Escribir el los registros del robot para conseguir el movimiento del robot a través de la app
4. Aplicación final utilizando lo aprendido anteriormente


## Comandos utilizados

1. Para ver a que red estamos conectados
2. scaneo de dispositivosde la red local
3. escaneo de puertos


## Desde la tablet del robot

1. Asignar IP estatica al robot (192.168.1.222) : Ajustes->sistema->red
2. Asignar IP + port de control remoto (192.168.10.103) : Instalacion->URCaps->ExternalControl



## Leer registros del robot

Hecho,
Archivos necesarios:
1. Depebdencas del proyecto
2. programa de ejemplo: lectura_datos_robot.py
3. archivos necesarios: 
	- Registros que se van a leer: record_configuration.xml
	- Output: robot_data.csv
	
## Mover Robot -> Cabiar registros
es necesario que en la tableta del robot se este ejecutando un programa de lectura de registros para que cuando los cambies desde el programa en python el robot los aplique



