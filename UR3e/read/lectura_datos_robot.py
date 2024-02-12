


import argparse
import logging
import sys

from datetime import datetime as datetime
import time

import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import rtde.csv_writer as csv_writer
import rtde.csv_binary_writer as csv_binary_writer


# def main():

#     print('INICIO')

# if __name__=="__main__":
#     main()

def writerow_v0(data_object):
        data = []
        for i in range(len(data_object.__names)):
            size = data_object.__types[i]
            value = data_object.__dict__[data_object.__names[i]]
            if size > 1:
                data.extend(value)
            else:
                data.append(value)
        return data

# parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", default="localhost", help="name of host to connect to (localhost)"
)
parser.add_argument("--port", type=int, default=30004, help="port number (30004)")
parser.add_argument(
    "--samples", type=int, default=0, help="number of samples to record"
)
parser.add_argument(
    "--frequency", type=int, default=125, help="the sampling frequency in Herz"
)
parser.add_argument(
    "--config",
    default="record_configuration.xml",
    help="data configuration file to use (record_configuration.xml)",
)
parser.add_argument(
    "--output",
    default="robot_data.csv",
    help="data output file to write to (robot_data.csv)",
)
parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
parser.add_argument(
    "--buffered",
    help="Use buffered receive which doesn't skip data",
    action="store_true",
)
parser.add_argument(
    "--binary", help="save the data in binary format", action="store_true"
)
args = parser.parse_args()


# CAMBIAMOS PARAMETROS 
args.host = "192.168.10.222" # ip del robot
# el puerto ya ha sido cambiado


if args.verbose:
    logging.basicConfig(level=logging.INFO)

conf = rtde_config.ConfigFile(args.config)
output_names, output_types = conf.get_recipe("out")

# 
print('Output names 0: ',output_names[0])
print('Output types 0: ',output_types[0])

# host y puerto 
print('Robot Host: ', args.host)
print('Port: ', args.port)
print('output csv path: ',args.output)

# instancia de la conexion 
con = rtde.RTDE(args.host, args.port)
con.connect()

# get controller version
print('controler version: ', con.get_controller_version())


# setup recipes
if not con.send_output_setup(output_names, output_types, frequency=args.frequency):
    logging.error("Unable to configure output")
    sys.exit()

# start data synchronization
if not con.send_start():
    logging.error("Unable to start synchronization")
    sys.exit()

# COMIENZO DE LA LECTURA -------------------------------------------------------------------    
    
writeModes = "wb" if args.binary else "w"
with open(args.output, writeModes) as csvfile:
    writer = None

    # encabezado del CSV
    if args.binary:
        writer = csv_binary_writer.CSVBinaryWriter(csvfile, output_names, output_types)
    else:
        writer = csv_writer.CSVWriter(csvfile, output_names, output_types)

    writer.writeheader()

    print('args bufferd',args.buffered) # default -> False
    print('args binary',args.binary)    # default -> False
    print('\n')
    # lectura de los datos del xml
    i = 0
    print ('ACTUAL TCP POSE respecto de la base')
    while True:
        r_data = {}
        try:
            if args.buffered:
                state = con.receive_buffered(args.binary)
            else: # esta entrando aqui
                state = con.receive(args.binary)
            if state is not None:
                writer.writerow(state)
                # toda la info de los registros del xml en modo dict
                r_data = state.__dict__
        except:
            print('ERROR RARO')

        # manipulacion de los dqtos extraidos del robot
        # 1. transformacion del timestap a legible
        # actual_TCP_pose = [X, Y, Z, RX, RY, RZ]
        r_data['timestamp'] = datetime.fromtimestamp(int(r_data['timestamp']))
        print(r_data['actual_TCP_pose']) # respecto de la base

        # otros regustros
        print(r_data['input_double_register_0'])
        
        time.sleep(1)
        i += 1
        if i>10: break

    # BUCLE DE LECTURA DE DATOS ---------------------------------------------------------------------
    # # Filas de datos del csv
    # i = 1
    # keep_running = True
    # while keep_running:
    #     # print('\rpaso ', i, end='', flush=True)
    #     # escritura de consola
    #     if i % args.frequency == 0:
    #         if args.samples > 0:
    #             sys.stdout.write("\r")
    #             sys.stdout.write("{:.2%} done.".format(float(i) / float(args.samples)))
    #             sys.stdout.flush()
    #         else:
    #             sys.stdout.write("\r") # sobrescribe la linea actual
    #             sys.stdout.write("{:3d} samples.".format(i)) # escribe esto
    #             sys.stdout.flush()
            
    #     if args.samples > 0 and i >= args.samples:
    #         keep_running = False
    #     try:
    #         if args.buffered:
    #             state = con.receive_buffered(args.binary)
    #         else:
    #             state = con.receive(args.binary)
    #         if state is not None:
    #             writer.writerow(state)
    #             # algoooo
    #             # print(state)
    #             i += 1

    #     except KeyboardInterrupt:
    #         keep_running = False
    #     except rtde.RTDEException:
    #         con.disconnect()
    #         sys.exit()
    # ----------------------------------------------------------------------------------------

sys.stdout.write("\nComplete!            \n")
# Finalizar conexión con el robot
con.send_pause()
con.disconnect()