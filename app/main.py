"""
        main.py

    Main script

"""

# -------------------- PACKAGES ------------------------------------------------------------------------------------------ #

import functions.main_functions as mf


# -------------------- FUNCTIONS ----------------------------------------------------------------------------------------- #

# Con apriltags
def main():
    # 1. conexion con el robot (verificar con algun registro)

    # 1.1 mover robot a posicion inicial o de reposo

    # 2. (thread) visualizar con la camara el area donde se encuentran las piezas (el robot en reposo ya apunta a este area)
    mf.init_camera_and_visualize()
    # 2.1 adquirimos frame (es necesario que se vea el apriltag de ref)

    # 2.2 deteccion de apriltags con las funciones de la libreria apriltags. Sabemos la pieza que es porque tiene un tag_id conocido.

    # 2.3 ubicar centro del april de las piezas como punto 3d respecto a la base del robot (matrices de transferencia)

    # 3. movimiento del robot para conger la pieza y dejarla en su respectivo hoyo (posicion conocida)

    # 4. repetimos en bucle hasta que no haya mas piezas

    # 5. (opcional) revisar que los hoyos no esten ocupados para poder mover la pieza a su hoyo

    # 6 .(opcional) poner un apriltag en el madero de los hoyos y asi no es necesario saber la posicion exacta de cada hoyo, solo la relativa respecto al april2

    return


    

# Con red neuronal
def main2():
    # 1. conexion con el robot (verificar con algun registro)

    # 1.1 mover robot a posicion inicial o de reposo

    # 2. (thread) visualizar con la camara el area donde se encuentran las piezas (el robot en reposo ya apunta a este area)

    # 2.1 adquirimos frame (es necesario que se vea el apriltag de ref)

    # 2.2 deteccion de la pieza con red neuronal. Pose estimation con la red. por lo que concocemos el centro de la cara superior de la pieza

    # 2.3 pasamos pixel (centro) a punto en 3d de la nube de puntos (respecto de la camara)

    # 2.4 pasamos el punto 3d respecto de la camara al sistema de ref del robot utilizando el apriltag de referencia

    # 3. movimiento del robot para conger la pieza y dejarla en su respectivo hoyo (posicion conocida)
    

    return



if __name__ == '__main__':
    main()
    # main2()