import numpy as np


# apriltag to baserobot transformation matrix

punto_april = np.array([0,0,0])

punto_pieza_respecto_april = np.array([0.3, 0.3, 0.3])

t_april_to_robot = np.array([[1, 0, 0, -0.1075],
                    [0, 1, 0, -0.3839],
                    [0, 0, 1, 0.0108],
                    [0, 0, 0, 1]])


def reference_system_transformation(point: np.ndarray, transformation_matrix: np.ndarray):
    # Agrega una coordenada homogénea al punto
    h_point= np.append(point, 1)
    # Multiplica la matriz de transformación por el punto homogéneo
    h_t_point = np.dot(transformation_matrix, h_point)

    # Normaliza dividiendo por la coordenada homogénea final
    t_point = np.array([h_t_point[0] / h_t_point[3], 
               h_t_point[1] / h_t_point[3],
               h_t_point[2] / h_t_point[3]])
    return t_point

t_point = reference_system_transformation(punto_pieza_respecto_april, t_april_to_robot)

print(t_point)