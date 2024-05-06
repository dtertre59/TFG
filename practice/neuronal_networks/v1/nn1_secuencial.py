# # import tensorflow
# # import keras
# import matplotlib
# import matplotlib.pyplot as plt

# import numpy as np



# features = [(0, 54, 2, 4, 0, 1, 0, 0),
#             (0, 152, 2, 4, 1, 1, 3, 1),
#             (0, 64, 3, 4, 0, 1, 0, 0),
#             (0, 154, 5, 4, 1, 1, 1, 1),
#             (0, 100, 1, 5, 1, 1, 1, 0),
#             (0, 140, 5, 2, 1, 1, 2, 0),
#             (0, 120, 3, 2, 1, 1, 1, 1),
#             (0, 70, 2, 3, 1, 1, 1, 0),
#             (0, 60, 2, 2, 0, 1, 1, 1),
#             (0, 129, 3, 18, 1, 1, 2, 1),
#             (0, 93, 1, 3, 1, 1, 2, 0),
#             (0, 52, 2, 2, 0, 1, 1, 1),
#             (0, 110, 3, 5, 1, 1, 1, 1),
#             (0, 63, 3, 2, 1, 1, 1, 0),
#             (0, 160, 1, 4, 1, 1, 2, 0)
#             ]

# targets = [750, 2000, 650, 1500, 900, 1000, 1300, 750, 900, 1800, 975, 880, 1400, 750, 1050]


# capa_entrada = keras.layers.Dense(units=8, input_shape=[8])
# capa_oculta = keras.layers.Dense(units=8)
# capa_salida = keras.layers.Dense(units=1)

# modelo = keras.Sequential([capa_entrada, capa_oculta, capa_salida])

# modelo.compile(
#     optimizer = keras.optimizers.Adam(0.1),
#     loss = 'mean_squared_error'
# )

# print('Inicio del entrenamiento')
# # historial = modelo.fit(features,targets,epochs=1000, verbose=False)
# print('fin del entrenamiento')


# # ----- PLOT 

# # plt.xlabel('#Época')
# # plt.ylabel('Mágnitud de pérdida')
# # plt.plot(historial.history['loss'])
# # plt.show()


# # ----- GUARDAR MODELO

# # # Guardar la arquitectura del modelo en formato JSON
# # model_json = modelo.to_json()
# # with open('./modelos/m_pisos_config.json', 'w') as json_file:
# #     json_file.write(model_json)

# # # Guardar los pesos del modelo en formato HDF5
# # modelo.save_weights(filepath='./modelos/m_pisos_weights.h5')

# # # Guardar el modelo en formato keras
# # modelo.save(filepath='./modelos/m_pisos_keras.keras')

# # # export
# # modelo.export(filepath='./modelos/m_pisos_export')


# # ----- CARGAR MODELO

# model = keras.models.load_model('./modelos/m_pisos_keras.keras')
# # model.summary()

# # model.load_weights("./modelos/m_pisos_weights.h5")

# # Leer el contenido del archivo JSON como una cadena
# # with open('./modelos/m_pisos_config.json', "r") as archivo:
# #     model_config = archivo.read()
# # model = keras.models.model_from_json(model_config)




# # resumen
# print(model.to_json())
# model.summary()


# # Datos de entrada para la predicción
# # datos_entrada = np.array([[0, 54, 2, 4, 0, 1, 0, 0]])
# datos_entrada = np.random.rand(1, 8)  # Genera un array 1x8 con valores aleatorios entre 0 y 1
# print(datos_entrada)
# # Realizar la predicción
# prediccion = modelo.predict(datos_entrada)
# # Imprimir la predicción
# print("Predicción:", prediccion)