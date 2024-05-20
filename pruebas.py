from ultralytics import YOLO

#Load a model
# model = YOLO('yolov8n-pose.pt')  # load an official model
model = YOLO('app/assets/nn_models/yolov8s_pose_v3.pt')

# model = YOLO(fr'square_model_1.pt')

# Predict with the model
results = model('/home/robotica/Documents/dtertre59/TFG/app/assets/pictures/train/square/square_12.png')  # predict on an image

print(results)

# View results
# for r in results:
#     print(r.keypoints)  # print the Keypoints object containing the detected keypoints
#     r.show()

    

# def ejemplo(*args, **kwargs):
#     print("Argumentos posicionales:")
#     for arg in args:
#         print(arg)
        
#     print("\nArgumentos de palabra clave:")
#     for key, value in kwargs.items():
#         print(f"{key}: {value}")

# # Llamada a la función ejemplo con diferentes tipos de argumentos
# ejemplo(1, 2, 3, nombre="Alice", edad=30)