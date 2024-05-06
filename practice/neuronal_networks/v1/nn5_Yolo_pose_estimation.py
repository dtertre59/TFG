from ultralytics import YOLO
# import cv2

# # Load a model
# model = YOLO('yolov8n-pose.pt')  # load an official model
# # model = YOLO(fr'practice\neuronal_networks\yolov8n-pose.pt')  # pretrained YOLOv8n model
# # model =  YOLO(fr'practice\neuronal_networks\v1\tigre_model_1.pt') 

# # Predict with the model
# results = model.predict('https://ultralytics.com/images/bus.jpg')  # predict on an image

# # View results
# for r in results:
#     # print(r.keypoints)  # print the Keypoints object containing the detected keypoints
#     r.show()