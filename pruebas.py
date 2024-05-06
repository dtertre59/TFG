from ultralytics import YOLO

#Load a model
# model = YOLO('yolov8n-pose.pt')  # load an official model
model = YOLO(fr'tigre_model_1.pt')
# model = YOLO(fr'square_model_1.pt')

# Predict with the model
results = model('tigre2.png')  # predict on an image

# View results
for r in results:
    print(r.keypoints)  # print the Keypoints object containing the detected keypoints
    r.show()