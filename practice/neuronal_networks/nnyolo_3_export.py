from ultralytics import YOLO

# Load a model
model = YOLO('./yolov8n.pt')  # load a pretrained model
path = model.export(format="onnx")  # export the model to ONNX format
print(path)

