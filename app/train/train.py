from ultralytics import YOLO
# from roboflow import Roboflow
from pathlib import Path


# NO VA
# def download_images():
    # from roboflow import Roboflow
    # rf = Roboflow(api_key="H8VTKGgmS64rRZ5BzCRa")
    # project = rf.workspace("dtertre59workspace").project("tfg_pose_640")
    # version = project.version(5)
    # dataset = version.download("yolov8", location=str(Path(__file__).parent.resolve()))

# SI; las imagenes me las eh descarado en zip desde roboflow
def train():
    # model = YOLO(str(Path(__file__).parent.resolve() / 'yolov8m-pose.pt'))
    model = YOLO('yolov8s-pose.pt')
    results = model.train(data=str(Path(__file__).parent.resolve() / 'TFG_pose_640_v2_v2' / 'data.yaml'), epochs=450, imgsz=640, project=str(Path(__file__).parent.resolve() / 'runs2' ))


# train()