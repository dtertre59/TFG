import cv2
from depthai_sdk import OakCamera
from depthai_sdk.classes import DetectionPacket

with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color)

    # Callback
    def cb(packet: DetectionPacket):
        print(packet.img_detections)
        cv2.imshow(packet.name, packet.frame)

    # 1. Callback after visualization:
    oak.visualize(nn.out.main, fps=True, callback=cb)

    # 2. Callback:
    oak.callback(nn.out.main, callback=cb, enable_visualizer=True)

    oak.start(blocking=True)