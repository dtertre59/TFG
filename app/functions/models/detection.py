
import cv2
import numpy as np

from functions.models.myCamera import MyCameraConfig, MyCamera
from functions.models.myApriltag import MyApriltagConfig, MyApriltag

from functions.models.vectors import Vector2D



class myDetection():
    def __init__(self, name: str, ) -> None:
        self.name = None
        self.bounding_box = None
        self.corners = None
        self.pixelpoint = None
        self.point3d = None
        self.rotation = None


class DetectionsCoordinator():

    @staticmethod
    def apriltag_detections(frame, camera: MyCamera, apriltag: MyApriltag) -> list:
        # 1. Camera params
        camera_params = [camera.f.x, camera.f.y, camera.c.x, camera.c.y]
        # 2. frame to grayscale
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 3. detections
        detections = apriltag.detector.detect(frame_grayscale, True, camera_params=camera_params, tag_size=apriltag.size)

        if detections:
            return frame, True
        return frame, False
        

    
    @staticmethod
    def nn_object_detection(frame) -> myDetection:
        
        pass

    @staticmethod
    def nn_poseEstimation_detection(frame) -> myDetection:
        pass