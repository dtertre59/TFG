
import numpy as np

import pupil_apriltags as apriltag


class MyApriltagConfig():
    def __init__(self, family: str, size: float) -> None:
        self.family = family  # Familia del AprilTag
        self.size = size  # Tamaño del AprilTag

        
class MyApriltag(MyApriltagConfig):
    def __init__(self, family: str, size: float) -> None:
        super().__init__(family, size)

        self.detector = apriltag.Detector(families=family)

        self.id = None
        self.corners = None
        self.center = None
        self.T = None

    
