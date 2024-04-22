
import numpy as np
from numpy.core.multiarray import array as array

class Vector2D():
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    
    def __str__(self) -> str:
        return f"Vetor2d: [x={self.x}, y={self.y}]"
        


class ApriltagConfig():
    def __init__(self, family: str, size: float) -> None:
        self.family = family  # Familia del AprilTag
        self.size = size  # Tamaño del AprilTag

        
class Apriltag(ApriltagConfig):
    def __init__(self, id: int, family: str, size: float, corners: np.array, center: np.array, T: np.ndarray) -> None:
        super().__init__(family, size)
        self.id = id
        self.corners = corners
        self.center = center
        self.T = T


class CameraConfig():
    def __init__(self, width: int, height: int, fx: float, fy: float) -> None:
        """ Camera config
            Resolution: pixels
            Camera center: pixels
            Focal lenth: pixels
            """
        self.resolution = Vector2D(width, height)
        self.f = Vector2D(fx, fy)
        self.c = Vector2D(width/2, height/2)


