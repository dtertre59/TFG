import numpy as np


class Vector2D():
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
    
    def __str__(self) -> str:
        return f"Vetor2D: [x={self.x}, y={self.y}]"
    
class Vector3D(Vector2D):
    def __init__(self, x: float, y: float, z: float) -> None:
        super().__init__(x,y)
        self.z = z
    
    def __str__(self) -> str:
        return "Vector3D"

class Vector6D(Vector3D):
    def __init__(self, x: float, y: float, z: float, rx: float, ry: float, rz: float) -> None:
        super().__init__(x, y, z)
        self.rx = rx
        self.ry = ry
        self.rz = rz

    def __str__(self) -> str:
        return f"Vetor6D"
    
    def get_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.rx, self.ry, self.rz])
