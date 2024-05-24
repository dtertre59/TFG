import numpy as np
from typing import overload, Union


class Vector2D():
    @overload
    def __init__(self, x: int|float, y: int|float) -> None: ...
        
    @overload
    def __init__(self, array: np.ndarray) ->None: ...

    def __init__(self, *args: Union[float|int, np.ndarray]) -> None:
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            array = args[0]
            if array.shape != (2,):
                raise ValueError("Array must have shape (2,)")
            self.x = float(array[0])
            self.y = float(array[1])
        elif len(args) == 2 and all(isinstance(arg, float|int) for arg in args):
            self.x = args[0]
            self.y = args[1]
        else:
            raise TypeError("Invalid arguments. Must be either two floats or a numpy array of shape (2,)")
            
 
    def __str__(self) -> str:
        return f"Vetor2D: [x={self.x}, y={self.y}]"
    
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def get_array(self) -> np.ndarray:
        return np.array([self.x, self.y])
    

class Vector3D(Vector2D):
    @overload
    def __init__(self, x: int|float, y: int|float, z: int|float) -> None: ...
        
    @overload
    def __init__(self, array: np.ndarray) ->None: ...

    def __init__(self, *args: Union[int|float, np.ndarray]) -> None:
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            array = args[0]
            if array.shape != (3,):
                raise ValueError("Array must have shape (3,)")
            super().__init__(float(array[0]), float(array[1]))
            self.z = float(array[2])
        elif len(args) == 3 and all(isinstance(arg, float|int) for arg in args):
            super().__init__(args[0], args[1])
            self.z = args[2]
        else:
            raise TypeError("Invalid arguments. Must be either three floats or a numpy array of shape (3,)")
    
    def __str__(self) -> str:
        return f"Vector3D: [x={self.x}, y={self.y}, z={self.z}]"


class Vector6D(Vector3D):
    @overload
    def __init__(self, x: int|float, y: int|float, z: int|float, rx: int|float, ry: int|float, rz: int|float) -> None: ...
        
    @overload
    def __init__(self, array: np.ndarray) ->None: ...

    def __init__(self, *args: Union[int|float, np.ndarray]) -> None:
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            array = args[0]
            if array.shape != (6,):
                raise ValueError("Array must have shape (3,)")
            super().__init__(float(array[0]), float(array[1]), float(array[2]))
            self.rx = float(array[3])
            self.ry = float(array[4])
            self.rz = float(array[5])
        elif len(args) == 6 and all(isinstance(arg, float|int) for arg in args):
            super().__init__(args[0], args[1], args[3])
            self.rx = args[3]
            self.ry = args[4]
            self.rz = args[5]
        else:
            raise TypeError("Invalid arguments. Must be either six floats or a numpy array of shape (6,)")

    def __str__(self) -> str:
        return f"Vector 6D: {str(self.get_array())}"
    
    def get_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.rx, self.ry, self.rz])
    
