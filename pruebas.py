class Piece:
    def __init__(self, name: str, coordinates: tuple, color: tuple):
        self.name = name
        self.coordinates = coordinates
        self.color = color



a = Piece(name='a', coordinates=(0,9,9), color=(255,0,0))
print(a.__dict__)