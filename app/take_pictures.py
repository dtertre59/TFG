from models.camera import Camera


camera = Camera(width=1920, height=1080, fx= 1498.367322, fy=1497.377563) 
camera.init_rgb()

camera.run_with_condition()