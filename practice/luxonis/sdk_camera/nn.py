from depthai_sdk import OakCamera, ResizeMode

with OakCamera() as oak:
    color = oak.create_camera('color')
    # nn = oak.create_nn('mobilenet-ssd', color)
    # nn = oak.create_nn('vehicle-detection-0202', color)
    nn = oak.create_nn('face-detection-retail-0004', color, tracker=True)
    # nn.config_nn(resize_mode='stretch')
    oak.visualize([nn.out.tracker, nn.out.passthrough], fps=True)
    oak.start(blocking=True)