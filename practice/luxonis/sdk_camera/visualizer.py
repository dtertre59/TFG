from depthai_sdk import OakCamera
from depthai_sdk.visualize.configs import BboxStyle, TextPosition

with OakCamera() as oak:
    camera = oak.create_camera('color')

    det = oak.create_nn('face-detection-retail-0004', camera)

    visualizer = oak.visualize(det.out.main, fps=True)
    visualizer.detections(
        color=(0, 0, 255),
        thickness=2,
        bbox_style=BboxStyle.RECTANGLE,
        label_position=TextPosition.TOP_LEFT,
    ).text(
        font_color=(0, 0, 255),
        auto_scale=True
    ).tracking(
        line_thickness=5
    )

    oak.start(blocking=True)