from depthai_sdk import OakCamera

with OakCamera() as oak:
    # Create color camera
    color = oak.create_camera('color')

    # Visualize color camera frame stream
    oak.visualize(color.out.main, fps=True)
    # Start the pipeline, continuously poll
    oak.start(blocking=True)