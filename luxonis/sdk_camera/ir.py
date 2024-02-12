from depthai_sdk import OakCamera

with OakCamera() as oak:
    left = oak.create_camera('left', resolution='480p')
    right = oak.create_camera('right', resolution='480p')
    stereo = oak.create_stereo(left=left, right=right, resolution="480p")

    # Automatically estimate IR brightness and adjust it continuously
    # stereo.(auto_mode=True, continuous_mode=True)

    oak.visualize([stereo.out.disparity, left])
    oak.start(blocking=True)