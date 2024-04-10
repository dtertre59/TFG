import depthai as dai
import numpy as np
import cv2
import sys

import open3d as o3d


# Continue with the DepthAI and Open3D setup as before
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

depth = pipeline.create(dai.node.StereoDepth)
pointcloud = pipeline.create(dai.node.PointCloud)
sync = pipeline.create(dai.node.Sync)
xOut = pipeline.create(dai.node.XLinkOut)
xOut.input.setBlocking(False)


camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
# camRgb.setIspScale(1,3)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")

monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")


depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

depth.setLeftRightCheck(True)
depth.setExtendedDisparity(False)
depth.setSubpixel(True)
# depth.setLeftRightCheck(False)
# depth.setExtendedDisparity(True)
# depth.setSubpixel(False)
depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)

config = depth.initialConfig.get()
config.postProcessing.thresholdFilter.minRange = 100
config.postProcessing.thresholdFilter.maxRange = 1000
depth.initialConfig.set(config)

# otras caracteristicas
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)

depth.depth.link(pointcloud.inputDepth)


camRgb.isp.link(sync.inputs["rgb"])
pointcloud.outputPointCloud.link(sync.inputs["pcl"])
pointcloud.initialConfig.setSparse(False)
sync.out.link(xOut.input)
xOut.setStreamName("out")

inConfig = pipeline.create(dai.node.XLinkIn)
inConfig.setStreamName("config")
inConfig.out.link(pointcloud.inputConfig)


xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")

depth.disparity.link(xoutDepth.input)

with dai.Device(pipeline) as device:
    # depth camara
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    q = device.getOutputQueue(name="out", maxSize=4, blocking=False)
    pclConfIn = device.getInputQueue(name="config", maxSize=4, blocking=False)


    pcd = o3d.geometry.PointCloud()
    

    first = True
    rot = 0
    while device.isPipelineRunning():
        
        inMessage = q.get()
        inColor = inMessage["rgb"]
        inPointCloud = inMessage["pcl"]
        cvColorFrame = inColor.getCvFrame()
        
        cvRGBFrame = cv2.cvtColor(cvColorFrame, cv2.COLOR_BGR2RGB)

        # in depth camera
        inDepth = qDepth.tryGet()

        # imagen depth
        if inDepth is not None:
            frame = inDepth.getCvFrame()
            # pintar frame
            cv2.imshow("depth", frame)

        # Nube de puntos
        if inPointCloud:
            # asignamos puntos
            points = inPointCloud.getPoints().astype(np.float64)
            pcd.points = o3d.utility.Vector3dVector(points)
            # asignamos colores
            colors = (cvRGBFrame.reshape(-1, 3) / 255.0).astype(np.float64)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # SEGUNDO METODO
            pointcloud = o3d.geometry.PointCloud()
            pointcloud.points = o3d.utility.Vector3dVector(points)
            pointcloud.colors = pcd.colors
            # Visualizar la nube de puntos
            o3d.visualization.draw_geometries([pointcloud])

    cv2.destroyAllWindows()