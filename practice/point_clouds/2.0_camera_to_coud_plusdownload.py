import depthai as dai
import numpy as np
import cv2
import sys
from pathlib import Path

import open3d as o3d


# Create pipeline
pipeline = dai.Pipeline()
device = dai.Device()

camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

depth = pipeline.create(dai.node.StereoDepth)
pointcloud = pipeline.create(dai.node.PointCloud)
sync = pipeline.create(dai.node.Sync)
xOut = pipeline.create(dai.node.XLinkOut)
xOut.input.setBlocking(False)


# Properties RGB CAM
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setFps(30)
if 1: camRgb.setIspScale(2, 3)

# For now, RGB needs fixed focus to properly align with depth.
# This value was used during calibration
try:
    calibData = device.readCalibration2()
    lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
    if lensPosition:
        camRgb.initialControl.setManualFocus(lensPosition)
except:
    raise

# Properties MONO CAMS
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.setCamera("left")

monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setCamera("right")


depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

depth.setLeftRightCheck(True)
# depth.setExtendedDisparity(False)
depth.setSubpixel(True)
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

with device:
    device.startPipeline(pipeline)
    
    # depth camara
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    q = device.getOutputQueue(name="out", maxSize=4, blocking=False)
    pclConfIn = device.getInputQueue(name="config", maxSize=4, blocking=False)


    # imagen 
    directory = Path(__file__).resolve().parent
    # image_directory = directory / 'assets'
    # imagen_disparidad = cv2.imread(str(image_directory / 'disparity_1.png'), cv2.IMREAD_GRAYSCALE)
    # cv2.namedWindow('imagen')
    # cv2.imshow('imagen', imagen_disparidad)

    # visualizador
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pcd = o3d.geometry.PointCloud()
    
    vis.add_geometry(pcd)

    # Eje de coordenadas
    coordinateFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=400, origin=[0,0,0])
    vis.add_geometry(coordinateFrame)

    first = True
    rot = 0
    picture_counter = 0
    while device.isPipelineRunning():
        
        inMessage = q.get()
        inColor = inMessage["rgb"]
        inPointCloud = inMessage["pcl"]
        cvColorFrame = inColor.getCvFrame()
        
        cvRGBFrame = cv2.cvtColor(cvColorFrame, cv2.COLOR_BGR2RGB)

        # in rgb
        if inColor:
            frameRGB = inColor.getCvFrame()
            cv2.imshow("depth", frameRGB)

        # in depth camera
        inDepth = qDepth.tryGet()

        # imagen depth
        # if inDepth is not None:
        #     frame = inDepth.getCvFrame()
        #     # pintar frame
        #     cv2.imshow("depth", frame)

        # Nube de puntos
        if inPointCloud:
            # resolucion pixeles = 1280*720
            # hei = inPointCloud.getHeight()
            # wid = inPointCloud.getWidth()
            # print('width: ', wid, '       height: ', hei)
            # asignamos puntos
            points = inPointCloud.getPoints().astype(np.float64)
            # print(points[7200:14399])
            pcd.points = o3d.utility.Vector3dVector(points)
            # asignamos colores
            colors = (cvRGBFrame.reshape(-1, 3) / 255.0).astype(np.float64)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            vis.update_geometry(pcd)

            # # SEGUNDO METODO
            # pointcloud = o3d.geometry.PointCloud()
            # pointcloud.points = o3d.utility.Vector3dVector(points)
            # pointcloud.colors = pcd.colors
            # # Visualizar la nube de puntos
            # o3d.visualization.draw_geometries([pointcloud])
        vis.poll_events()
        vis.update_renderer()

        key = cv2.waitKey(1)
        if key == ord('d'):
            print('export point cloud and rgb photo')
            name = 'april_square_2'
            o3d.io.write_point_cloud(str(directory / 'clouds' / f'{name}_{picture_counter}.ply'), pcd)
            cv2.imwrite(filename=str(directory / 'assets' / f'{name}_{picture_counter}.png'), img=frameRGB)
            picture_counter += 1
        
        if key == ord('q'):
            break

    vis.destroy_window()
    cv2.destroyAllWindows()