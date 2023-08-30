import matplotlib.pyplot as plt
import numpy as np
import cv2
import open3d as o3d

def visualize_point_cloud(point_cloud):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([cloud])


