import matplotlib.pyplot as plt
import numpy as np
import cv2
import open3d as o3d

def visualize_point_cloud(point_cloud):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([cloud])


    # min_xyz = point_cloud.min(axis=0)
    # max_xyz = point_cloud.max(axis=0)
    # scaled_woman_point_cloud = 500 + (point_cloud - min_xyz) * (500 - 100) / (max_xyz-min_xyz)
    # new_cloud = point_cloud[[point_cloud[:, 2] > num1]]
    # row_indices, col_indices = np.where(point_cloud > num1)

    # Create the new point cloud array
    # new_point_cloud = np.column_stack((col_indices, row_indices, np.ones_like(row_indices)))
    # new_point_cloud = (new_point_cloud - num1)*(5000)/(255 - num1 )

    # Overlay the point cloud on the original image
    # for point in new_point_cloud:
    #     x, y, z = point
    #     cv2.circle(leftImage, (x, y), 3, (255, 0, 0), -1)

    # # Display the image with the overlay
    # cv2.imshow("Point Cloud Overlay", leftImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



