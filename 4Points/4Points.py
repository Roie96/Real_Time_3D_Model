import numpy as np
import matplotlib.pyplot as plt

# Define the 3D points as a numpy array
points_3d_im1 = np.array([[-1, -1, 1],
                      [-1, 1, 1],
                      [1, -1, 1],
                      [1, 1, 1]])

points_3d_im2 = np.array([[-1, 1, 1],
                      [-1, 3, 1],
                      [1, 1, 1],
                      [1, 3, 1]])

# Define the camera matrix
camera_matrix = np.array([[500, 0, 320],
                          [0, 500, 240],
                          [0, 0, 1]])

# Perform the projection
projected_points_homogeneous_im1 = np.dot(camera_matrix, points_3d_im1.T).T
projected_points_homogeneous_im2 = np.dot(camera_matrix, points_3d_im2.T).T

# Normalize the homogeneous coordinates
projected_points_im1 = projected_points_homogeneous_im1[:, :2] / projected_points_homogeneous_im1[:, 2:]
projected_points_im2 = projected_points_homogeneous_im2[:, :2] / projected_points_homogeneous_im2[:, 2:]

depth = camera_matrix[1][1] * 2 / np.abs(projected_points_im1[:,1] - projected_points_im2[:,1])
print(depth)