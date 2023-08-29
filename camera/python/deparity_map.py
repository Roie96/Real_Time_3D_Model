import cv2
import numpy as np
from visualisation import visualize_point_cloud
from numba import jit, uint8

import time

def find_Disparity(left, right, window_size, median_size):
    cost_volume = generate_census(left, right, window_size, median_size)
    aggregated_cost = np.empty_like(cost_volume)
    for d in range(31):
        aggregated_cost[:, window_size//2:, d] = cv2.medianBlur(cost_volume[:,window_size//2:,d], median_size)
    return np.argmin(aggregated_cost, axis=2)

@jit(uint8[:,:,:](uint8[:, :], uint8[:, :], uint8, uint8), nopython=True, fastmath=True)
def generate_census(left, right, window_size, median_size):
    max_disparity = 31
    left_census = np.empty((left.shape[0], left.shape[1], window_size**2), dtype=np.uint8)
    right_census = np.empty((right.shape[0], right.shape[1], window_size**2), dtype=np.uint8)
    for i in range(window_size // 2, left.shape[0] - window_size // 2):
        for j in range(window_size // 2, left.shape[1] - window_size // 2):
            left_window = left[i - window_size // 2:i + window_size // 2 + 1,
                          j - window_size // 2:j + window_size // 2 + 1]
            right_window = right[i - window_size // 2:i + window_size // 2 + 1,
                           j - window_size // 2:j + window_size // 2 + 1]
            left_census[i, j] = (left_window >= left[i, j]).ravel()
            right_census[i, j] = (right_window >= right[i, j]).ravel()

    cost_volume = np.empty((left.shape[0], left.shape[1], max_disparity),dtype=np.uint8)

    cost_volume[:, :, 0] = np.sum(left_census[:, :] != right_census[:, :], axis=2)
    for d in range(1, max_disparity):
        cost_volume[d:, :, d] = np.sum(left_census[:-d, :] != right_census[d:, :], axis=2)

    return cost_volume

def generate_point_cloud(disparity_map, baseline, focal_length):
    rows, cols = disparity_map.shape
    cloud = np.empty((disparity.shape[0], disparity.shape[1], 3), dtype=np.float32)
    cloud[:, :, 0] = np.arange(cloud.shape[0]*cloud.shape[1]).reshape(cloud.shape[0], cloud.shape[1]) % cloud.shape[1]
    cloud[:, :, 1] = (np.arange(cloud.shape[0]*cloud.shape[1]).reshape(cloud.shape[1], cloud.shape[0]) % cloud.shape[0]).T
    cloud[:, :, 2] = baseline*focal_length/disparity_map

    # remove outliers
    cloud = cloud[np.isfinite(cloud[:, :, 2])]
    threshold = np.percentile(cloud[:, 2], 50)
    cloud = cloud[cloud[:, 2] < threshold]
    cloud[:, 0]  = -(cloud[:, 0]-cols/2)/focal_length*cloud[:, 2]
    cloud[:, 1]  = -(cloud[:, 1]-rows/2)/focal_length*cloud[:, 2]
    return cloud

if __name__ == '__main__':

    path = "../videos/rot_low.h264"
    cap = cv2.VideoCapture(path)
    for i in range(40):
        cap.read()
    ret1,leftImage = cap.read()
    #slice the human in image
    leftImage = leftImage[0:500,250:400]
    
    for i in range(8):
        cap.read()
    ret2,rightImage = cap.read()

    #slice the human in image
    rightImage = rightImage[0:500,250:400]
    camera_matrix = np.loadtxt('../data/K.txt')
    focal_length = camera_matrix[0][0]

    left = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)

    #Var for disparty map
    baseline = 2
    window_size=7
    threshold = 0.9
    median_size = 13 #remove noise

    # Calculate disparity

    start_total = time.time()

    start_disparity = time.time()
    disparity = find_Disparity(left, right, window_size, median_size)
    end_disparity = time.time()
    cv2.imwrite("disp.png",disparity)
    print("Disparity calculation time:", end_disparity - start_disparity, "seconds")

    # Apply Gaussian blur

    start_blur = time.time()
    blur_kernel_size = (5, 5)
    sigma = 0.8
    disparity = cv2.GaussianBlur(disparity.astype(np.float32), blur_kernel_size, sigma)
    end_blur = time.time()
    print("Blur time:", end_blur - start_blur, "seconds")

    # Generate point cloud and visualize

    start_point_cloud = time.time()
    point_cloud = generate_point_cloud(disparity, baseline, focal_length)
    end_point_cloud = time.time()
    print("Point cloud generation time:", end_point_cloud - start_point_cloud, "seconds")

    end_total = time.time()
    print("Total execution time:", end_total - start_total, "seconds")

    
    # visualize_point_cloud(point_cloud[7500:,:])

    



    