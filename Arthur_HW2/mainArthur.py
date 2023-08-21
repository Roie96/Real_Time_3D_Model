import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

def compute_depth_map(disparity_map, baseline, focal_length):

    depth_map = np.zeros_like(disparity_map, dtype=np.float32)
    depth_map[disparity_map > 0] = -baseline * focal_length / disparity_map[disparity_map > 0]

    return depth_map


def find_Disparity(left, right, window_size, threshold, median_size):

    # with open('set_'+str(n)+'/max_disp.txt', 'r') as f:
    #     max_disparity = f.readline()
    #     max_disparity = int(max_disparity)
    max_disparity = 31
    # left_census = np.zeros_like(left, dtype=np.ndarray)
    # right_census = np.zeros_like(right, dtype=np.ndarray)

    left_census = np.ndarray(shape=(left.shape[0], left.shape[1], window_size[0] * window_size[1]))#, dtype=bool)
    right_census = np.ndarray(shape=(right.shape[0], right.shape[1], window_size[0] * window_size[1]))#, dtype=bool)
    for i in range(window_size[0] // 2, left.shape[0] - window_size[0] // 2):
        for j in range(window_size[1] // 2, left.shape[1] - window_size[1] // 2):
            left_window = left[i - window_size[0] // 2:i + window_size[0] // 2 + 1,
                          j - window_size[1] // 2:j + window_size[1] // 2 + 1]
            right_window = right[i - window_size[0] // 2:i + window_size[0] // 2 + 1,
                           j - window_size[1] // 2:j + window_size[1] // 2 + 1]

            left_census[i, j] = np.array([left_window >= left[i, j]]).flatten()
            right_census[i, j] = np.array([right_window >= right[i, j]]).ravel()

    left_census = left_census.astype(np.uint8)
    right_census = right_census.astype(np.uint8)

    cost_volume = np.zeros((left.shape[0], left.shape[1], max_disparity),dtype=np.uint8)
    inverse_cost_volume = np.zeros_like(cost_volume)
    for d in range(max_disparity):
        for y in range(window_size[0] // 2, left.shape[0] - window_size[0] // 2): #--
            for x in range(window_size[1] // 2, left.shape[1] - window_size[1] // 2):  #--

                if x - d >= (window_size[1] // 2):  #0--
                    cost_volume[y, x, d] = np.count_nonzero(left_census[y, x] != right_census[y, x - d])
                if x+d < (left.shape[1] - window_size[1]// 2): #left.shape[1]--
                    inverse_cost_volume[y, x, d] = np.count_nonzero(left_census[y, x+d] != right_census[y, x])

    # Apply aggregation
    # disparity = cv2.medianBlur(initial_disparity.astype(np.uint8), median_size)
    # disparity_inv = cv2.medianBlur(initial_disparity_inverse.astype(np.uint8), median_size)

    aggregated_cost = np.zeros_like(cost_volume)
    aggregated_cost_inv = np.zeros_like(cost_volume)
    for d in range(max_disparity):
        aggregated_cost[:, window_size[1]//2:, d] = cv2.medianBlur(cost_volume[:,window_size[1]//2:,d], median_size)
        #aggregated_cost[:, :, d] = cv2.medianBlur(cost_volume[:,:,d], median_size)
        aggregated_cost_inv[:,:left.shape[1]-window_size[1]//2, d] = cv2.medianBlur(inverse_cost_volume[:,:left.shape[1]-window_size[1]//2,d], median_size)
        #aggregated_cost_inv[:,:, d] = cv2.medianBlur(inverse_cost_volume[:,:,d], median_size)
        #aggregated_cost[:, window_size[1]//2:, d] = cv2.blur(cost_volume[:,window_size[1]//2:,d], (3,7))
        #aggregated_cost_inv[:,:left.shape[1]-window_size[1]//2, d] = cv2.blur(inverse_cost_volume[:,:left.shape[1]-window_size[1]//2,d], (median_size,median_size))

    # disparity = cost_aggregation(cost_volume, median_size)
    # disparity_inv = cost_aggregation(inverse_cost_volume, median_size)

    disparity = np.argmin(aggregated_cost, axis=2)
    disparity_inv = np.argmin(aggregated_cost_inv, axis=2)
    #disparity[:, :window_size[1]//2] = disparity_inv[:, left.shape[1]-window_size[1]//2-1:] =disparity_inv[:, :window_size[1]//2]= 0

    #Compute the consistency check and filter the initial disparity map
    for i in range(window_size[0]//2, left.shape[0]-window_size[0]//2):
        for j in range(window_size[1]//2, left.shape[1]-window_size[1]//2):
            d = disparity[i, j]
            if j - d >= 0:
                consistency_check_l = abs(d - disparity_inv[i, j - d]) <= threshold
                if not consistency_check_l:
                    disparity[i, j] = 0

            d = disparity_inv[i, j]
            if j + d < right.shape[1]:
                consistency_check_r = abs(d - disparity[i, j + d]) <= threshold
                if not consistency_check_r:
                    disparity_inv[i, j] = 0


    # save before norm- only for save them re-calculation (to use the original disparity values to find depths)
    # disparity = disparity.astype(np.uint8)
    # disparity_inv = disparity_inv.astype(np.uint8)
    # cv2.imwrite('disp_left.jpg', disparity)
    # cv2.imwrite('disp_right.jpg', disparity_inv)

    disparity_normalized = ((disparity - np.min(disparity)) / (np.max(disparity) - np.min(disparity)) * 255).astype( np.uint8)
    disparity_inv_normalized = ((disparity_inv - np.min(disparity_inv)) / (np.max(disparity_inv) - np.min(disparity_inv)) * 255).astype( np.uint8)
    # cv2.imshow('Disparity  left', disparity_normalized)
    # cv2.imshow('Disparity  right', disparity_inv_normalized)

    # save the display mood of norm values- to submit results!!
    cv2.imwrite('disp_left_n.jpg',disparity_normalized)
    cv2.imwrite('disp_right_n.jpg',disparity_inv_normalized)

    return disparity, disparity_inv

def generate_point_cloud(disparity_map, depth_map, baseline, focal_length):
    rows, cols = disparity_map.shape
    point_cloud = []

    for y in range(rows):
        for x in range(cols):
            disparity = disparity_map[y, x]
            depth = depth_map[y, x]
            if disparity > 0 and depth > 0:
                z = (baseline * focal_length) / disparity
                x_3d = (x - cols/2) * (z / focal_length)
                y_3d = (y - rows/2) * (z / focal_length)
                point_cloud.append([x_3d, y_3d, z])

    return np.array(point_cloud)

def visualize_point_cloud(point_cloud):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__ == '__main__':

    path = "rot_low.h264"
    cap = cv2.VideoCapture(path)
    for i in range(40):
        cap.read()
    ret1,leftImage = cap.read()
    leftImage = cv2.rotate(leftImage, cv2.ROTATE_90_CLOCKWISE)
    leftImage = leftImage[250:400,0:400]
    
    for i in range(8):
        cap.read()
    ret2,rightImage = cap.read()
    rightImage = cv2.rotate(rightImage, cv2.ROTATE_90_CLOCKWISE)
    rightImage = rightImage[250:400,0:400]

    camera_matrix = np.loadtxt('K.txt')
    focal_length = camera_matrix[0][0]

    left = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)

    baseline = -10
    window_size=[7,7]
    threshold = 0.9
    median_size= 9
    disparity, disparity_inv = find_Disparity(left, right, window_size, threshold, median_size)

    # # Define the file name
    # csv_file_name = "data.csv"

    # # Open the CSV file in write mode
    # with open(csv_file_name, mode='w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
        
    #     # Write each row of the array as a CSV row
    #     for row in disparity:
    #         csv_writer.writerow(row)

    # print(f"Array saved to {csv_file_name}")


    depth_left = compute_depth_map(disparity, baseline, focal_length)
    depth_right = compute_depth_map(disparity_inv, baseline, focal_length)

    depth_left = depth_left.astype(np.uint8)
    depth_right = depth_right.astype(np.uint8)

    cv2.imwrite('depth_left.jpg', depth_left)
    cv2.imwrite('depth_right.jpg',depth_right)

    point_cloud = generate_point_cloud(disparity,depth_left,baseline,focal_length)
    
    # After generating the depth_left and depth_right matrices

    # # Set your depth range thresholds for the woman's region
    # woman_depth_lower = 100  # Adjust this based on your scene
    # woman_depth_upper = 300  # Adjust this based on your scene

    # # Create a mask for points within the woman's depth range
    # woman_mask = (depth_left >= woman_depth_lower) & (depth_left <= woman_depth_upper)

    # Create a copy of your point_cloud
    woman_point_cloud = point_cloud.copy()

    # Apply the mask to keep only the woman's points
    # woman_point_cloud = woman_point_cloud[woman_mask.flatten()]

    min_xyz = woman_point_cloud.min(axis=0)
    max_xyz = woman_point_cloud.max(axis=0)
    scaled_woman_point_cloud = 400 + (woman_point_cloud - min_xyz) * (500 - 100) / (max_xyz-min_xyz)

    # Visualize the point cloud with only the woman's points in white
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(scaled_woman_point_cloud[:, 0], scaled_woman_point_cloud[:, 1], scaled_woman_point_cloud[:, 2], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    

  

    # disparity =  cv2.imread('disp_left.jpg', cv2.COLOR_BGR2GRAY)
    # disparity_inv = cv2.imread('disp_right.jpg', cv2.COLOR_BGR2GRAY)
    # left_d = cv2.imread('results/set_'+'/depth_left.jpg')
    # right_d = cv2.imread('results/set_'+'/depth_right.jpg')
    # height, width = left.shape[:2]


