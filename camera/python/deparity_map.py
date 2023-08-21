import csv
import cv2
import numpy as np
from visualisation import visualize_point_cloud

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

def compute_depth_map(disparity_map, baseline, focal_length):

    depth_map = np.zeros_like(disparity_map, dtype=np.float32)
    depth_map[disparity_map > 0] = -baseline * focal_length / disparity_map[disparity_map > 0]

    return depth_map

def generate_point_cloud(disparity_map, baseline, focal_length):
    rows, cols = disparity_map.shape
    cloud = np.empty((disparity.shape[0], disparity.shape[1], 3), dtype=np.float32)
    cloud[:, :, 0] = np.arange(cloud.shape[0]*cloud.shape[1]).reshape(cloud.shape[0], cloud.shape[1]) % cloud.shape[1]
    cloud[:, :, 1] = (np.arange(cloud.shape[0]*cloud.shape[1]).reshape(cloud.shape[1], cloud.shape[0]) % cloud.shape[0]).T
    cloud[:, :, 2] = baseline*focal_length/disparity_map
    cloud = cloud[np.isfinite(cloud[:, :, 2])]
    # threshold = np.percentile(cloud[:, 2], 90)
    # cloud = cloud[cloud[:, 2] < threshold]
    cloud[:, 0]  = (cloud[:, 0]-cols/2)/focal_length*cloud[:, 2]
    cloud[:, 1]  = (cloud[:, 1]-rows/2)/focal_length*cloud[:, 2]
    return cloud

if __name__ == '__main__':

    path = "../videos/rot_low.h264"
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

    camera_matrix = np.loadtxt('../data/K.txt')
    focal_length = camera_matrix[0][0]

    left = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)

    baseline = 10
    window_size=[7,7]
    threshold = 0.9
    median_size= 9
    disparity, disparity_inv = find_Disparity(left, right, window_size, threshold, median_size)

    depth_left = compute_depth_map(disparity, baseline, focal_length)
    depth_right = compute_depth_map(disparity_inv, baseline, focal_length)

    depth_left = depth_left.astype(np.uint8)
    depth_right = depth_right.astype(np.uint8)

    cv2.imwrite('depth_left.jpg', depth_left)
    cv2.imwrite('depth_right.jpg',depth_right)

    point_cloud = generate_point_cloud(disparity, baseline, focal_length)
    # with open('depth_left1.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)

    #     # Write the data to the CSV file row by row
    #     for row in depth_left:
    #         writer.writerow(row)
    visualize_point_cloud(point_cloud,240,focal_length,leftImage)
# cv2.imshow("t",depth_left)
# cv2.waitKey(0)
# cv2.destroyAllWindows(0)

    

