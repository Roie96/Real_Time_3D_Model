import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import csv
# from analysis.MotionArrowsOverlay import overlay_arrows_combined_frames
from depth_analysis import depth_from_h264_vectors
from pyntcloud import PyntCloud

from Constants import Constants
from file import load_camera_data_json
max_depth = 700
def topdown_view(depth: np.ndarray):
    image = np.full((700, 700, 3), 255, dtype=np.uint8)
    """
    This function returns an image (WxHx3 matrix of uint8),
    representing a top-down view of the given depth map
    remember to scale the points to the scale of the image resolution you selected, and to center them around the image center.
    """
    # depth map to cloud, clip it at 700cm to prevent outliers
    depth[:, 2] = np.clip(depth[:, 2], 0, 700)

    # Scale and center the 3D points around the image center
    depth[:, :2] = np.squeeze(cv2.undistortPoints(depth[None, :, :2], cam_mat, dist_coeff)) * depth[:, 2:]

    centerData = []

    for pixel in depth:
        # Scale the pixel coordinates to fit the 700x700 top-down image
        pixel = np.clip((pixel * 700 / max_depth + 350).astype(int), -700, 700)
        centerData.append(pixel)
        # Draw a white dot on the top_down_image
        cv2.circle(image, (pixel[0], pixel[2]), 1, (0, 0, 0), -1)

    return image, centerData
cam_dir = "C:/Users/talha/RealTimeProject/Real_Time_3D_Model/mapping_wrappers/camera_config/pi/camera_data_480p.json"
cam_mat, dist_coeff, _ = load_camera_data_json(cam_dir)
cap1 = cv2.VideoCapture("aruko/low_res.h264")
hightFile = pd.read_csv("aruko/output.csv")
hightFile = np.array(hightFile, dtype=float)
detector = cv2.ORB_create(nfeatures=1000)
depth_frame = np.zeros((700, 700, 3))
allData = np.empty((0, 3))
index1 = 0
index2 = 0
numDisparities = 16
blockSize = 37
preFilterType = 0
preFilterSize = 27
preFilterCap = 15
textureThreshold = 2
uniquenessRatio = 18
speckleRange = 0
speckleWindowSize = 0
disp12MaxDiff = 12
minDisparity = 5
stereo = cv2.StereoBM_create()
stereo.setNumDisparities(numDisparities)
stereo.setBlockSize(blockSize)
stereo.setPreFilterType(preFilterType)
stereo.setPreFilterSize(preFilterSize)
stereo.setPreFilterCap(preFilterCap)
stereo.setTextureThreshold(textureThreshold)
stereo.setUniquenessRatio(uniquenessRatio)
stereo.setSpeckleRange(speckleRange)
stereo.setSpeckleWindowSize(speckleWindowSize)
stereo.setDisp12MaxDiff(disp12MaxDiff)
stereo.setMinDisparity(minDisparity)

for i in range(10):
    cap1.read()
    index1+=1
for i in range(2):
    ret1, frame1 = cap1.read()
    index1+=1
    index2 = index1
    for i in range(10):
        cap1.read()
        index2 +=1
    ret2, frame2 = cap1.read()
    index2 +=1
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # keypoints1, descriptors1 = detector.detectAndCompute(gray1.T, None)
    # keypoints2, descriptors2 = detector.detectAndCompute(gray2.T, None)
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # # Perform the matching
    # matches = matcher.match(descriptors1, descriptors2)
    # # show matches
    # # frame_with_matches = cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, matches[:100], None)
    # # cv2.imshow("ORB matches", frame_with_matches)
    # # cv2.waitKey(0)
    # # continue
    # # show depth
    # points1 = np.array([keypoints1[match.queryIdx].pt for match in matches])

    

    disparity = stereo.compute(gray1.T, gray2.T).T
    diffHight = np.abs(hightFile[index1][1] - hightFile[index2][1])
    depth = cam_mat[1,1]*diffHight/disparity

    # Calculate the updated depth values
    updated_depth = depth + hightFile[index1][1]

    # depth map to cloud, clip it at 700cm to prevent outliers
    updated_depth[:, 2] = np.clip(depth[:, 2], 0, 700)
    image = np.full((700, 700, 3), 255, dtype=np.uint8)
    centerData = []
    # Scale and center the 3D points around the image center
    updated_depth[:, :2] = np.squeeze(cv2.undistortPoints(updated_depth[None, :, :2], cam_mat, dist_coeff)) * updated_depth[:, 2:]
    for pixel in updated_depth:
        # Scale the pixel coordinates to fit the 700x700 top-down image
        pixel_scaled = np.clip((pixel * 700 / max_depth + 350).astype(int), -700, 700)
        centerData.append(pixel_scaled)
        # Draw a white dot on the top_down_image
        cv2.circle(image, (pixel_scaled[0], pixel_scaled[2]), 1, (0, 0, 0), -1)

    # Append the updated depth values to allData
    allData = np.append(allData, updated_depth)
# csv_file_name = "cloudMap.csv"
# with open(csv_file_name, mode='w', newline='') as csv_file:
    # csv_writer = csv.writer(csv_file)
    # for row in allData:
        # csv_writer.writerow([row])

