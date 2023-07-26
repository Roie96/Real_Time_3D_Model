import os

import cv2
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from analysis.MotionArrowsOverlay import overlay_arrows_combined_frames
from depth_trackers.depth_analysis import depth_from_h264_vectors

from mapping.utils.Constants import Constants
from mapping.utils.file import load_camera_data_json

allData = np.empty((0, 3), dtype=np.float32)
max_depth = 700

def visualize_3d_points(points3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], c=points3d[:, 2], cmap='jet')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def saveData(data, fileName):
    with open(fileName, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)  # create file
        csv_writer.writerows(data)


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


save_video: bool = False  # turn off when saved video not required
show_video: bool = False  # turn off when video window not required
top_down: bool = True  # turn on when working on topdown view
cam_dir = os.path.join(Constants.ROOT_DIR, "mapping_wrappers/camera_config/pi/camera_data_480p.json")
cam_mat, dist_coeff, _ = load_camera_data_json(cam_dir)
path = os.path.join(Constants.ROOT_DIR, "results/depth_test1")
# angles1 = np.loadtxt(os.path.join(path, "tello_angles1.csv"))
# angles2 = np.loadtxt(os.path.join(path, "tello_angles2.csv"))
# best_pair = generate_angle_pairs(angles1, angles2)  # doesn't work
cap1 = cv2.VideoCapture(os.path.join(path, "far.h264"))
cap2 = cv2.VideoCapture(os.path.join(path, "close.h264"))
hightFile = pd.read_csv(os.path.join(path, "tello_heights_far.csv"))
hightFile = np.array(hightFile, dtype=float)

if save_video:
    writer = cv2.VideoWriter(os.path.join(path, "close.mp4"), -1, 40, (640, 480))
else:
    writer = None  # just to stop warning
# Initialize the feature detector (e.g., ORB, SIFT, etc.)
detector = cv2.ORB_create(nfeatures=1000)
depth_frame = np.zeros((700, 700, 3))
ret1, frame1 = cap1.read()
currentIndex = 0
nextIndex = 0

frameList = []
#Get all unqiue frames by hight
while nextIndex < hightFile.shape[0] - 1:  
    currentIndex = nextIndex
    ret2, frame2 = ret1, frame1
    while nextIndex < hightFile.shape[0] - 1 and hightFile[nextIndex] - hightFile[currentIndex] < 1:
        ret1, frame1 = cap1.read()
        nextIndex += 1
    ret1, frame1 = cap1.read()
    frameList.append((frame2, hightFile[currentIndex]))
    nextIndex += 1
    
    
i=0  
while True:
    if(i+1 > len(frameList)):
        break
    frame1 = frameList[i]
    frame2 = frameList[i+1]
    #frame3 = frameList[i+2]
    # cap2.set(cv2.CAP_PROP_POS_FRAMES, pair - 1)  # seek to best pair, doesn't work
    #if not ret1:
    #    if save_video:
    #        writer.release()
    #    break
    # ret2, frame2 = cap2.read()
    # combined_frame = np.concatenate((frame1, frame2), axis=1)
    # cv2.imshow("frames", combined_frame)
    
    
    # ORB
    gray1 = cv2.cvtColor(frame1[0], cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2[0], cv2.COLOR_BGR2GRAY)
    #gray3 = cv2.cvtColor(frame3[0], cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)
    #keypoints3, descriptors3 = detector.detectAndCompute(gray3, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Perform the matching
    matches = matcher.match(descriptors1, descriptors2)
    
    
    #print(len(matches), "matches using ORB")
    # show matches
    # frame_with_matches = cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, matches[:100], None)
    # cv2.imshow("ORB matches", frame_with_matches)
    # cv2.waitKey(0)
    # continue
    # show depth
    
    distances = np.array([match.distance for match in matches])
    thershold = np.percentile(distances, 90)
    points1 = np.array([keypoints1[match.queryIdx].pt for match in matches if match.distance<thershold])
    points2 = np.array([keypoints2[match.trainIdx].pt for match in matches if match.distance<thershold])
    
    #points1 = np.array([keypoints1[match.queryIdx].pt for match in matches])
    #points2 = np.array([keypoints2[match.trainIdx].pt for match in matches])
 
   
    diffHight = abs(frame1[1]- frame2[1])
    depth = depth_from_h264_vectors(np.hstack((points1, points2)), cam_mat, diffHight)# you might want to save one of these for the topdown view
    

    if top_down:
        depth_frame1, data1 = topdown_view(np.hstack((points1, depth[:, None])))
        allData = np.append(allData, data1, axis=0)
        
        #threshold = 0  # Change if needed
        #mask = allData[:, 2] > threshold 
        #filteredData = allData[mask]
        
    else:
        depth_frame = frame1.copy()
        int_points1 = points1.astype(int)
        depth_color = np.clip(depth * 255 / 500, 0, 255)[:,
                      None]  # clip  values from 0 to 5m and scale to 0-255(color range)
        for color, point in zip(depth_color, int_points1):
            cv2.rectangle(depth_frame, point[::] - 5, point[::] + 5, color, -1)
    if show_video:
        cv2.imshow("depth ORB", depth_frame)
        cv2.waitKey(1)  # need some minimum time because opencv doesnt work without it

    if save_video:
        writer.write(depth_frame)
    i += 2
random_indices = np.random.choice(allData.shape[0], int(allData.shape[0]/3), replace=False)
filteredData = allData[random_indices] 
#filteredData = allData   
saveData(filteredData, os.path.join(path, "3dPoints.csv"))
visualize_3d_points(filteredData)

# similar method that uses motion vectors
# points3d = triangulate_points(keypoints1, keypoints2, matches, 60, cam_mat, dist_coeff)
# depth_frame = frame1.copy()
# frame1[(points3d[:, :2]).astype(int)] = 255*(points3d[:, 3]/np.max(points3d[:, 3]))

# Motion Vectors
# vectors = ffmpeg_encode_extract(frame1, frame2, subpixel=False)
# print(len(vectors), "matches using MV")
# combined_frame = np.concatenate((frame1, frame2), axis=1)
# combined_frame = overlay_arrows_combined_frames(combined_frame, vectors, max_vectors=50)
# cv2.imshow("MV matches", combined_frame)
# cv2.waitKey(0)
