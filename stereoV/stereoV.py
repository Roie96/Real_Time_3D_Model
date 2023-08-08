from itertools import cycle

import numpy as np
import cv2
from tqdm import tqdm

# Check for left and right camera IDs
# These values can change depending on the system
path = "close.h264"
cap = cv2.VideoCapture(path)
# imgL_gray = cv2.equalizeHist(imgL_gray)

kernelSize = 5
sigmaX = 2.457
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


# Setting the updated parameters before computing disparity map
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
for i in range(20):
    cap.read()
frames = []
for i in range(179):
    ret, frame = cap.read()
    frames.append(frame)

# Creating an object of StereoBM algorithm

stereo.setMinDisparity(minDisparity)
for frame in tqdm(cycle(frames)):

    # Capturing and storing left and right camera images
    imgB = frames[0].copy()
    imgT = frame.copy()
    imgR_gray = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY)

    imgR_gray = cv2.equalizeHist(imgR_gray)
    imgL_gray = cv2.equalizeHist(imgL_gray)
    imgR_gray = cv2.GaussianBlur(imgR_gray, (kernelSize, kernelSize), sigmaX)
    imgL_gray = cv2.GaussianBlur(imgL_gray, (kernelSize, kernelSize), sigmaX)

    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(imgL_gray.T, imgR_gray.T).T
    # NOTE: compute returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it
    # is essential to convert it to CV_32F and scale it down 16 times.

    # Converting to float32
    # disparity = disparity.astype(np.float32)
    #
    # Scaling down the disparity values and normalizing them
    # disparity = (disparity / 16.0 - minDisparity) / numDisparities
