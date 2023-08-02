import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read the images
lImage = cv.imread('image50.jpg', cv.IMREAD_GRAYSCALE)
lImage = cv.equalizeHist(lImage)
lImage = cv.GaussianBlur(lImage, (5, 5), 1.5)

for i in range(50, 70):
    rImage = cv.imread('image'+str(i)+'.jpg', cv.IMREAD_GRAYSCALE)

    # Apply Histogram Equalization for better contrast
    rImage = cv.equalizeHist(rImage)

    # Apply Gaussian Blur for noise reduction
    rImage = cv.GaussianBlur(rImage, (5, 5), 1.5)
    # Create the StereoBM object with adjusted parameters
    # Use the distance differences to scale the numDisparities parameter
    blockSize = 21
    stereo = cv.StereoBM_create(numDisparities=16, blockSize=blockSize)

    # Compute the disparity map
    depth = stereo.compute(lImage, rImage)

    # Convert the depth map to a 32-bit floating-point format
    # depth = np.float32(depth)

    depth = cv.rotate(depth, cv.ROTATE_90_COUNTERCLOCKWISE)

    # Show the original images
    # cv.imshow("Left", lImage)
    # cv.imshow("Right", rImage)
    # Converting to float32
    depth = depth.astype(np.float32)

    # Scaling down the disparity values and normalizing them
    depth = (depth / 16.0 - 0) / 16
    # Display the depth map with jet colormap
    cv.imshow('l', depth)
    cv.waitKey(0)
    plt.imshow(depth)
    plt.axis('off')
    plt.show()

# Save the depth map to a file
cv.imwrite("depth_map.jpg", depth)
