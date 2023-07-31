import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read the images
lImage = cv.imread('image0.0.jpg', cv.IMREAD_GRAYSCALE)
lImage = cv.equalizeHist(lImage)
lImage = cv.GaussianBlur(lImage, (5, 5), 0)

for i in range(1, 6):
    rImage = cv.imread('image'+str(i)+'.0.jpg', cv.IMREAD_GRAYSCALE)

    # Apply Histogram Equalization for better contrast
    rImage = cv.equalizeHist(rImage)

    # Apply Gaussian Blur for noise reduction
    rImage = cv.GaussianBlur(rImage, (5, 5), 0)

    # Create the StereoBM object with adjusted parameters
    # Use the distance differences to scale the numDisparities parameter
    blockSize = 21
    stereo = cv.StereoBM_create(numDisparities=16, blockSize=blockSize)

    # Compute the disparity map
    depth = stereo.compute(lImage, rImage)

    # Convert the depth map to a 32-bit floating-point format
    depth = np.float32(depth)

    depth = cv.rotate(depth, cv.ROTATE_90_COUNTERCLOCKWISE)

    # Show the original images
    # cv.imshow("Left", lImage)
    # cv.imshow("Right", rImage)

    # Display the depth map with jet colormap
    plt.imshow(depth, cmap='jet')
    plt.axis('off')
    plt.show()

# Save the depth map to a file
cv.imwrite("depth_map.jpg", depth)
