import numpy as np
import cv2
from matplotlib import pyplot as plt


cap1 = cv2.VideoCapture("C:/Users/talha/OneDrive/Desktop/תואר/RTlab/try/close.h264")
counter = 0
while counter <= 27:
    ret2, frame2 = cap1.read()

    # Check if frames have been read successfully
    if not ret2:
        print("Error reading video frames.")
        exit()

    if counter % 1 == 0:
        rotated_image1 = cv2.rotate(frame2, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite("image"+str(counter)+".jpg", rotated_image1)
    counter += 1
# ret1, frame1 = cap1.read()




# # Ensure both frames have the same dimensions
# if frame1.shape != frame2.shape:
#     frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
#
# # Create a subplot with 1 row and 2 columns, and activate the first subplot
# plt.subplot(1, 2, 1)
# plt.imshow(frame1, cmap='viridis')
# plt.title('Figure 1')
# plt.axis('off')
#
# # Activate the second subplot
# plt.subplot(1, 2, 2)
# plt.imshow(frame2, cmap='plasma')
# plt.title('Figure 2')
# plt.axis('off')
#
# image1 = frame1
# image2 = frame2
#
# # Show the plots
# plt.tight_layout()
# plt.show()
#
# # Rotate and place image1 on the canvas
# rotated_image1 = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)
# rotated_image2 = cv2.rotate(image2, cv2.ROTATE_90_CLOCKWISE)
# cv2.imwrite("image1.jpg", rotated_image1)
# cv2.imwrite("image2.jpg", rotated_image2)
#
