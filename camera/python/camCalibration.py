import numpy as np
import cv2

# Define the size of the calibration pattern (number of inner corners)
pattern_size = (9, 6)  # Change this to match your calibration pattern

# Prepare object points (assuming the chessboard pattern lies on a plane at Z=0)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
obj_points = []  # 3D points in real world space
img_points = []  # 2D points in image plane

# Capture images from the camera for calibration
cap = cv2.VideoCapture(0)  # Change the camera index if needed
print("cap")


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret_corners, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret_corners:
        obj_points.append(objp)
        img_points.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, pattern_size, corners, ret_corners)
        cv2.imshow('Calibration', frame)
        cv2.waitKey(100)

    if len(img_points) >= 10:  # Adjust this threshold as needed
        break

cap.release()
cv2.destroyAllWindows()

# Calibrate the camera
ret, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Print the camera matrix
print("Camera Matrix:")
print(camera_matrix)

# Save the camera matrix to a text file
with open('camera_matrix.txt', 'w') as f:
    for row in camera_matrix:
        f.write('\t'.join(map(str, row)) + '\n')
print("finish")
