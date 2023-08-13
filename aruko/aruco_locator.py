import os
import cv2
import numpy as np
import csv
from Constants import Constants
from file import load_camera_data_json

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
cap = cv2.VideoCapture("C:/Users/talha/RealTimeProject/Real_Time_3D_Model/aruko/low_res.h264")
marker_size = 16.6
cam_dir = "C:/Users/talha/RealTimeProject/Real_Time_3D_Model/mapping_wrappers/camera_config/pi/camera_data_480p.json"
cam_mat, dist_coeff, _ = load_camera_data_json(cam_dir)

base_marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                               [marker_size / 2, marker_size / 2, 0],
                               [marker_size / 2, -marker_size / 2, 0],
                               [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

csv_file_name = "output.csv"
data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break 
    (corners, ids, rejected) = detector.detectMarkers(frame)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    if corners:
        cord = []
        for corners_set, id in zip(corners, ids):
            if id != 64:
                continue
            _, rot_vec, t = cv2.solvePnP(base_marker_points, corners_set, cam_mat, dist_coeff, False,
                                         cv2.SOLVEPNP_IPPE_SQUARE)
            x, y, z = t.ravel()  # Extract x, y, z coordinates from t
            cord.append([x, y, z])  # Append to the cord list
            # cv2.putText(frame, str(t), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)
    data.extend(cord)  # Extend the data list with cord
    # cv2.imshow('frame', frame)
    # cv2.waitKey(1)
    
with open(csv_file_name, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    for row in data:
        csv_writer.writerow(row)
