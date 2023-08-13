import cv2

dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dict, parameters)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    (corners, ids, rejected) = detector.detectMarkers(frame)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    