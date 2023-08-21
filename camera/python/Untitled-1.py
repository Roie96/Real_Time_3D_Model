import cv2

cap1 = cv2.VideoCapture(2)
while True:
    ret, frame = cap1.read()
    if not ret:
        exit()
    cv2.imshow('t', frame)
    cv2.waitKey(1)