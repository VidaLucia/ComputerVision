import cv2
import mediapipe as mp
import time
import HandTracker as ht


previousTime =0
currentTime = 0
cap = cv2.VideoCapture(0)
detector = ht.HandDetector()
while True:
    ret, frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame)
    if len(lmList) != 0:
        print(lmList[4])
    currentTime = time.time()

    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)