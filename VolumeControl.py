import math

import cv2
import time
import numpy as np
import HandTracker as ht

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
detector = ht.HandDetector(detectionCon=0.8)
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw= False)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1],lmList[4][2]
        x2, y2 = lmList[8][1],lmList[8][2]
        cx,cy = (x1 + x2)//2, (y1 + y2)//2
        cv2.circle(img,(x1,y1),15,(255,0,0),cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)
        print(length)
        if length > 250:
            break
        if length <40:
            cv2.circle(img,(cx,cy), 15, (0, 0, 255), cv2.FILLED)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img,f'FPS: {int(fps)}',(40,70),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)
    cv2.imshow('Video', img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
