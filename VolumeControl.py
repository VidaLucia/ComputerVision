import math
import time
import cv2
import numpy as np
import HandTracker as ht  # Ensure this is your custom hand tracking module

# Volume control modules
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize camera
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize hand detector
detector = ht.HandDetector(detectionCon=0.8)
pTime = 0

# Initialize system volume interface
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volMin, volMax = volume.GetVolumeRange()[:2]  # Usually (-65.25, 0.0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hand and get landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Thumb tip = id 4, Index finger tip = id 8
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw landmarks and connecting line
        cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

        # Calculate distance
        length = math.hypot(x2 - x1, y2 - y1)

        # Convert length to volume
        vol = np.interp(length, [30, 250], [volMin, volMax])
        volume.SetMasterVolumeLevel(vol, None)

        # Draw volume bar and percent
        volBar = np.interp(length, [30, 250], [400, 150])
        volPerc = np.interp(length, [30, 250], [0, 100])
        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 0), 2)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPerc)} %', (40, 430),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        # Exit if hand is wide open
        if length > 250:
            print("Exiting — hand opened wide.")
            break

        # Highlight when fingers are pinched
        if length < 40:
            cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

    # FPS counter
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 70),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    cv2.imshow('Volume Control', img)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
