import cv2
import mediapipe as mp
import time
import HandTracker as ht
import PoseTracker as pt
import FaceTracker as ft
import FaceMeshTracker as fmt
previousTime = 0
cap = cv2.VideoCapture(0)

handDetector = ht.HandDetector()
poseDetector = pt.PoseDetector()
faceDetector = ft.FaceDetector()
faceMeshDetector = fmt.FaceMeshTracker()
option = "p"
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Uncomment any one detector below to test:
    if option == "h":
        frame = handDetector.findHands(frame)
        lmList = handDetector.findPosition(frame)
    elif option == "p":
        frame = poseDetector.findPose(frame)
        lmList = poseDetector.findPosition(frame)
    elif option == "f":
        frame, bboxes = faceDetector.findFaces(frame)
    elif option == "fm":
        frame = faceMeshDetector.findFaceMesh(frame)
        lmList = faceMeshDetector.findPosition(frame)
    else:
        break

    # Calculate and display FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
