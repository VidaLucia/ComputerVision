import cv2
import mediapipe as mp
import time

class FaceMeshTracker:
    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):
        """
        Initializes the FaceMeshTracker with MediaPipe's Face Mesh solution.
        """
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )

        self.results = None  # Store results here

    def findFaceMesh(self, frame, draw=True):
        """
        Detects the face mesh and optionally draws the landmarks on the frame.
        """
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)

        if self.results.multi_face_landmarks and draw:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(
                    frame, faceLms,
                    self.mpFaceMesh.FACEMESH_TESSELATION,
                    self.drawSpec, self.drawSpec
                )
        return frame

    def findPosition(self, frame, draw=True):
        """
        Returns landmark coordinates for the first detected face.

        Returns:
            lmList (list): A list of [id, x, y] for each landmark.
        """
        lmList = []
        if self.results and self.results.multi_face_landmarks:
            faceLM = self.results.multi_face_landmarks[0]
            h, w, c = frame.shape
            for id, lm in enumerate(faceLM.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 1, (255, 0, 0), cv2.FILLED)
        return lmList
