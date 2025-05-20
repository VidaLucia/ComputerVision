import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self, mode=False, smooth=True, detectionMin=0.5, trackMin=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionMin = detectionMin
        self.trackMin = trackMin

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionMin,
            min_tracking_confidence=self.trackMin
        )

    def findPose(self, frame, draw=True):
        """
        Finds and draws the landmarks on a body and connects them.

        Args:
            frame (ndarray): The current frame image (BGR).
            draw (bool): Whether to draw the landmarks on the frame.

        Returns:
            frame (ndarray): The frame with drawn landmarks if draw is True.
        """
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return frame

    def findPosition(self, frame,frameNo = 0, draw=True):
        """
        Finds all pose landmark positions and optionally draws the first landmark.

        Args:
            frame (ndarray): The current frame image.
            draw (bool): Whether to draw a circle on the first landmark.

        Returns:
            lmList (list): A list of [id, x, y] for each landmark.
        """
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList
