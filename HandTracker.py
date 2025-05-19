import cv2
import mediapipe as mp

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        """
        Initializes the hand detector with MediaPipe Hands.

        Args:
            mode (bool): Whether to treat the input images as a batch of static images.
            maxHands (int): Maximum number of hands to detect.
            detectionCon (float): Minimum detection confidence threshold.
            trackCon (float): Minimum tracking confidence threshold.
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, frame, draw=True):
        """
        Detects hands in the given frame and optionally draws landmarks.

        Args:
            frame (ndarray): The current BGR frame from OpenCV.
            draw (bool): Whether to draw hand landmarks on the frame.

        Returns:
            frame (ndarray): The frame with drawn hand landmarks if draw is True.
        """
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        frame, handLandmarks, self.mpHands.HAND_CONNECTIONS
                    )
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        """
        Finds the positions of hand landmarks and optionally draws a circle on the first one.

        Args:
            frame (ndarray): The current frame image.
            handNo (int): The index of the hand to track (0 is the first).
            draw (bool): Whether to draw a circle on the first landmark.

        Returns:
            lmList (list): A list of [id, x, y] for each landmark detected.
        """
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw and id == 0:  # highlight the first landmark (usually wrist or base)
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
        return lmList
