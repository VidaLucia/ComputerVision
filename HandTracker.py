import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        """
        Initializes the hand detector with MediaPipe Hands.
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
        """
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        frame, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        """
        Finds the positions of hand landmarks.

        Args:
            frame (ndarray): The current frame image.
            handNo (int): Index of the hand to track (0 = first hand detected).
            draw (bool): Whether to draw a circle on each landmark.

        Returns:
            lmList (list): A list of [id, x, y] for each landmark detected.
        """
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            try:
                myHand = self.results.multi_hand_landmarks[handNo]
                h, w, c = frame.shape
                for id, lm in enumerate(myHand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            except IndexError:
                pass  # in case handNo >= number of detected hands
        return lmList
