import cv2
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, detectionCon=0.5):
        """
        Initializes the face detector using MediaPipe's Face Detection solution.
        Args:
            detectionCon (float): Minimum detection confidence threshold.
        """
        self.detectionCon = detectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(model_selection=0,
                                                                 min_detection_confidence=self.detectionCon)
        self.results = None

    def findFaces(self, frame, draw=True):
        """
        Detects faces and optionally draws bounding boxes with confidence score.

        Args:
            frame (ndarray): The BGR image frame from OpenCV.
            draw (bool): Whether to draw the detection output.

        Returns:
            frame (ndarray): Frame with drawings (if enabled).
            bboxes (list): List of [id, (x, y, w, h), score] for each detected face.
        """
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxes = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih)
                )
                bboxes.append([id, bbox, detection.score])

                if draw:
                    self.fancyDraw(frame, bbox)
                    cv2.putText(frame, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 0, 255), 2)

        return frame, bboxes

    def fancyDraw(self, frame, bbox, l=30, t=5, rt=1):
        """
        Draws a fancy styled bounding box around the face.

        Args:
            frame (ndarray): The frame to draw on.
            bbox (tuple): Bounding box (x, y, w, h).
            l (int): Line length.
            t (int): Thickness.
            rt (int): Rectangle thickness.
        """
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        # Draw corners
        # Top Left
        cv2.line(frame, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(frame, (x, y), (x, y + l), (255, 0, 255), t)
        # Top Right
        cv2.line(frame, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(frame, (x1, y), (x1, y + l), (255, 0, 255), t)
        # Bottom Left
        cv2.line(frame, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(frame, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right
        cv2.line(frame, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(frame, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        # Full rectangle
        cv2.rectangle(frame, bbox, (255, 0, 255), rt)

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    previousTime = 0

    while True:
        success, frame = cap.read()
        frame, bboxes = detector.findFaces(frame)

        # FPS Calculation
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        # Display FPS
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show output
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
