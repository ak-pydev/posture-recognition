import cv2
import mediapipe as mp
import numpy as np
import time

class PoseDetector:
    def __init__(self, mode=False, model_complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 detection_confidence=0.5, tracking_confidence=0.5):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.lm_list = []

    def find_pose(self, img, draw=True):
        if img is None:
            raise ValueError("Input image is None")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks,
                                        mp.solutions.pose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        self.lm_list = []
        if self.results.pose_landmarks:
            h, w, _ = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append((id, cx, cy))
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lm_list

    def find_angle(self, img, p1, p2, p3, draw=True):
        if not self.lm_list:
            raise ValueError("Landmark list is empty")
        if any(p >= len(self.lm_list) for p in (p1, p2, p3)):
            raise IndexError("Landmark index out of range")
        _, x1, y1 = self.lm_list[p1]
        _, x2, y2 = self.lm_list[p2]
        _, x3, y3 = self.lm_list[p3]

        v1 = np.array([x1 - x2, y1 - y2])
        v2 = np.array([x3 - x2, y3 - y2])
        angle = np.degrees(np.arccos(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            for x, y in [(x1, y1), (x2, y2), (x3, y3)]:
                cv2.circle(img, (x, y), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x, y), 15, (0, 0, 255), 2)
            cv2.putText(img, f'{int(angle)}°', (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle
def main():
    cap = cv2.VideoCapture('PoseVideos/1.mp4')
    if not cap.isOpened():
        raise IOError("Cannot open video file")
    p_time = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        if not success:
            print("Ignoring empty frame.")
            break
        img = detector.find_pose(img)
        lm_list = detector.find_position(img, draw=False)
        if lm_list:
            cv2.circle(img, (lm_list[14][1], lm_list[14][2]), 15, (0, 0, 255), cv2.FILLED)
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
