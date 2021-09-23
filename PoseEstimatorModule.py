import cv2
import mediapipe as mp
import time
import math


class PoseEstimator():
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False,
                 smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpose = mp.solutions.pose
        self.pose = self.mpose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks,
                                    self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence,
                                    self.min_tracking_confidence)
        self.mdraw = mp.solutions.drawing_utils

    def drawPos(self, img, draw=True):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(rgb_img)

        if self.result.pose_landmarks:
            if draw:
                self.mdraw.draw_landmarks(img, self.result.pose_landmarks, self.mpose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=False):
        self.lmlist = []
        if self.result.pose_landmarks:
            for id, lm in enumerate(self.result.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), 2, cv2.FILLED)
        return self.lmlist

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        x3, y3 = self.lmlist[p3][1:]

        angle = math.degrees(math.atan2(y1 - y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2))
        #angle=360-angle
        if angle < 0:
            angle+=360
        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,255,255),2,cv2.LINE_4)
            cv2.line(img,(x2,y2),(x3,y3),(255,255,255),2,cv2.LINE_4)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2, )
            #cv2.putText(img, f'{(x1,y1)}', (x1 - 50, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #cv2.putText(img, f'{(x2,y2)}', (x2 - 80, y2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #cv2.putText(img, f'{(x3,y3)}', (x3 - 50, y3 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2, )
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2, )
            #cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return angle


def main():
    ptime = 0
    cam = cv2.VideoCapture(0)
    wcam, hcam = 1280, 720
    cam.set(3, wcam)
    cam.set(4, hcam)
    estimator = PoseEstimator()
    while True:
        img = cam.read()[1]
        img = estimator.drawPos(img)
        lm_list = estimator.findPosition(img)
        if len(lm_list) != 0:
            print(lm_list[5])
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
