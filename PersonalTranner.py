import cv2
import PoseEstimatorModule
import time
import numpy as np

cam = cv2.VideoCapture(0)
ptime = 0
wcam, hcam = 1280,720
cam.set(3, wcam)
cam.set(4, hcam)
count, dir = 0, 0
detector = PoseEstimatorModule.PoseEstimator()
while True:
    img = cam.read()[1]
    # print(img.shape)
    # img=cv2.flip(img,1)
    img = detector.drawPos(img, False)
    lmlist = detector.findPosition(img)
    color = (0, 255, 0)
    if len(lmlist) != 0:
        # for i in lmlist:
        #    cv2.putText(img, f'{i[0]}', (i[1] - 50, i[2] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # detector.findAngle(img, 12, 14, 16)
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (210, 310), (0, 100))
        rec_per = np.interp(angle, (210, 310), (350, 80))
        if per == 100:
            color = (255, 0, 255)
            if dir == 1:
                count += 0.5
                dir = 0
        if per == 0:
            if dir == 0:
                count += 0.5
                dir = 1
        cv2.rectangle(img, (10, 430), (180, 590), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), ((50, 550) if count < 10 else (5, 560)), cv2.FONT_HERSHEY_SIMPLEX,
                    (4 if count < 100 else 3), (255, 0, 0), 5)
        cv2.rectangle(img, (50, 80), (75, 350), color, 1)
        cv2.rectangle(img, (50, int(rec_per)), (75, 350), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (50, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) == ord('q'):
        break
