import cv2
import mediapipe as mp
import time

ptime = 0

cam = cv2.VideoCapture(0)
    
mpose = mp.solutions.pose  # Creating Instance for Pose
pose = mpose.Pose()
mdraw = mp.solutions.drawing_utils

while True:
    img = cam.read()[1]
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_img)

    if result.pose_landmarks:
        mdraw.draw_landmarks(img, result.pose_landmarks, mpose.POSE_CONNECTIONS)
        for id,lm in enumerate(result.pose_landmarks.landmark):
            h,w,c=img.shape
            cx,cy=int(lm.x*w),int(lm.y*h)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break
