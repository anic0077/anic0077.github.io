import cv2
import mediapipe as mp
import time
import os

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'PoseVideos/1.mp4')
filename = filename.replace("\\", "/")
cap = cv2.VideoCapture(filename)
pTime = 0

playing = True
interupt_time = 0

while interupt_time < 1000:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx,cy), 10, (255,0,0), cv2.FILLED)

            interupt_time += 1

    

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow("Image", img)


    cv2.waitKey(1)