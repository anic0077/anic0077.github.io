import cv2
import time
import Pose_Module_2 as pm


cap = cv2.cv2.VideoCapture('C:/python_projects/PoseVideos/1.mp4')
pTime = 0
detector = pm.poseDetector()
runtime = 0
while runtime < 200:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)
    if lmList:
        print(lmList[14])
        cv2.circle(img, (lmList[1][1], lmList[1][2]),15,(0,0,255),cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    runtime += 1

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)