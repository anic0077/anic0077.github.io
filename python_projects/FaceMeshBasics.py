from typing_extensions import runtime
import cv2
import mediapipe as mp
import time
import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'PoseVideos/2.mp4')
filename = filename.replace("\\", "/")



#cv2.VideoCapture("C:/python_projects/PoseVideos/2.mp4")
cap = cv2.VideoCapture(filename)
pTime = 0
runtime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 2)
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=2)

while runtime < 180:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    print(results.multi_face_landmarks)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION,drawSpec,drawSpec)
            for lm in enumerate(faceLms.landmark):
                print(lm)
                ih,iw,ic = img.shape
                #x,y = int(lm.x*iw), int(lm.y*ih)
                print(ih,iw)




    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    runtime += 1
    cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
