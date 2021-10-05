from typing_extensions import runtime
import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self,staticMode=False,maxFaces=2,micDetectionCon=0.5,minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.micDetectionCon = micDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,self.micDetectionCon,self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1,circle_radius=2)

    def findFaceMesh(self,img,draw=True):


        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(self.imgRGB)
        print(results.multi_face_landmarks)
        if results.multi_face_landmarks:
            if draw:
                for faceLms in results.multi_face_landmarks:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,self.drawSpec,self.drawSpec)
                        
        return img




    

def main():
    #cap = cv2.VideoCapture("C:/python_projects/PoseVideos/2.mp4")
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    runtime = 0
    while runtime < 180:
        success, img = cap.read()
        img = detector.findFaceMesh(img, True)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime=cTime
        runtime += 1
        cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)





if __name__ == "__main__":
    main()