from typing_extensions import runtime
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import HandTrackingModule as htm

brushThickness = 15
rubberThickness = 50

dirname = os.path.dirname(__file__)
folderPath = os.path.join(dirname, 'Header')
folderPath = folderPath.replace("\\", "/")

#folderPath = "C:/python_projects/Header"

myList = os.listdir(folderPath)
print(myList)

overlayList = []

for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

print(len(overlayList))

header = overlayList[0]
drawColour = (0,0,255)

cap = cv2.VideoCapture(0)

cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon = 0.85)
xp,yp = 0,0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

runTime = 0
open = True

while open ==True:
    #import image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    runTime += 1

    #find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)

    if lmList:
        #print(lmList)
        #tip of index and middle fingers
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]

    #check which fingers are up

        fingers = detector.fingersUp()
        print(fingers)

    #if selection mode - 2 fingers are up
        if fingers[1] and fingers[2]:
            xp,yp = 0,0
            print("selection Mode")
            
            if y1 < 125:
                if 0<x1<150:
                    open = False
                elif 250<x1<450:
                    header = overlayList[0]
                    drawColour = (0,0,255)
                elif 550<x1<750:
                    header = overlayList[1]
                    drawColour = (0,255,255)
                elif 800<x1<950:
                    header = overlayList[2]
                    drawColour = (0,255,0)
                elif 1050<x1<1200:
                    header = overlayList[3]
                    drawColour = (0,0,0)
            cv2.rectangle(img, (x1,y1-25), (x2, y2+25), (drawColour), cv2.FILLED)

    #if Drawing mpde - index finger is up
        if fingers[1] and fingers[2]==False:
            print("drawing Mode")
            cv2.circle(img, (x1,y1), 15, (drawColour), cv2.FILLED)
            if xp==0 and yp==0:
                xp,yp = x1,y1
            
            if drawColour == (0,0,0):

                cv2.line(img, (xp,yp),(x1,y1), drawColour,rubberThickness)
                cv2.line(imgCanvas, (xp,yp),(x1,y1), drawColour,rubberThickness)
            else:

                cv2.line(img, (xp,yp),(x1,y1), drawColour,brushThickness)
                cv2.line(imgCanvas, (xp,yp),(x1,y1), drawColour,brushThickness)

            xp,yp = x1,y1
            


    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    #setting header image
    img[0:125,0:1280] = header
    #img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image",img)
    cv2.imshow("Canvas",imgCanvas)
    cv2.waitKey(1)