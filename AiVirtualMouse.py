import HandTrackingModule as htm
import cv2 as cv
import numpy as np
import time
import pyautogui


wCam, hCam = 640, 480
frameR = 100
smoothening = 3

pLocationX, pLocationY = 0, 0
currLocationX, currLocationY = 0, 0

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = htm.HandDetector(maxHands=1)
screenW, screenH = pyautogui.size()

while True:
  success, img = cap.read()
  
  img = detector.findHands(img)
  lmList, bbox = detector.findPosition(img, draw=True)
  cv.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
  
  if len(lmList):
    x1, y1 = lmList[8][1:]
    x2, y2 = lmList[12][1:]
    
    fingers = detector.fingersUp()

    if fingers[1] == 1 and fingers[2] == 0:
      x3 = np.interp(x1, (frameR, wCam - frameR), (0, screenW))
      y3 = np.interp(y1, (frameR, hCam - frameR), (0, screenH))
      
      currLocationX = pLocationX + (x3 - pLocationX) / smoothening
      currLocationY = pLocationY + (y3 - pLocationY) / smoothening
      
      pyautogui.moveTo(screenW - currLocationX, currLocationY, 0)
      
      pLocationX, pLocationY = currLocationX, currLocationY
      
    if fingers[1] == 1 and fingers[2] == 1:
      length, img, lineInfo = detector.findDistance(8, 12, img, r = 5, t = 1)
      if length < 15:
        cv.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv.FILLED)
        pyautogui.click()
  cTime = time.time()
  fps = 1 / (cTime - pTime)
  pTime = cTime
  
  cv.putText(img, str(int(fps)), (10, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
  
  cv.imshow("Image", img)
  cv.waitKey(1)