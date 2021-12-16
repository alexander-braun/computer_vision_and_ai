import cv2 as cv
import numpy as np
import time
import os
import HandTrackingModule as htm

overlay = []
for i in range(4):
  image = cv.imread(f'./resources/paint/{i + 1}.jpg')
  overlay.append(image)
  
header = overlay[0]
detector = htm.HandDetector(detectionConfidence=0.85)

colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0)]
currentColor = colors[0]
brushThickness = 20

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 960)

xp, yp = 0, 0
imgCanvas = np.zeros((960, 1280, 3), np.uint8)

while True:
  success, img = cap.read()  
  img = cv.flip(img, 1)
  
  img = detector.findHands(img)
  positions = detector.findPosition(img, draw=False)
  
  indexUp = False
  ringUp = False
  
  if len(positions):
    indexX, indexY = positions[8][1:]
    middleX, middleY = positions[12][1:]
    
    if positions[6][2] > indexY:
      indexUp = True
    if positions[10][2] > middleY:
      ringUp = True
      
    if indexUp and ringUp:
      xp, yp = 0, 0
      cv.rectangle(img, (indexX - 15, indexY - 25), (middleX + 15, middleY + 25), (255, 255, 0), cv.FILLED)
      if indexY < 208:
        if 0 < indexX < 273:
          header = overlay[0]
          currentColor = colors[0]
        elif indexX < 545:
          header = overlay[1]
          currentColor = colors[1]
        elif indexX < 818:
          header = overlay[2]
          currentColor = colors[2]
        else:
          header = overlay[3]
          currentColor = colors[3]
    elif indexUp:
      cv.circle(img, (indexX, indexY), 15, currentColor, cv.FILLED)
      
      if xp == 0 and yp == 0:
        xp, yp = indexX, indexY
        
      if currentColor == (0, 0, 0):
        cv.line(img, (xp, yp), (indexX, indexY), currentColor, brushThickness + 20)
        cv.line(imgCanvas, (xp, yp), (indexX, indexY), currentColor, brushThickness + 20)

      cv.line(imgCanvas, (xp, yp), (indexX, indexY), currentColor, brushThickness)
      xp, yp = indexX, indexY
  
  # overlay canvas over img
  imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
  _, imgInv = cv.threshold(imgGray, 10, 255, cv.THRESH_BINARY_INV)
  imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
  img = cv.bitwise_and(img, imgInv)
  img = cv.bitwise_or(img, imgCanvas)

  img[0:208, 0:1280] = header
  cv.imshow('IMG', img)
  cv.waitKey(1)