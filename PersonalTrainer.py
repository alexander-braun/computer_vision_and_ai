import cv2 as cv
import mediapipe as mp
import time
import PoseEstimationModule as pm


cap = cv.VideoCapture('./resources/pexels-kampus-production-6893316.mp4')
detector = pm.PoseTrackingModule(minDetectionConfidence=0.9)

left = 0
right = 0
leftOnTop = False
rightOnTop = False
pTime = 0

while True:
  success, img = cap.read()
  img = cv.resize(img, (1280, 700))
  img = detector.findPose(img)
  position = detector.findPosition(img)
  
  
  if len(position):
    if position[19][2] < position[23][2] and leftOnTop == False:
      leftOnTop = True
      left += 1
    elif position[19][2] > position[23][2]:
      leftOnTop = False
      
    if position[20][2] < position[24][2] and rightOnTop == False:
      rightOnTop = True
      right += 1
    elif position[20][2] > position[24][2]:
      rightOnTop = False
      
  cTime = time.time()
  fps = 1 / (cTime - pTime)
  pTime = cTime
  
  cv.putText(img, f'left arm lifts: {left}, right arm lifts: {right}', (30, 30), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
      
  print(right, left)
  
  cv.imshow('Video', img)
  cv.waitKey(1)