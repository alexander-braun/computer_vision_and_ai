import cv2 as cv
import time
import HandTrackingModule
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

#### VOLUME PACKAGE
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
####

camW, camH = 640, 480

cap = cv.VideoCapture(1)

cap.set(cv.CAP_PROP_FRAME_WIDTH, camW)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, camH)

pTime = 0

detector = HandTrackingModule.HandDetector(detectionConfidence=0.75)

while True:
  success, img = cap.read()
  img = detector.findHands(img)
  landmarks = detector.findPosition(img)
  
  if len(landmarks):
    x1, y1 = landmarks[4][1], landmarks[4][2]
    x2, y2 = landmarks[8][1], landmarks[8][2]
    
    xZ, yZ = landmarks[5][1], landmarks[5][2]
    xY, yY = landmarks[0][1], landmarks[0][2]
    centerX, centerY = (x1 + x2) // 2, (y1 + y2) // 2
    
    cv.circle(img, (x1, y1), 15, (255, 255, 0), 2)
    cv.circle(img, (x2, y2), 15, (255, 255, 0), 2)
    cv.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv.circle(img, (centerX, centerY), 10, (0, 0, 255), 3)
    
    lengthLine = math.hypot(x2 - x1, y2 - y1)
    ratio = 100 / math.hypot(xY - xZ, yY - yZ) 
    
    if lengthLine * ratio < 15:
      cv.circle(img, (centerX, centerY), int(10 / ratio), (0, 255, 0), cv.FILLED)
    
    vol = np.interp(lengthLine * ratio, [15, 100], [-65.25, 0])
    volBar = np.interp(lengthLine * ratio, [15, 100], [400, 150])
    volume.SetMasterVolumeLevel(vol, None)
    
    cv.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv.FILLED)
    print(int(volBar))
  cTime = time.time()
  fps = 1 / (cTime - pTime)
  pTime = cTime
  cv.putText(img, f'Fps: {int(fps)}', (50, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
  
  cv.imshow('IMG', img)
  
  cv.waitKey(1)