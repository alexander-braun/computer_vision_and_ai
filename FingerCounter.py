import cv2 as cv
import mediapipe as mp
import time
import HandTrackingModule

list = []
for i in range(6):
  image = cv.imread(f'./resources/fingers/{i}.jpg')
  list.append(image)

# key 0 because different webcam
cap = cv.VideoCapture(0)

pTime = 0

detector = HandTrackingModule.HandDetector(detectionConfidence=0.75)

while True:
  success, img = cap.read()
  
  img = detector.findHands(img)
  points = detector.findPosition(img, draw=False)
  
  connections = [[6, 8], [10, 12], [14, 16], [18, 20]]
  
  fingersUp = 0
  if len(points):
    for connection in connections:
      if points[connection[0]][2] > points[connection[1]][2]:
        fingersUp += 1

    if points[6][1] < points[4][1]:
      fingersUp += 1
      
  h, w, c = list[0].shape
  img[0:h, 0:w] = list[fingersUp]
  
  cTime = time.time()
  fps = 1 / (cTime - pTime)
  pTime = cTime
  cv.putText(img, f'FPS: {str(int(fps))}', (10, 400), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
  
  cv.imshow('Image', img)
  cv.waitKey(1)