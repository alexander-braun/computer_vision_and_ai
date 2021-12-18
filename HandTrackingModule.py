import cv2 as cv
import mediapipe as mp
import time
import math


class HandDetector():
  def __init__(self, mode = False, maxHands = 2, detectionConfidence = 0.5, trackConfidence = 0.5):
    self.mode = mode
    self.maxHands = maxHands
    self.detectionConfidence = detectionConfidence
    self.trackConfidence = trackConfidence
    
    self.mpHands = mp.solutions.mediapipe.python.solutions.hands
    self.hands = self.mpHands.Hands(
      static_image_mode=self.mode, 
      max_num_hands=self.maxHands, 
      min_detection_confidence=self.detectionConfidence, 
      min_tracking_confidence=self.trackConfidence)
    self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils

  def findHands(self, img, draw = True):
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    self.results = self.hands.process(imgRGB)
    if self.results.multi_hand_landmarks:
      for handLandmark in self.results.multi_hand_landmarks:
        if draw:
          self.mpDraw.draw_landmarks(img, handLandmark, self.mpHands.HAND_CONNECTIONS)
    return img
  
  def findPosition(self, img, handNumber = 0, draw = True):
    xList = []
    yList = []  
    bbox = []
    self.landmarkList = []
    if self.results.multi_hand_landmarks:
      hand = self.results.multi_hand_landmarks[handNumber]
      for id, landmark in enumerate(hand.landmark):
        h, w, c = img.shape
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        self.landmarkList.append([id, cx, cy])
        xList.append(cx)
        yList.append(cy)
        if draw:
          cv.circle(img, (cx, cy), 3, (255, 0, 0), cv.FILLED)
          
      xmin, xmax = min(xList), max(xList)
      ymin, ymax = min(yList), max(yList)
      bbox = xmin, ymin, xmax, ymax    
      
      if draw:
        cv.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

    return self.landmarkList, bbox
  
  def fingersUp(self):
    fingers = []
    if self.landmarkList[4][1] > self.landmarkList[5][1]:
      fingers.append(1)
    else:
      fingers.append(0)
      
    if self.landmarkList[8][2] < self.landmarkList[6][2]:
      fingers.append(1)
    else:
      fingers.append(0)
      
    if self.landmarkList[12][2] < self.landmarkList[10][2]:
      fingers.append(1)
    else:
      fingers.append(0)
      
    if self.landmarkList[16][2] < self.landmarkList[14][2]:
      fingers.append(1)
    else:
      fingers.append(0)  
    
          
    if self.landmarkList[20][2] < self.landmarkList[18][2]:
      fingers.append(1)
    else:
      fingers.append(0)  
      
    return fingers
  
  def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
    x1, y1 = self.landmarkList[p1][1:]
    x2, y2 = self.landmarkList[p2][1:]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
    if draw:
      cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
      cv.circle(img, (x1, y1), r, (255, 0, 255), cv.FILLED)
      cv.circle(img, (x2, y2), r, (255, 0, 255), cv.FILLED)
      cv.circle(img, (cx, cy), r, (255, 0, 255), cv.FILLED)
      
    length = math.hypot(x2 - x1, y2 - y1)
    
    return length, img, [x1, y1, x2, y2, cx, cy]
      
def main():
  previousTime = 0
  currentTime = 0
  cap = cv.VideoCapture(0)
  detector = HandDetector()
  
  while True:
    # read the capture from webcam
    success, img = cap.read()
    img = detector.findHands(img=img)
    positionList = detector.findPosition(img)
    print(positionList)
    
    if len(positionList) != 0:
      print(positionList[4])
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    
    cv.putText(img, 'Framerate:' + str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
    
    cv.imshow('Image', img)
    cv.waitKey(1)

if __name__ == '__main__':
  main()