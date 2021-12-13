import cv2 as cv
import mediapipe as mp
import time


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
    landmarkList = []
    if self.results.multi_hand_landmarks:
      hand = self.results.multi_hand_landmarks[handNumber]
      for id, landmark in enumerate(hand.landmark):
        h, w, c = img.shape
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        landmarkList.append([id, cx, cy])
        if draw:
          cv.circle(img, (cx, cy), 3, (255, 0, 0), cv.FILLED)
    return landmarkList

def main():
  previousTime = 0
  currentTime = 0
  cap = cv.VideoCapture(1)
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