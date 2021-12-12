import cv2 as cv
import mediapipe as mp
import time

# Get Webcam
cap = cv.VideoCapture(1)

# Get hands module
mpHands = mp.solutions.mediapipe.python.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils

previousTime = 0
currentTime = 0


while True:
  # read the capture from webcam
  success, img = cap.read()
  # convert captured image from BGR to RGB
  imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  results = hands.process(imgRGB)
  # check for multiple hands and draw on original image
  if results.multi_hand_landmarks:
    for handLandmark in results.multi_hand_landmarks:
      for id, landmark in enumerate(handLandmark.landmark):
        h, w, c = img.shape
        cx, cy = int(landmark.x * w), int(landmark.y * h)
      mpDraw.draw_landmarks(img, handLandmark, mpHands.HAND_CONNECTIONS)
  
  currentTime = time.time()
  fps = 1 / (currentTime - previousTime)
  previousTime = currentTime
  
  cv.putText(img, 'Framerate:' + str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
  
  cv.imshow('Image', img)
  cv.waitKey(1)
