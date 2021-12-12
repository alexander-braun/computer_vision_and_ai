import cv2 as cv
import mediapipe as mp
import time

mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
mpPose = mp.solutions.mediapipe.python.solutions.pose
pose = mpPose.Pose()

#cap = cv.VideoCapture('./resources/pexels-artem-podrez-6003986.mp4')
cap = cv.VideoCapture(1)

pTime = 0

while True:
  # Read frame from video
  success, img = cap.read()
  imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  results = pose.process(imgRGB)
  
  if results.pose_landmarks:
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
      h, w, c = img.shape
      cx, cy = int(lm.x * w), int(lm.y * h)
      cv.circle(img, (cx, cy), 10, (255, 0, 255), 1)
      
  
  
  cTime = time.time()
  fps = 1 / (cTime - pTime)
  pTime = cTime
  
  # put fps in vid
  cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)
  cv.imshow('Vid', img)
  cv.waitKey(1)