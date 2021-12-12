import cv2 as cv
import mediapipe as mp
import time

from mediapipe.python.solutions.face_mesh import FaceMesh


cap = cv.VideoCapture(1)
pTime = 0

mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
mpFacemesh = mp.solutions.mediapipe.python.solutions.face_mesh
faceMesh = mpFacemesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec((0, 255, 0), thickness=1, circle_radius=1)

while True:
  success, img = cap.read()

  cTime = time.time()
  fps = 1 / (cTime - pTime)
  pTime = cTime
  cv.putText(img, f'Fps: {int(fps)}', (50, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
  
  
  imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  results = faceMesh.process(imgRGB)
  
  
  if results.multi_face_landmarks:
    for landmark in results.multi_face_landmarks:
      mpDraw.draw_landmarks(img, landmark, mpFacemesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
      
      for id, point in enumerate(landmark.landmark):
        h, w, c = img.shape
        x, y = int(point.x * w), int(point.y * h)
        
  

  cv.imshow('Image', img)
  cv.waitKey(1)
