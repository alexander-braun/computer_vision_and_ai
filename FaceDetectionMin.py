import cv2 as cv
import mediapipe as mp
import time


# Get Webcam
cap = cv.VideoCapture(1)
pTime = 0

mpFaceDetection = mp.solutions.mediapipe.python.solutions.face_detection
mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.55)

while True:
  # read the capture from webcam
  success, img = cap.read()
  # convert captured image from BGR to RGB
  imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  results = faceDetection.process(imgRGB)
  if results.detections:
    for id, detection in enumerate(results.detections):
      bboxC = detection.location_data.relative_bounding_box
      imageHeight, imageWidth, imageChannel = img.shape
      bbox = int(bboxC.xmin * imageWidth), int(bboxC.ymin * imageHeight), int(bboxC.width * imageWidth), int(bboxC.height * imageHeight)
      cv.rectangle(img, bbox, color=(255, 0, 255), thickness=2)
      cv.putText(img, str(int(detection.score[0] * 100)) + '%', (int(bboxC.xmin * imageWidth), int(bboxC.ymin * imageHeight) + int(bboxC.height * imageHeight) + 25), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
  cTime = time.time()
  fps = 1 / (cTime - pTime)
  pTime = cTime
  cv.putText(img, str(int(fps)), (50, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)


  cv.imshow('Image', img)
  cv.waitKey(1)
