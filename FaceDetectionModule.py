import cv2 as cv
import mediapipe as mp
import time


class FaceDetectionModule():
  
  def __init__(self, minDetectionConfidence = 0.5):
    self.minDetectionConfidence = minDetectionConfidence
    
    self.mpFaceDetection = mp.solutions.mediapipe.python.solutions.face_detection
    self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
    self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionConfidence)
    
  def findFaces(self, img, draw = True):
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    self.results = self.faceDetection.process(imgRGB)
    bboxes = []
    
    if self.results.detections:
      for id, detection in enumerate(self.results.detections):
        bboxC = detection.location_data.relative_bounding_box
        imageHeight, imageWidth, imageChannel = img.shape
        bbox = int(bboxC.xmin * imageWidth), int(bboxC.ymin * imageHeight), int(bboxC.width * imageWidth), int(bboxC.height * imageHeight)
        bboxes.append([bbox, detection.score])
        
        if draw:
          img = self.fancyDraw(img, bbox)
          cv.putText(img, str(int(detection.score[0] * 100)) + '%', (int(bboxC.xmin * imageWidth), int(bboxC.ymin * imageHeight) + int(bboxC.height * imageHeight) + 35), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
    return img, bboxes
  
  def fancyDraw(self, img, bbox, l = 20, t = 7):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    
    cv.rectangle(img, bbox, color=(255, 0, 255), thickness=1)
    
    cv.line(img, (x, y), (x + l, y), (255, 0, 255), t)
    cv.line(img, (x, y), (x, y + l), (255, 0, 255), t)
    
    cv.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
    cv.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
    
    cv.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
    cv.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
    
    cv.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
    cv.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
    
    return img
  
def main():
  cap = cv.VideoCapture(1)
  pTime = 0
  FaceDetection = FaceDetectionModule(0.75)
  
  while True:
    success, img = cap.read()
    img, bboxes = FaceDetection.findFaces(img)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, str(int(fps)), (50, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
    cv.imshow('Image', img)
    cv.waitKey(1)
  
if __name__ == '__main__':
  main()