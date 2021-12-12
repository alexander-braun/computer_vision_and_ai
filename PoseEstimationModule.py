import cv2 as cv
import mediapipe as mp
import time

class PoseTrackingModule():
  
  def __init__(self, mode = False, upperBodyOnly = False, smoothLandmarks = True, minDetectionConfidence = 0.5, minTrackingConfidence = 0.5):
    self.mode = mode
    self.upperBodyOnly = upperBodyOnly
    self.smoothLandmarks = smoothLandmarks
    self.minDetectionConfidence = minDetectionConfidence
    self.minTrackingConfidence = minTrackingConfidence

    self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
    self.mpPose = mp.solutions.mediapipe.python.solutions.pose
    self.pose = self.mpPose.Pose(
      static_image_mode=self.mode,
      smooth_landmarks=self.smoothLandmarks,
      min_detection_confidence=self.minDetectionConfidence,
      min_tracking_confidence=self.minTrackingConfidence
    )
    
    
  def findPose(self, img, draw = True):
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    self.results = self.pose.process(imgRGB)
    if draw and self.results.pose_landmarks:
      self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
    return img

  def findPosition(self, img, draw = True):
    lmList = []
    if self.results.pose_landmarks:
      for id, lm in enumerate(self.results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])
        if draw:
          cv.circle(img, (cx, cy), 10, (255, 0, 255), 1)
    return lmList


def main():
  cap = cv.VideoCapture(1)
  pTime = 0
  
  detector = PoseTrackingModule()

  while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    
    # Example
    for element in lmList:
      if (element[0] == 11):
        cv.putText(img, 'Left Shoulder', (int(element[1]), int(element[2])), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
      elif (element[0] == 10):
        cv.putText(img, 'Mouth Right', (int(element[1]), int(element[2])), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
      elif (element[0] == 12):
        cv.putText(img, 'Right Shoulder', (int(element[1]), int(element[2])), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
      elif (element[0] == 2 or element[0] == 5):
        cv.circle(img, (element[1], element[2]), 10, (0, 0, 255), cv.FILLED)
        
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)
    cv.imshow('Vid', img)
    cv.waitKey(1)

if __name__ == '__main__':
  main()