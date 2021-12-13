import cv2 as cv
import mediapipe as mp
import time
from mediapipe.python.solutions.face_mesh import FaceMesh

class FaceMeshModule():
  def __init__(self, staticMode = False, maxFaces = 2, minDetectionConfidence = 0.5, minTrackingConfidence = 0.5):
    self.staticMode = staticMode
    self.maxFaces = maxFaces
    self.minDetectionConfidence = minDetectionConfidence
    self.minTrackingConfidence = minTrackingConfidence
    
    self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
    self.mpFacemesh = mp.solutions.mediapipe.python.solutions.face_mesh
    self.faceMesh = self.mpFacemesh.FaceMesh(
      max_num_faces = maxFaces, 
      static_image_mode = staticMode, 
      min_detection_confidence = minDetectionConfidence, 
      min_tracking_confidence = minTrackingConfidence
    )
    self.drawSpec = self.mpDraw.DrawingSpec((0, 255, 0), thickness=1, circle_radius=1)
    
  def findFaceMesh(self, img, draw = True):
    self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    self.results = self.faceMesh.process(self.imgRGB)
    
    faces = []
    if self.results.multi_face_landmarks:
      for landmark in self.results.multi_face_landmarks:
        if draw:
          self.mpDraw.draw_landmarks(img, landmark, self.mpFacemesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
        face = []
        for id, point in enumerate(landmark.landmark):
          h, w, c = img.shape
          x, y = int(point.x * w), int(point.y * h)
          face.append([x, y])
        faces.append(face)
    return img, faces

def main():
  cap = cv.VideoCapture(1)
  pTime = 0
  FaceMesh = FaceMeshModule()
  
  while True:
    success, img = cap.read()
    cTime = time.time()
    
    img, faces = FaceMesh.findFaceMesh(img)
    
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f'Fps: {int(fps)}', (50, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    
    cv.imshow('Image', img)
    cv.waitKey(1)

if __name__ == "__main__":
  main()