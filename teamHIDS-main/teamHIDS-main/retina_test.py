import cv2 
from retinaface import RetinaFace

data2 = RetinaFace.detect_faces("face_crowd2.png")
print("completed")