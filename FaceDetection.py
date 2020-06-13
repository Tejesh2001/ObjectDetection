import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import YOLO

# This is for detecting faces
image = cv2.imread('tejesh.jpg')
faces, conf = cv.detect_face(image)
for face in faces:
    (startX,startY) = face[0],face[1]
    (endX,endY) = face[2],face[3]
    # draw rectangle over face
    cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)
# display output
cv2.imwrite('face detection.jpg', image)
