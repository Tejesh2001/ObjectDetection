import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import YOLO
import numpy as np
from cvlib.object_detection import draw_bbox

# This is for detecting objects in the COCO dataset
im = cv2.imread('dog.jpg')
# confidence is accuracy percentage, bbox give the coordinates (top left, bottom right)
# in the form of list and label is the object name
bbox, label, conf = cv.detect_common_objects(im)
print(bbox)
# Draws a rectangular box according to the bbox coordinates
output_image = draw_bbox(im, bbox, label, conf)
cv2.imwrite('objects.png', output_image)
