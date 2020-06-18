import argparse
from cvlib.object_detection import YOLO
import sys
import cv2

# usage: python3 yolo_custom_weights_inference.py <yolov3.weights> <yolov3.config> <labels.names> <input_image

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image_file', required=True,
                help = 'input image absolute path')
ap.add_argument('-c', '--config_file', required=True,
                help = 'yolo config file absolute path')
ap.add_argument('-w', '--weights', required=True,
                help = 'yolo pre-trained weights absolute path')
ap.add_argument('-cl', '--labels', required=True,
                help = 'labels file (.names)')
args = ap.parse_args()
# read input image
image = cv2.imread(args.image_file)
yolo = YOLO(args.weights, args.config_file, args.labels)

# object detection
bbox, label, conf = yolo.detect_objects(image)

print(bbox, label, conf)

# bounding box over detected objects
yolo.draw_bbox(image, bbox, label, conf, write_conf=True)

# save output
cv2.imwrite("custom_object_detection.jpg", image)

