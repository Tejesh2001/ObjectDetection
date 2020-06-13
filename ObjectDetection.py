import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import YOLO
import numpy as np
from cvlib.object_detection import draw_bbox



""""
yolo = YOLO(weights, config, labels)
bbox, label, conf = yolo.detect_objects(img)
output_image = yolo.draw_bbox(img, bbox, label, conf)
"""


# This is for detecting objects in the COCO dataset
im = cv2.imread('dog.jpg')
bbox, label, conf = cv.detect_common_objects(im)
print(bbox)
output_image = draw_bbox(im, bbox, label, conf)
cv2.imwrite('objects.png', output_image)
"""""
# This is for detecting faces
image = cv2.imread('face.jpg')
faces, conf = cv.detect_face(image)
for face in faces:
    (startX,startY) = face[0],face[1]
    (endX,endY) = face[2],face[3]
    # draw rectangle over face
    cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)
# display output
cv2.imshow('face', image)
cv2.imwrite('face detection.jpg', image)

img = cv2.imread('face.jpg')
faces, conf = cv.detect_gender(image)
face, conf = cv.detect_face(img)

padding = 20

# loop through detected faces
for f in face:
    (startX, startY) = max(0, f[0] - padding), max(0, f[1] - padding)
    (endX, endY) = min(img.shape[1] - 1, f[2] + padding), min(img.shape[0] - 1, f[3] + padding)

    # draw rectangle over face
    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

    face_crop = np.copy(img[startY:endY, startX:endX])

    # apply gender detection
    (label, confidence) = cv.detect_gender(face_crop)

    print(confidence)
    print(label)

    idx = np.argmax(confidence)
    label = label[idx]

    label = "{}: {:.2f}%".format(label, confidence[idx] * 100)

    Y = startY - 10 if startY - 10 > 10 else startY + 10

    cv2.putText(img, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

# display output
# press any key to close window
cv2.imshow("gender detection", img)
cv2.waitKey()

# save output
cv2.imwrite("gender_detection.jpg", img)

# release resources
cv2.destroyAllWindows()
"""""