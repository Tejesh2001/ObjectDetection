import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import YOLO
import numpy as np
from cvlib.object_detection import draw_bbox
# Insert your image file name in the argument
img = cv2.imread('INSERT_YOUT_IMAGE.jpg')
face, conf = cv.detect_face(img)

padding = 20
# Without padding accuracy drops 2 percent

# loop through detected faces
for f in face:
    startX, startY = max(0, f[0] - padding), max(0, f[1] - padding)
    endX, endY = min(img.shape[1] - 1, f[2] + padding), min(img.shape[0] - 1, f[3] + padding)

    # draw rectangle over face
    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

    face_crop = np.copy(img[startY:endY, startX:endX])

    # apply gender detection
    (label, confidence) = cv.detect_gender(face_crop)
    idx = np.argmax(confidence)
    label = label[idx]

    label = "{}: {:.2f}%".format(label, confidence[idx] * 100)

    Y = startY - 10 if startY - 10 > 10 else startY + 10

    cv2.putText(img, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

# save output
cv2.imwrite("gender_detection.jpg", img)
