

import os
import time
from datetime import datetime
import cv2
import numpy as np


# Camera settimgs
cam_width = 640  # Cam sensor width settings
cam_height = 480  # Cam sensor height settings


# Initialize the camera
# TODO: Use more stable identifiers

right = cv2.VideoCapture(2)
left = cv2.VideoCapture(1)

left.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
left.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
right.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
right.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

LEFT_PATH = "/Users/pmaksimkin/PycharmProjects/depthMapV2/pictures/left/{:06d}.png"
RIGHT_PATH = "/Users/pmaksimkin/PycharmProjects/depthMapV2/pictures/right/{:06d}.png"

# Lets start taking photos!
counter = 0
t2 = datetime.now()
print("Starting photo sequence")
frame_counter = 0

while True:
    if not (left.grab() and right.grab()):
        print("No more frames")
        break

    _, leftFrame = left.retrieve()
    _, rightFrame = right.retrieve()

    if cv2.waitKey(1) & 0xFF == ord('c'):
        frame_counter += 1
        cv2.imwrite(LEFT_PATH.format(frame_counter), leftFrame)
        cv2.imwrite(RIGHT_PATH.format(frame_counter), rightFrame)
        print("wrote file")

    cv2.imshow('left', leftFrame)
    cv2.imshow('right', rightFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


left.release()
right.release()
cv2.destroyAllWindows()