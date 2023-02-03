# import required libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt

# read two input images
imgL = cv2.imread('/Users/pmaksimkin/PycharmProjects/depthMapV2/test_photo_L.png',0)
imgR = cv2.imread('/Users/pmaksimkin/PycharmProjects/depthMapV2/test_photo_R.png',0)

# Initiate and StereoBM object
stereo = cv2.StereoBM_create(numDisparities=0, blockSize=31)

# compute the disparity map
disparity = stereo.compute(imgL,imgR)
disparity1 = stereo.compute(imgR,imgL)
plt.imshow(disparity,'gray')
plt.show()