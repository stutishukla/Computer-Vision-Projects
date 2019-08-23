import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('/Users/stutishukla/Downloads/data/tsucuba_left.png',cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('/Users/stutishukla/Downloads/data/tsucuba_right.png',cv2.IMREAD_GRAYSCALE) 

'''Referred code from :https://docs.opencv.org/trunk/d2/d85/classcv_1_1StereoSGBM.html'''
stereo = cv2.StereoSGBM_create(numDisparities=32, blockSize=20)
disparity = stereo.compute(imgL,imgR)
cv2.imwrite('/Users/stutishukla/Downloads/task2_images/task2_disparity.jpg',disparity)

