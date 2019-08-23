
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:22:50 2018

@author: stutishukla
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
'''
Reference for this codes:https://docs.opencv.org/3.1.0/da/de9/tutorial_py_epipolar_geometry.html
'''
img1 = cv2.imread('/Users/stutishukla/Downloads/data/tsucuba_left.png',cv2.IMREAD_GRAYSCALE)          # queryImage
img2 = cv2.imread('/Users/stutishukla/Downloads/data/tsucuba_right.png',cv2.IMREAD_GRAYSCALE)          # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

bf = cv2.BFMatcher()
# Feature Matching using BFMatcher with default params
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good_points = []
for m,n in matches:
   if m.distance < 0.75*n.distance:  #descriptors distances in two images
        good_points.append(m)


#Homography
if len(good_points) > 10:  #if there are atleast 10 points then only I ll be finding homography, otherwise not.
    query_pts = np.float32([kp1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2) #extracting the position of the  good points in query image 
    train_pts = np.float32([kp2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)#extracting the position of the  good points in train image 
F, mask = cv2.findFundamentalMat(query_pts,train_pts,cv2.RANSAC)

print(F)

'''
Reference for this codes:https://docs.opencv.org/3.1.0/da/de9/tutorial_py_epipolar_geometry.html
'''
