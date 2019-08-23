#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:22:50 2018

@author: stutishukla
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
print(cv2.__version__)

img1 = cv2.imread('/Users/stutishukla/Downloads/data/tsucuba_left.png',cv2.IMREAD_GRAYSCALE)          # queryImage
img2 = cv2.imread('/Users/stutishukla/Downloads/data/tsucuba_right.png',cv2.IMREAD_GRAYSCALE)          # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# Feature Matching using BFMatcher with default params
#matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
#good = []
#for m,n in matches:
#    if m.distance < 0.75*n.distance:  #descriptors distances in two images
 #       good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
#img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)



#Feature Matching using flann
index_params = dict(algorithm=0, trees=5)
search_params= dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# Apply ratio test
good_points = []
for m,n in matches:
   if m.distance < 0.75*n.distance:  #descriptors distances in two images
       good_points.append(m)


img3= cv2.drawMatches(img1, kp1, img2, kp2, good_points, img2)

cv2.imwrite('/Users/stutishukla/Downloads/task1_images/task1.2.png',img3)


