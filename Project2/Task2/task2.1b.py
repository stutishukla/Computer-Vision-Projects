#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:22:50 2018

@author: stutishukla
"""
'''
Referred code from ::: https://docs.opencv.org/3.4.3/dc/dc3/tutorial_py_matcher.html 
Topic ::: Brute-Force Matching with SIFT Descriptors and Ratio Test

'''
import numpy as np
import cv2
from matplotlib import pyplot as plt


img1 = cv2.imread('/Users/stutishukla/Downloads/data/tsucuba_left.png',cv2.IMREAD_COLOR)          # queryImage
img2 = cv2.imread('/Users/stutishukla/Downloads/data/tsucuba_right.png',cv2.IMREAD_COLOR)          # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

bf = cv2.BFMatcher()
# Feature Matching using BFMatcher with default params
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
   if m.distance < 0.75*n.distance:  #descriptors distances in two images
        good.append([m])


#cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

cv2.imwrite('/Users/stutishukla/Downloads/task2_images/task2_matches_knn.jpg',img3)

'''
Referred code from ::: https://docs.opencv.org/3.4.3/dc/dc3/tutorial_py_matcher.html 
Topic ::: Brute-Force Matching with SIFT Descriptors and Ratio Test

'''
