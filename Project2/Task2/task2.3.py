#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Sun Oct 28 15:22:50 2018

@author: stutishukla
"""

import numpy as np
import cv2
import random


'''
Reference for this code drawEpipolarlines: 
    https://docs.opencv.org/3.1.0/da/de9/tutorial_py_epipolar_geometry.html

'''

UBIT = 'stutishu'
random.seed(sum(map(ord, UBIT)))

img1 = cv2.imread('/Users/stutishukla/Downloads/data/tsucuba_left.png',cv2.IMREAD_COLOR)          # queryImage
img2 = cv2.imread('/Users/stutishukla/Downloads/data/tsucuba_right.png',cv2.IMREAD_COLOR)          # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

#Feature Matching using flann
index_params = dict(algorithm=0, trees=5)
search_params= dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Apply ratio test
good_points = []
points1 = []
points2 = []

for i,(m,n) in enumerate(matches):
    if m.distance < 0.75*n.distance:
        good_points.append(m)
        points2.append(kp2[m.trainIdx].pt)
        points1.append(kp1[m.queryIdx].pt)

#for point in range(10):
#    points2.append(np.random.randint(kp2[m.trainIdx].pt))

points1 = np.int32(points1)
points2 = np.int32(points2)

F, mask = cv2.findFundamentalMat(points1,points2,cv2.LMEDS)

# We select only inlier points
points1 = points1[mask.ravel()==1]
points2 = points2[mask.ravel()==1]  

#Selecting 10 random inliers
points1_new=[]
points2_new=[]
for i in range(10):
    points1_new.append(random.choice(points1))
points1_new=np.int32(points1_new)

for i in range(10):
    points2_new.append(random.choice(points2))
points2_new=np.int32(points2_new)


def drawEpipolarlines(image1,image2,lines,ptsImage1,ptsImage2):
   
    row,column,x = image1.shape
  
    for row,pt1,pt2 in zip(lines,ptsImage1,ptsImage2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -row[2]/row[1] ])
        x1,y1 = map(int, [column, -(row[2]+row[0]*column)/row[1] ])
        image1 = cv2.line(image1, (x0,y0), (x1,y1), color,1)
        image1 = cv2.circle(image1,tuple(pt1),5,color,-1)
        image2 = cv2.circle(image2,tuple(pt2),5,color,-1)
    return image1,image2

# Find epilines corresponding to points in right image (second image) and drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(points2_new.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
image5,image6 = drawEpipolarlines(img1,img2,lines1,points1_new,points2_new)
     
# Find epilines corresponding to points in left image (first image) and drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(points1_new.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
image3,image4 = drawEpipolarlines(img2,img1,lines2,points2_new,points1_new)


cv2.imwrite('/Users/stutishukla/Downloads/task2_images/task2_epi_right.jpg',image5)
cv2.imwrite('/Users/stutishukla/Downloads/task2_images/task2_epi_left.jpg',image3)


'''
Reference for the code drawEpipolarlines: 
    https://docs.opencv.org/3.1.0/da/de9/tutorial_py_epipolar_geometry.html


Note: I tried working this code out using BFMatcher but I wasn't able to resolve the issues that it was throwing.
Hence, switched to FlannMatcher instead.
'''
