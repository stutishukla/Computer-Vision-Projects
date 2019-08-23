#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:22:50 2018

@author: stutishukla
"""
import numpy as np
import cv2


print(cv2.__version__)
'''
References for the code of warpImages:
  1.http://answers.opencv.org/question/144252/perspective-transform-without-crop/
  2.https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective/20355545#20355545

'''

img1 = cv2.imread('/Users/stutishukla/Downloads/data/mountain1.jpg',cv2.IMREAD_GRAYSCALE)          # queryImage
img2 = cv2.imread('/Users/stutishukla/Downloads/data/mountain2.jpg',cv2.IMREAD_GRAYSCALE)          # trainImage

def warpImages(img1, img2, M):
    
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    v1 = np.float32([[0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0]]).reshape(-1,1,2) #getting the corner points of image1
    v2 = np.float32([[0,0],[0,h2-1],[w2-1,h2-1],[w2-1,0]]).reshape(-1,1,2) #getting the corner points of image2
    v3 = cv2.perspectiveTransform(v2, M)
    v = np.concatenate((v1, v3))
    x1, y1, x2, y2 = cv2.boundingRect(v) # Find the bounding rectangle
    Mt = np.array([[1,0,-x1],[0,1,-y1],[0,0,1]]) # Computing the translation matrix to move x1,x2 to (0,0)
    result = cv2.warpPerspective(img2, Mt.dot(M), (x2, y2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT) #Applying transformation
    result[(-y1):h1+(-y1),(-x1):w1+(-x1)] = img1
    return result

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
    matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
    
#Finally we draw our inliers (if successfully found the object) or matching keypoints (if failed).
result = warpImages(img2, img1, matrix)

cv2.imwrite('/Users/stutishukla/Downloads/task1_images/task1_pano.jpg',result)

'''
References for the code of warpImages:
  1.http://answers.opencv.org/question/144252/perspective-transform-without-crop/
  2.https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective/20355545#20355545

'''
