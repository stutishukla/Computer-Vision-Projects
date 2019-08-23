#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 21:49:10 2018

@author: stutishukla
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:22:50 2018

@author: stutishukla
"""
'''
Referred code from here: https://www.programcreek.com/python/example/89444/cv2.drawMatches
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
#print(cv2.__version__)
UBIT = 'stutishu'
np.random.seed(sum(map(ord, UBIT)))

img1 = cv2.imread('/Users/stutishukla/Downloads/data/mountain1.jpg',cv2.IMREAD_GRAYSCALE)          # queryImage
img2 = cv2.imread('/Users/stutishukla/Downloads/data/mountain2.jpg',cv2.IMREAD_GRAYSCALE)          # trainImage

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

if len(good_points) > 10:  #if there are atleast 10 points then only I ll be finding homography, otherwise not.
    query_pts = np.float32([kp1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2) #extracting the position of the  good points in query image 
    train_pts = np.float32([kp2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)#extracting the position of the  good points in train image 


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        flags = 2)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,np.random.choice(good_points, 10),None,**draw_params)
cv2.imwrite('/Users/stutishukla/Downloads/task1_images/task1_matches.jpg',img3)


'''
Referred code from here: https://www.programcreek.com/python/example/89444/cv2.drawMatches

'''