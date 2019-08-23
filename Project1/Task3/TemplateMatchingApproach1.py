#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 14:41:00 2018

@author: stutishukla
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/Users/stutishukla/Downloads/proj1_cse573/task3/image1.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread('/Users/stutishukla/Downloads/proj1_cse573/task3/template.png')
template_gray=cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

def getDimensionsOfMatrix(a):
    matHeight = len(a)
    if (matHeight == 0):
        return matHeight, matHeight
    matWidth = len(a[0])
    return matWidth, matHeight

def resizeMatrix(a):
    resizedMatrix = []
    matrixWidth, matrixHeight = getDimensionsOfMatrix(a)

    for heightIndex in range(matrixHeight):

        if(heightIndex%2 == 1):
            continue
        tempRow = []
        for widthIndex in range(matrixWidth):
            if(widthIndex%2 == 0):
                tempRow.append(a[heightIndex][widthIndex])
        resizedMatrix.append(tempRow)

    return resizedMatrix

template_grayResized=resizeMatrix(template_gray)
a=np.asarray(template_grayResized)

blurImage = cv2.GaussianBlur(img_gray,(3,3),0)
blurTemplate = cv2.GaussianBlur(a,(3,3),0)

laplacian1 = cv2.Laplacian(blurImage,cv2.CV_64F)
laplacian2 = cv2.Laplacian(blurTemplate,cv2.CV_64F)

w, h= a.shape[::-1]

# Apply template Matching
res = cv2.matchTemplate(laplacian1.astype(np.float32),laplacian2.astype(np.float32),cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)


top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img,top_left, bottom_right, 255, 2)
cv2.imshow('Task3.png',img)
cv2.imwrite('/Users/stutishukla/Downloads/Result/task3/image1.png',img)
    

