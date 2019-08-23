#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 21:29:53 2018

@author: stutishukla
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
#img = cv2.imread("/Users/stutishukla/Downloads/proj3_cse573/original_imgs/noise.jpg", cv2.IMREAD_GRAYSCALE)
#img=img/255

#list_img = list(img)
#list_img = [list(row) for row in list_img]

img = cv2.imread("/Users/stutishukla/Downloads/proj3_cse573/original_imgs/noise.jpg", cv2.IMREAD_GRAYSCALE)
img=img/255
a = np.asarray(img)
print(a)
kernelW, kernelH = 3, 3
kernelData = [[0 for x in range(kernelW)] for y in range(kernelH)]
kernelData[0][0] = 1
kernelData[0][1] = 1
kernelData[0][2] = 1
kernelData[1][0] = 1
kernelData[1][1] = 1
kernelData[1][2] = 1
kernelData[2][0] = 1
kernelData[2][1] = 1
kernelData[2][2] = 1

imageH=len(img)
imageW=len(img[0])

updatedImageData = [[0 for x in range(imageW)] for y in range(imageH)]

def calculateSobelat(widthindex , heightindex):

    result = 0;

    for i in range(kernelH):
        for j in range(kernelW):
            currentWidthIndex = widthindex-1+j
            currentHeightIndex = heightindex-1+i

            if((currentHeightIndex<0) or (currentHeightIndex >= imageH)):
                continue

            if ((currentWidthIndex < 0) or (currentWidthIndex >= imageW)):
                continue
            
            if((a[currentHeightIndex][currentWidthIndex]==1) and (kernelData[i][j]==1)):
                result=1
                break

    return result

def getDimensions(a):
    
    matHeight = len(a)
    if (matHeight == 0):
        return matHeight, matHeight
    matWidth = len(a[0])
    return matHeight, matWidth


def normalizeImage(updatedImage):
    
    imgHeight, imgWidth = getDimensions(updatedImage)
    updatedImagenorm = [[0 for x in range(imgWidth)] for y in range(imgHeight)]
   
    for i in range(imgHeight):
        for j in range(imgWidth):
            updatedImagenorm[i][j] = 128 + int(updatedImage[i][j]/2)
    return updatedImagenorm

for i in range(imageH):
    for j in range(imageW):
        currentHeightOffset = i
        currentWidthOffset = j
        sobelValue = calculateSobelat(currentWidthOffset,currentHeightOffset)
        updatedImageData[i][j] = sobelValue
#image=normalizeImage(updatedImageData)

#print(updatedImageData)
dialation=np.asarray(updatedImageData)
cv2.imwrite('/Users/stutishukla/Downloads/Project3_images/task1/dialation.png',dialation)