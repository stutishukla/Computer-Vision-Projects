# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script fle.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("/Users/stutishukla/Downloads/proj1_cse573/task1.png", cv2.IMREAD_GRAYSCALE)

list_img = list(img)
list_img = [list(row) for row in list_img]

img = cv2.imread("/Users/stutishukla/Downloads/proj1_cse573/task1.png", cv2.IMREAD_GRAYSCALE)
a = np.asarray(img)
sobelW, sobelH = 3, 3
sobelData = [[0 for x in range(sobelW)] for y in range(sobelH)]
sobelData[0][0] = -1
sobelData[0][1] = -2
sobelData[0][2] = -1
sobelData[1][0] = 0
sobelData[1][1] = 0
sobelData[1][2] = 0
sobelData[2][0] = 1
sobelData[2][1] = 2
sobelData[2][2] = 1

imageH=len(img)
imageW=len(img[0])

updatedImageData = [[0 for x in range(imageW)] for y in range(imageH)]

def calculateSobelat(widthindex , heightindex):

    result = 0;
    for i in range(sobelH):
        for j in range(sobelW):
            currentWidthIndex = widthindex-1+j
            currentHeightIndex = heightindex-1+i

            if((currentHeightIndex<0) or (currentHeightIndex >= imageH)):
                continue

            if ((currentWidthIndex < 0) or (currentWidthIndex >= imageW)):
                continue

            result += (a[currentHeightIndex][currentWidthIndex]*sobelData[i][j])

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
        currentWidthOffset = j
        currentHeightOffset = i
        sobelValue = calculateSobelat(currentWidthOffset,currentHeightOffset)
        updatedImageData[i][j] = sobelValue
image=normalizeImage(updatedImageData)

#print(updatedImageData)
gradY=np.asarray(image)
cv2.imwrite('/Users/stutishukla/Downloads/Result/task1/GradientY.png',gradY)


    


    
            
        
    
        
