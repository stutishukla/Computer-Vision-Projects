#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 00:03:08 2018

@author: stutishukla
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
img = cv2.imread("/Users/stutishukla/Downloads/proj3_cse573/original_imgs/hough.jpg", cv2.IMREAD_GRAYSCALE)

a = np.asarray(img)
img2 = cv2.imread("/Users/stutishukla/Downloads/proj3_cse573/original_imgs/hough.jpg")
b=np.asarray(img2)
originCopy = b.copy()
originCopy2 = b.copy()
#print(originCopy)
sobelW, sobelH = 3, 3
sobelData = [[0 for x in range(sobelW)] for y in range(sobelH)]
sobelData[0][0] = -1
sobelData[0][1] = 0
sobelData[0][2] = 1
sobelData[1][0] = -2
sobelData[1][1] = 0
sobelData[1][2] = 2
sobelData[2][0] = -1
sobelData[2][1] = 0
sobelData[2][2] = 1

#sobelWy, sobelHy = 3, 3
sobelDataY = [[0 for x in range(sobelW)] for y in range(sobelH)]
sobelDataY[0][0] = -1
sobelDataY[0][1] = -2
sobelDataY[0][2] = -1
sobelDataY[1][0] = 0
sobelDataY[1][1] = 0
sobelDataY[1][2] = 0
sobelDataY[2][0] = 1
sobelDataY[2][1] = 2
sobelDataY[2][2] = 1
imageH=len(img)
imageW=len(img[0])
#print(imageH)
#print(imageW)
updatedImageDataX = [[0 for x in range(imageW)] for y in range(imageH)]
updatedImageDataY = [[0 for x in range(imageW)] for y in range(imageH)]
def calculateSobelatX(widthindex , heightindex):

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

def calculateSobelatY(widthindex , heightindex):

    result = 0;
    for i in range(sobelH):
        for j in range(sobelW):
            currentWidthIndex = widthindex-1+j
            currentHeightIndex = heightindex-1+i

            if((currentHeightIndex<0) or (currentHeightIndex >= imageH)):
                continue

            if ((currentWidthIndex < 0) or (currentWidthIndex >= imageW)):
                continue

            result += (a[currentHeightIndex][currentWidthIndex]*sobelDataY[i][j])

    return result


def getDimensionsOfMatrix(a):
    matHeight = len(a)
    if (matHeight == 0):
        return matHeight, matHeight
    matWidth = len(a[0])
    return matWidth, matHeight

# changes
def thresholding(threshold, matrix):
    w1, h1 = matrix.shape[1], matrix.shape[0]
    for x in range(h1):
        for y in range(w1):
            if matrix[x][y] > threshold:
                matrix[x][y] = 255
            elif matrix[x][y] <= threshold:
                matrix[x][y] = 0
    return matrix

def calculateDTheta(image):
    dList=[]
    for x in range(matrixHeight):
        for y in range(matrixWidth):
            if(image[x][y]==255):
               for theta in range(0, 181):
                   d = (x * (math.cos(math.radians(theta))) - y * (math.sin(math.radians(theta))))
                   dList.append([d, theta])
    dList = np.asarray(dList)
    print(dList[0][0], dList[0][1])
    minD =int(np.min(dList[:, 0]))
    print('minD = ', minD)
    dList[:, 0] = (dList[:, 0] + abs(minD)) 
    print('new values ',dList[0][0], dList[0][1])
    return dList, minD
    
    
def calculateArray(finalList):
     maxD = int(np.max(finalList[:, 0]))
     print(maxD)
     AccMatrix = [[0 for x in range(181)] for y in range(maxD+1)]
    # AccMatrix = np.zeros((maxD+1, 181))
     for it in finalList:
         d=int(it[0])
         theta=int(it[1])
         AccMatrix[d][theta] += 1
     return AccMatrix    

def calculateIndices(matrix):
    newDTheta=[]
    w,h=getDimensionsOfMatrix(matrix)
    print(w)
    print(h)
    maxVal= int(np.max(matrix))
    print(maxVal)
    threshold=maxVal*0.4
    print(threshold)
    for i in range(h):
        for j in range(w):
            if(matrix[i][j]>threshold):
                newDTheta.append([i,j])
    print(newDTheta)
    return newDTheta

def formLines(newDTheta, minD):
   # minD =int(np.min(newDTheta[:, 0]))
    orgHeightX, orgWidthY=getDimensionsOfMatrix(originCopy)
    print(orgWidthY)
    y1=0
    y2=900
    theta1=[87,88,89]
    theta2=[53,54,55]
    for someList in newDTheta:
        newD=someList[0]-abs(minD)
        theta=someList[1]
        if theta in theta1:
            x1=int((newD + (y1 * (math.sin(math.radians(theta)))))/(math.cos(math.radians(theta))))
            x2=int((newD + (y2 * (math.sin(math.radians(theta)))))/(math.cos(math.radians(theta))))
            cv2.line(originCopy,(y1,x1),(y2,x2),(0,0,255),2)
            cv2.imwrite('/Users/stutishukla/Downloads/Project3_images/task2/red_lines.jpg',originCopy)
        if theta in theta2:
            x1=int((newD + (y1 * (math.sin(math.radians(theta)))))/(math.cos(math.radians(theta))))
            x2=int((newD + (y2 * (math.sin(math.radians(theta)))))/(math.cos(math.radians(theta))))
            cv2.line(originCopy2,(y1,x1),(y2,x2),(255,0,0),2)
            cv2.imwrite('/Users/stutishukla/Downloads/Project3_images/task2/blue_lines.jpg',originCopy2)

        
for i in range(imageH):
    for j in range(imageW):
        currentWidthOffset = j
        currentHeightOffset = i
        sobelValueX = calculateSobelatX(currentWidthOffset,currentHeightOffset)
        updatedImageDataX[i][j] = sobelValueX
        
for i in range(imageH):
    for j in range(imageW):
        currentWidthOffset = j
        currentHeightOffset = i
        sobelValueY = calculateSobelatY(currentWidthOffset,currentHeightOffset)
        updatedImageDataY[i][j] = sobelValueY
        
newMatrix = []
finalMatrix=[]
matrixWidth, matrixHeight = getDimensionsOfMatrix(updatedImageDataX)
for heightIndex in range(matrixHeight):
    for widthIndex in range(matrixWidth):
        try:
            newMatrix.append(math.sqrt((updatedImageDataX[heightIndex][widthIndex]**2)+(updatedImageDataY[heightIndex][widthIndex]**2)))
        except:
            print("exception")
            print(heightIndex)
            print(widthIndex)
    finalMatrix.append(newMatrix)
    newMatrix = []
                
                
print(np.max(finalMatrix))
image=finalMatrix/np.max(finalMatrix)

hough=np.asarray(image)
image = hough*255
#print(np.max(image))
#print(np.min(image))

threshold_value = 25
image = thresholding(threshold_value, image)
finalList,minD=calculateDTheta(image)
matrix=calculateArray(finalList)
newDTheta=calculateIndices(matrix)
formLines(newDTheta, minD)
matrix=np.asarray(matrix)
#print(finalList[0][0], finalList[0][1])
#print("after thresholding")
#print(np.max(image))
#print(np.min(image))
cv2.imwrite('/Users/stutishukla/Downloads/Project3_images/task2/hough.png',image)
cv2.imwrite('/Users/stutishukla/Downloads/Project3_images/task2/matrix.png',matrix)