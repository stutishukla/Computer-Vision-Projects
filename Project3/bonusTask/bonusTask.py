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

img = cv2.imread("hough.jpg", cv2.IMREAD_GRAYSCALE)
a=np.asarray(img)
img2 = cv2.imread("hough.jpg")
img2=np.asarray(img2)
# b=np.asarray(img2)
# originCopy = b.copy()
# originCopy2 = b.copy()
# print(originCopy)
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
# sobelWy, sobelHy = 3, 3
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
imageH = len(img)
imageW = len(img[0])
# print(imageH)
# print(imageW)
updatedImageDataX = [[0 for x in range(imageW)] for y in range(imageH)]
updatedImageDataY = [[0 for x in range(imageW)] for y in range(imageH)]


def calculateSobelatX(widthindex, heightindex):
    result = 0;
    for i in range(sobelH):
        for j in range(sobelW):
            currentWidthIndex = widthindex - 1 + j
            currentHeightIndex = heightindex - 1 + i
            if ((currentHeightIndex < 0) or (currentHeightIndex >= imageH)):
                continue
            if ((currentWidthIndex < 0) or (currentWidthIndex >= imageW)):
                continue
            result += (a[currentHeightIndex][currentWidthIndex] * sobelData[i][j])
    return result


def calculateSobelatY(widthindex, heightindex):
    result = 0;
    for i in range(sobelH):
        for j in range(sobelW):
            currentWidthIndex = widthindex - 1 + j
            currentHeightIndex = heightindex - 1 + i
            if ((currentHeightIndex < 0) or (currentHeightIndex >= imageH)):
                continue
            if ((currentWidthIndex < 0) or (currentWidthIndex >= imageW)):
                continue
            result += (a[currentHeightIndex][currentWidthIndex] * sobelDataY[i][j])
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


def calculateABR(image):
    abrList = []
    for x in range(matrixHeight):
        for y in range(matrixWidth):
            if (image[x][y] == 255):
                for r in range(22, 25):
                    for theta in range(0, 361):
                        # d = (x * (math.cos(math.radians(theta))) - y * (math.sin(math.radians(theta))))
                        a = int(y - (r * (math.cos(math.radians(theta)))))
                        # print(a)
                        b = int(x + (r * (math.sin(math.radians(theta)))))
                        # print(b)
                        abrList.append([a, b, r])
    # print(abrList)
    abrList = np.asarray(abrList)
    return abrList


def calculateArray(finalList):
    maxA = int(np.max(finalList[:, 0]))
    maxB = int(np.max(finalList[:, 1]))
    maxR = int(np.max(finalList[:, 2]))
    print(maxA)
    print(maxB)
    print(maxR)
    # AccMatrix = [[[0 for x in range(maxA+1)] for y in range(maxB+1)] for z in range(maxR+1)]
    AccMatrix = np.zeros((maxA + 1, maxB + 1, maxR + 1))
    print("hola")
    for i in range(len(finalList)):
        if (finalList[i][0] >= 0) and (finalList[i][1] >= 0):
            AccMatrix[finalList[i][0], finalList[i][1], finalList[i][2]] += 1
    # for it in finalList:
    #     a = int(it[0])
    #     b = int(it[1])
    #     r = int(it[2])
    #     # AccMatrix[a][b][r] += 1
    #     AccMatrix[a][b][r] += 1
    # print(AccMatrix)
    max_val_list = [maxA, maxB, maxR]
    return AccMatrix, max_val_list


def calculateIndices(matrix, val_list):
    newDTheta = []
    h = val_list[0]
    w = val_list[1]
    r = val_list[2]
    maxVal = int(np.max(matrix))
    print('maxVal'+str(maxVal))
    threshold = maxVal * 0.7
    print(threshold)
    for i in range(h+1):
        for j in range(w+1):
            for k in range(r+1):
                # if matrix[i][j][k] > threshold:
                #     newDTheta.append([i, j, k])
                if matrix[i, j, k] > threshold:
                    newDTheta.append((i, j, k))
    print('string'+str(newDTheta))
    return newDTheta


def formCircles(newDTheta):
  for someList in newDTheta:
       cv2.circle(img2, (someList[0], someList[1]), (someList[2]), (255, 0, 0), 1)


for i in range(imageH):
    for j in range(imageW):
        currentWidthOffset = j
        currentHeightOffset = i
        sobelValueX = calculateSobelatX(currentWidthOffset, currentHeightOffset)
        updatedImageDataX[i][j] = sobelValueX

for i in range(imageH):
    for j in range(imageW):
        currentWidthOffset = j
        currentHeightOffset = i
        sobelValueY = calculateSobelatY(currentWidthOffset, currentHeightOffset)
        updatedImageDataY[i][j] = sobelValueY

newMatrix = []
finalMatrix = []
matrixWidth, matrixHeight = getDimensionsOfMatrix(updatedImageDataX)
for heightIndex in range(matrixHeight):
    for widthIndex in range(matrixWidth):
        try:
            newMatrix.append(math.sqrt(
                (updatedImageDataX[heightIndex][widthIndex] ** 2) + (updatedImageDataY[heightIndex][widthIndex] ** 2)))
        except:
            print("exception")
            print(heightIndex)
            print(widthIndex)
    finalMatrix.append(newMatrix)
    newMatrix = []

print(np.max(finalMatrix))
image = finalMatrix / np.max(finalMatrix)
hough = np.asarray(image)
image = hough * 255
threshold_value = 25
image = thresholding(threshold_value, image)
finalList = calculateABR(image)
matrix, values_list = calculateArray(finalList)
newDTheta = calculateIndices(matrix, values_list)
formCircles(newDTheta)
cv2.imwrite('stuti/houghCircle.png', img2)
