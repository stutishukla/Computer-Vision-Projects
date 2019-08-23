#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 00:25:04 2018

@author: stutishukla
"""
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("/Users/stutishukla/Downloads/proj1_cse573/task2.jpg", cv2.IMREAD_GRAYSCALE)

a=np.asarray(img)


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



UpdatedMatrix = resizeMatrix(a)
b=np.asarray(UpdatedMatrix)


def getDimensions(a):
    matHeight = len(a)
    if (matHeight == 0):
        return matHeight, matHeight
    matWidth = len(a[0])
    return matHeight, matWidth



def calculateGaussat(imagedata , widthindex, heightindex , gauss_kernel):
    result = 0
    kernalHeight,kernalWidth = getDimensions((gauss_kernel))
    if(kernalHeight != kernalWidth):
        return 0
    midPointOffset = (kernalWidth-1)/2
    imageHeight,imageWidth = getDimensions(imagedata)
    for i in range(kernalHeight):
        for j in range(kernalWidth):
            currentHeightIndex = heightindex - midPointOffset + i
            currentWidthIndex = widthindex - midPointOffset + j
            if ((currentHeightIndex < 0) or (currentHeightIndex >= imageHeight)):
                continue
            if ((currentWidthIndex < 0) or (currentWidthIndex >= imageWidth)):
                continue
            result += (UpdatedMatrix[int(currentHeightIndex)][int(currentWidthIndex)] * gauss_kernel[int(i)][int(j)])
    return int(result)



def fxy(x,y,sigma):
    temp1 = 1/(2 * math.pi * sigma * sigma)
    temp2 = math.exp( -1* ((x*x + y*y)/(2*sigma*sigma)))
    return  temp1*temp2



def getGuassianKernalforSigma(sigmaValue):
    total=0;
    gauss_kernel = [[0 for x in range(7)] for y in range(7)]
    for k in range(-3,4):
        for j in range(3,-4,-1):
           # result = ((1/(2*3.14*(sigmaValues[i]*sigmaValues[i])))*(math.exp(-((j*j)+(k*k))/2*sigmaValues[i]*sigmaValues[i])))
           result=fxy(j,k,sigmaValue)
           gauss_kernel[k + 3][3 - j] = result
           total=total+result
    return gauss_kernel,total

def applyGuassianKernaltoImage(imageData , gaussianKernal):
    imageHeight,imageWidth = getDimensions(imageData)
    updatedImageData = [[0 for x in range(imageWidth)] for y in range(imageHeight)]
    for i in range(imageHeight):
        for j in range(imageWidth):
            currentHeightOffset = i
            currentWidthOffset = j
            gossValue = calculateGaussat(imageData,currentWidthOffset,currentHeightOffset,gaussianKernal)
            updatedImageData[i][j] = gossValue
    return updatedImageData


def normalizeGaussKernel(gauss_kernel_raw,total):
     kernalHeight,kernalWidth = getDimensions((gauss_kernel_raw))
     for i in range(kernalHeight):
        for j in range(kernalWidth):
            gauss_kernel_raw[i][j]=gauss_kernel_raw[i][j]/total
     return gauss_kernel_raw
 
def DiffOfGauss(img1, img2):
    imgHeight, imgWidth = getDimensions(img2)
    DOG = [[0 for x in range(imgWidth)] for y in range(imgHeight)]
    for i in range(imgHeight):
        for j in range(imgWidth):
            DOG[i][j] = (img2[i][j] - img1[i][j])
    return DOG


def normalizeDOG(DOG):
    imgHeight, imgWidth = getDimensions(DOG)
    DOGnorm = [[0 for x in range(imgWidth)] for y in range(imgHeight)]
    
    for i in range(imgHeight):
        for j in range(imgWidth):
            DOGnorm[i][j] = 255*(DOG[i][j])
    return DOGnorm


def keypointMaximumDetection(dog_mid, dog_up, dog_down,finaMatrix):
    height, width = getDimensions(dog_mid)
    for h in range(1, height - 1):
        for w in range(1, width - 1):
            # traversing and comparing 26 neighbours'
            is_maxima = True
            for i in range(h - 1, h + 2):
                for j in range(w - 1, w + 2):
                    if (dog_mid[h][w] < dog_mid[i][j]) or (dog_mid[h][w] < dog_up[i][j]) or (dog_mid[h][w] < dog_down[i][j]):
                        is_maxima = False
                        break
                if not is_maxima:
                        break
            if is_maxima:
                finaMatrix.append([h,w])
            
    return


def keypointMinimumDetection(dog_mid, dog_up, dog_down,finaMatrix):
    height, width = getDimensions(dog_mid)
    for h in range(1, height - 1):
        for w in range(1, width - 1):
            # traversing and comparing 26 neighbours'
            is_minima = True
            for i in range(h - 1, h + 2):
                for j in range(w - 1, w + 2):
                    if (dog_mid[h][w] > dog_mid[i][j]) or (dog_mid[h][w] > dog_up[i][j]) or (dog_mid[h][w] > dog_down[i][j]):
                        is_minima = False
                        break
                if not is_minima:
                        break
            if is_minima:
                finaMatrix.append([h,w])
    return
            
# Program Starts from Here :

#img = [[0 for x in range(100)] for y in range(100)] #Use CV2 for reading image in img

#sigmaValues = [(1 / math.sqrt(2)), 1, (math.sqrt(2)), 2, 2 * (math.sqrt(2))] # 5 sigma values for which there will be 5 output images
#UpdatedMatrix=resizeMatrix(c)
sigmaValue1=(math.sqrt(2))
# for sigmaValue in sigmaValues: # Iterating for each sigma value
gauss_kernel_raw, total = getGuassianKernalforSigma(sigmaValue1)  # computing gaussian kernal for given sigma value
gauss_kernel = normalizeGaussKernel(gauss_kernel_raw, total)
outputImage1 = applyGuassianKernaltoImage(UpdatedMatrix, gauss_kernel)  # applying computed gaussian kernal to our image
# print outputImage here
b1 = np.asarray(outputImage1)
cv2.imwrite('/Users/stutishukla/Downloads/Result/octave2/task2-1.png',b1)

sigmaValue2=2
# for sigmaValue in sigmaValues: # Iterating for each sigma value
gauss_kernel_raw, total = getGuassianKernalforSigma(sigmaValue2)  # computing gaussian kernal for given sigma value
gauss_kernel = normalizeGaussKernel(gauss_kernel_raw, total)
outputImage2 = applyGuassianKernaltoImage(UpdatedMatrix, gauss_kernel)  # applying computed gaussian kernal to our image
# print outputImage here
b2 = np.asarray(outputImage2)
cv2.imwrite('/Users/stutishukla/Downloads/Result/octave2/task2-2.png',b2)

DogI = DiffOfGauss(b1, b2)
DogIA = normalizeDOG(DogI)
DogInorm = np.asarray(DogIA)
cv2.imwrite('/Users/stutishukla/Downloads/Result/octave2/task2-1DOG.png', DogInorm)

sigmaValue3=2*(math.sqrt(2))
# for sigmaValue in sigmaValues: # Iterating for each sigma value
gauss_kernel_raw, total = getGuassianKernalforSigma(sigmaValue3)  # computing gaussian kernal for given sigma value
gauss_kernel = normalizeGaussKernel(gauss_kernel_raw, total)
outputImage3 = applyGuassianKernaltoImage(UpdatedMatrix, gauss_kernel)  # applying computed gaussian kernal to our image
# print outputImage here
b3 = np.asarray(outputImage3)
cv2.imwrite('/Users/stutishukla/Downloads/Result/octave2/task2-3.png',b3)

DogII = DiffOfGauss(b2, b3)
DogIIA = normalizeDOG(DogII)
DogInorm = np.asarray(DogIIA)
cv2.imwrite('/Users/stutishukla/Downloads/Result/octave2/task2-2DOG.png', DogInorm)

sigmaValue4=4
# for sigmaValue in sigmaValues: # Iterating for each sigma value
gauss_kernel_raw, total = getGuassianKernalforSigma(sigmaValue4)  # computing gaussian kernal for given sigma value
gauss_kernel = normalizeGaussKernel(gauss_kernel_raw, total)
outputImage4 = applyGuassianKernaltoImage(UpdatedMatrix, gauss_kernel)  # applying computed gaussian kernal to our image
# print outputImage here
b4 = np.asarray(outputImage4)
cv2.imwrite('/Users/stutishukla/Downloads/Result/octave2/task2-4.png',b4)

DogIII = DiffOfGauss(b3, b4)
DogIIIA = normalizeDOG(DogIII)
DogInorm = np.asarray(DogIIIA)
cv2.imwrite('/Users/stutishukla/Downloads/Result/octave2/task2-3DOG.png', DogInorm)

sigmaValue5=4*(math.sqrt(2))
# for sigmaValue in sigmaValues: # Iterating for each sigma value
gauss_kernel_raw, total = getGuassianKernalforSigma(sigmaValue5)  # computing gaussian kernal for given sigma value
gauss_kernel = normalizeGaussKernel(gauss_kernel_raw, total)
outputImage5 = applyGuassianKernaltoImage(UpdatedMatrix, gauss_kernel)  # applying computed gaussian kernal to our image
# print outputImage here
b5 = np.asarray(outputImage5)
cv2.imwrite('/Users/stutishukla/Downloads/Result/octave2/task2-5.png',b5)

DogIV = DiffOfGauss(b4, b5)
DogIVA = normalizeDOG(DogIV)
DogInorm = np.asarray(DogIVA)
cv2.imwrite('/Users/stutishukla/Downloads/Result/octave2/task2-4DOG.png', DogInorm)

finaMatrix = []

#keypointMaximumDetection(DogIIA, DogIA, DogIIIA,finaMatrix)
keypointMaximumDetection(DogIIA, DogIA, DogIIIA,finaMatrix)
#keypointDetectMax = np.asarray(finaMatrix)
for keypoint in finaMatrix:
    cv2.circle(b,(keypoint[1],keypoint[0]),1,(255,255,0), -1)
keypointDetectMax=np.asarray(img)
cv2.imwrite('/Users/stutishukla/Downloads/Result/task2/keyPointDetection/OctaveII/keyPointDetectMaxDOGII.png', b)

keypointMinimumDetection(DogIIA, DogIA, DogIIIA,finaMatrix)
#keypointMinimumDetection(DogIIA, DogIA, DogIIIA,finaMatrix)
for keypoint in finaMatrix:
    cv2.circle(b,(keypoint[1],keypoint[0]),1,(255,255,0), -1)
cv2.imwrite('/Users/stutishukla/Downloads/Result/task2/keyPointDetection/OctaveII/keyPointDetectDOGII.png', b)

keypointMaximumDetection(DogIIIA, DogIIA, DogIVA,finaMatrix)
#keypointDetectMax = np.asarray(finaMatrix)
for keypoint in finaMatrix:
    cv2.circle(b,(keypoint[1],keypoint[0]),1,(255,255,0), -1)
keypointDetectMax=np.asarray(img)
cv2.imwrite('/Users/stutishukla/Downloads/Result/task2/keyPointDetection/OctaveII/keyPointDetectMaxDOGIII.png', b)

keypointMinimumDetection(DogIIIA, DogIIA, DogIVA,finaMatrix)
#keypointMinimumDetection(DogIIA, DogIA, DogIIIA,finaMatrix)
for keypoint in finaMatrix:
    cv2.circle(b,(keypoint[1],keypoint[0]),1,(255,255,0), -1)
cv2.imwrite('/Users/stutishukla/Downloads/Result/task2/keyPointDetection/OctaveII/keyPointDetectDOGIII.png', b)