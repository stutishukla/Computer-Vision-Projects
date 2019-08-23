#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 21:29:53 2018

@author: stutishukla
"""
import cv2
import numpy as np
#import matplotlib.pyplot as plt
#img = cv2.imread("/Users/stutishukla/Downloads/proj3_cse573/original_imgs/noise.jpg", cv2.IMREAD_GRAYSCALE)
#img=img/255

#list_img = list(img)
#list_img = [list(row) for row in list_img]

img = cv2.imread("/Users/stutishukla/Downloads/proj3_cse573/original_imgs/noise.jpg", cv2.IMREAD_GRAYSCALE)
#print(img)
img=img/255

a = np.asarray(img)
#print(a)
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
newUpdatedImageData = [[0 for x in range(imageW)] for y in range(imageH)]
updatedImageDataII = [[0 for x in range(imageW)] for y in range(imageH)]
newUpdatedImageDataII = [[0 for x in range(imageW)] for y in range(imageH)]
finalImage = [[0 for x in range(imageW)] for y in range(imageH)]


def calculateDilationat(widthindex , heightindex, curr_img, order=None):

    result = 0;

    for i in range(kernelH):
        for j in range(kernelW):
            currentWidthIndex = widthindex-1+j
            currentHeightIndex = heightindex-1+i

            if((currentHeightIndex<0) or (currentHeightIndex >= imageH)):
                continue

            if ((currentWidthIndex < 0) or (currentWidthIndex >= imageW)):
                continue
            
            if((curr_img[currentHeightIndex][currentWidthIndex]==1) and (kernelData[i][j]==1)):
                #print('255')
                if order == 1:
                    result = 255
                else:
                    result=1
                break
                

    return result

def calculateErosionat(widthindex , heightindex, curr_img, order=None):

    result = 0

    for i in range(kernelH):
        for j in range(kernelW):
            currentWidthIndex = widthindex-1+j
            currentHeightIndex = heightindex-1+i

            if((currentHeightIndex<0) or (currentHeightIndex >= imageH)):
                continue

            if ((currentWidthIndex < 0) or (currentWidthIndex >= imageW)):
                continue
            
            if((curr_img[currentHeightIndex][currentWidthIndex]==1) and (kernelData[i][j]==1)):
                result += 1
                
    if(result==9):
        if order== 1:
            result = 255
        else:
            result=1
    else:
         result=0
         
    return result

for i in range(imageH):
    for j in range(imageW):
        currentHeightOffset = i
        currentWidthOffset = j
        kernelValue = calculateDilationat(currentWidthOffset,currentHeightOffset, a)
        updatedImageData[i][j] = kernelValue

updatedImageDataH=len(updatedImageData)
updatedImageDataW=len(updatedImageData[0])
b=np.asarray(updatedImageData)

for i in range(updatedImageDataH):
    for j in range(updatedImageDataW):
        currentHeightOffset = i
        currentWidthOffset = j
        kernelValue2 = calculateErosionat(currentWidthOffset,currentHeightOffset, b)
        newUpdatedImageData[i][j] = kernelValue2
        
newUpdatedImageDataH = len(newUpdatedImageData)
newUpdatedImageDataW = len(newUpdatedImageData[0])
c=np.asarray(newUpdatedImageData)

for i in range(newUpdatedImageDataH):
    for j in range(newUpdatedImageDataW):
        currentHeightOffset = i
        currentWidthOffset = j
        kernelValue3=calculateErosionat(currentWidthOffset,currentHeightOffset, c)
        updatedImageDataII[i][j]=kernelValue3

updatedImageDataIIH = len(updatedImageDataII)
updatedImageDataIIW = len(updatedImageDataII[0])
d=np.asarray(updatedImageDataII)

for i in range(updatedImageDataIIH):
    for j in range(updatedImageDataIIW):
        currentHeightOffset = i
        currentWidthOffset = j
        kernelValue4=calculateDilationat(currentWidthOffset,currentHeightOffset, d, order=1)
        newUpdatedImageDataII[i][j]=kernelValue4
        




#image=normalizeImage(updatedImageData)

#print(updatedImageData)
#erosion=np.asarray(updatedImageData,dtype='uint8')
#cv2.imwrite('/Users/stutishukla/Downloads/Project3_images/task1/erosion.png',erosion)
#dialation=np.asarray(updatedImageData,dtype='uint8')
#cv2.imwrite('/Users/stutishukla/Downloads/Project3_images/task1/dialation.png',dialation)
closingOpening=np.asarray(newUpdatedImageDataII,dtype='uint8')
cv2.imwrite('/Users/stutishukla/Downloads/Project3_images/task1/res_noise1.jpg',closingOpening)
#print(np.max(closing))
e=np.asarray(newUpdatedImageDataII) 
e=e/255
newUpdatedImageDataIIH = len(newUpdatedImageDataII)
newUpdatedImageDataIIW = len(newUpdatedImageDataII[0])
#e=np.asarray(newUpdatedImageDataII) 
for i in range(newUpdatedImageDataIIH):
    for j in range(newUpdatedImageDataIIW):
        currentHeightOffset = i
        currentWidthOffset = j
        kernelValue5=calculateErosionat(currentWidthOffset,currentHeightOffset, e, order=1)
        finalImage[i][j]=kernelValue5
        
finalImage=np.asarray(finalImage)
closingOpening=closingOpening-finalImage
cv2.imwrite('/Users/stutishukla/Downloads/Project3_images/task1/res_bound1.jpg',closingOpening)
