# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script fle.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("/Users/stutishukla/Downloads/proj3_cse573/original_imgs/turbine-blade.jpg", cv2.IMREAD_GRAYSCALE)
a = np.asarray(img)
print(a)
kernelW, kernelH = 3, 3
kernelData = [[0 for x in range(kernelW)] for y in range(kernelH)]
kernelData[0][0] = -1
kernelData[0][1] = -1
kernelData[0][2] = -1
kernelData[1][0] = -1
kernelData[1][1] = 8
kernelData[1][2] = -1
kernelData[2][0] = -1
kernelData[2][1] = -1
kernelData[2][2] = -1

imageH=len(img)
imageW=len(img[0])
#print(imageH)
#print(imageW)
updatedImageData = [[0 for x in range(imageW)] for y in range(imageH)]

def calculateCorrelat(widthindex , heightindex):

    result = 0;

    for i in range(kernelH):
        for j in range(kernelW):
            currentWidthIndex = widthindex-1+j
            currentHeightIndex = heightindex-1+i

            if((currentHeightIndex<0) or (currentHeightIndex >= imageH)):
                continue

            if ((currentWidthIndex < 0) or (currentWidthIndex >= imageW)):
                continue

            result += (a[currentHeightIndex][currentWidthIndex]*kernelData[i][j])

    return result

def getDimensionsOfMatrix(a):
    matHeight = len(a)
    if (matHeight == 0):
        return matHeight, matHeight
    matWidth = len(a[0])
    return matWidth, matHeight

def calculateThresholding(image):
    matrix=[]
    threshold=1100
    imageWidth, imageHeight= getDimensionsOfMatrix(image)
    for heightIndex in range(imageHeight):
        for widthIndex in range(imageWidth):
          if(image[heightIndex][widthIndex]>threshold):
              image[heightIndex][widthIndex]=255
              print(heightIndex, widthIndex)
              matrix.append([heightIndex, widthIndex])
         # elif(image[heightIndex][widthIndex] <= threshold):
          else:
              image[heightIndex][widthIndex] = 0
    return image,matrix

for i in range(imageH):
    for j in range(imageW):
        currentWidthOffset = j
        currentHeightOffset = i
        kernelValue = calculateCorrelat(currentWidthOffset,currentHeightOffset)
        updatedImageData[i][j] = kernelValue
        
image=np.asarray(updatedImageData)
image=np.abs(image)
print(np.max(image))
result, matrix=calculateThresholding(image)
print(matrix[0][0])
print(matrix[0][1])
result=np.asarray(result)
font = cv2.FONT_HERSHEY_SIMPLEX

#for someList in matrix:
cv2.imwrite('/Users/stutishukla/Downloads/Project3_images/task3/point2.jpg',result)
image=cv2.imread("/Users/stutishukla/Downloads/Project3_images/task3/point2.jpg")
cv2.circle(image,(matrix[0][1],matrix[0][0]), 10, (0,0,255),2) 
cv2.putText(image,'445,249',(280,300), font, 1,(0,255,0),2,cv2.LINE_AA)
cv2.circle(a,(matrix[0][1],matrix[0][0]), 40, (0,0,255),2) 
cv2.putText(a,'445,249',(280,300), font, 1,(0,255,0),2,cv2.LINE_AA)
cv2.imwrite('/Users/stutishukla/Downloads/Project3_images/task3/point.jpg',image)
cv2.imwrite('/Users/stutishukla/Downloads/Project3_images/task3/points.jpg',a)

  



    


    
            
        
    
        
