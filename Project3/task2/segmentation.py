# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script fle.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("/Users/stutishukla/Downloads/proj3_cse573/original_imgs/segment.jpg", cv2.IMREAD_GRAYSCALE)
a = np.asarray(img)

def getDimensionsOfMatrix(a):
    matHeight = len(a)
    if (matHeight == 0):
        return matHeight, matHeight
    matWidth = len(a[0])
    return matWidth, matHeight

def calculateThresholding(image):
   # matrix=[]
    threshold=204
    for heightIndex in range(imageHeight):
        for widthIndex in range(imageWidth):
          if(image[heightIndex][widthIndex]>threshold):
              image[heightIndex][widthIndex]=255
              #print(heightIndex, widthIndex)
          else:
              image[heightIndex][widthIndex] = 0
    return image

def calculateBoundingBox(image):
    matrix=[]
    found = True
    for widthIndex in range(imageWidth):
        for heightIndex in range(imageHeight):
            if(image[heightIndex][widthIndex]==255):
                found = False
                matrix.append([heightIndex,widthIndex])
                break
        if not found:
            break
    found=True
    
    for heightIndex in range(imageHeight-1,0,-1):
        for widthIndex in range(imageWidth):
            if(image[heightIndex][widthIndex]==255):
                found = False
                matrix.append([heightIndex,widthIndex])
                break
        if not found:
            break
    found=True
    
    for widthIndex in range(imageWidth-1,0,-1):
        for heightIndex in range(imageHeight):
            if(image[heightIndex][widthIndex]==255):
                found = False
                matrix.append([heightIndex,widthIndex])
                break
        if not found:
            break
    found=True
    
    for heightIndex in range(imageHeight):
        for widthIndex in range(imageWidth):
            if(image[heightIndex][widthIndex]==255):
                found = False
                matrix.append([heightIndex,widthIndex])
                break
        if not found:
            break
    found=True
    
    print(matrix)
    return matrix

def findPoints(image):
    histogram={}
    for i in range(imageHeight):
       for j in range(imageWidth):
        key = image[i][j]
        if key in histogram:
          histogram[key] = histogram.get(key)+1
        else:
          histogram[key] = 1
    return histogram
           
imageWidth, imageHeight= getDimensionsOfMatrix(a) 
histogram=findPoints(a)

keys=list(histogram.keys())
values=list(histogram.values())
plt.plot(keys, values)
plt.savefig('histogram.png')
plt.savefig('/Users/stutishukla/Downloads/Project3_images/task3/histogram.png', transparent=True, bbox_inches='tight')
   
image=calculateThresholding(a)
image=np.asarray(image)
cv2.imwrite('/Users/stutishukla/Downloads/Project3_images/task3/segment.jpg',image)
matrix=calculateBoundingBox(image)
x4=matrix[3][0]
y1=matrix[0][1]
x2=matrix[1][0] 
y3=matrix[2][1]
print(x4)
print(y1) 
print(x2)
print(y3)
cv2.rectangle(image,(y1,x4),(y3,x2),(255,255,255),3)
cv2.imwrite('/Users/stutishukla/Downloads/Project3_images/task3/boundingBox.jpg',image)
