#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 00:01:05 2018

@author: stutishukla
"""
import math
import matplotlib.pyplot as plt

Xw, Xh = 2, 10
X = [[0 for x in range(Xw)] for y in range(Xh)]
X[0][0] = 5.9
X[0][1] = 3.2
X[1][0] = 4.6
X[1][1] = 2.9
X[2][0] = 6.2
X[2][1] = 2.8
X[3][0] = 4.7
X[3][1] = 3.2
X[4][0] = 5.5
X[4][1] = 4.2
X[5][0] = 5.0
X[5][1] = 3.0
X[6][0] = 4.9
X[6][1] = 3.1
X[7][0] = 6.7
X[7][1] = 3.1
X[8][0] = 5.1
X[8][1] = 3.8
X[9][0] = 6.0
X[9][1] = 3.0




def Calculate(arr_mean1, arr_mean2, arr_mean3):
    temp1=[]
    temp2=[]
    temp3=[]
    
    #checker= 1
    for i in range(Xh):
        temp=[]
        for j in range(Xw):
            temp.append(X[i][j])
        
        dist1= fsqrt(temp,arr_mean1)
        dist2= fsqrt(temp,arr_mean2)
        dist3= fsqrt(temp,arr_mean3)
        if((dist1 < dist2) and (dist1 < dist3)):
            temp1.append(temp)
        elif((dist2 < dist1) and (dist2 < dist3)):
            temp2.append(temp)
        else:
            temp3.append(temp)
    print(temp1)
    print(temp2)
    print(temp3)
    #plotting temp1
    
    for point in temp1:
        plt.scatter(point[0],point[1],color='red', marker='^')

    for point in temp2:
        plt.scatter(point[0], point[1], color='blue', marker='^')

    for point in temp3:
        plt.scatter(point[0], point[1], color='green', marker='^')
        plt.savefig('/Users/stutishukla/Downloads/task3_images/task3_iter1_a.jpg', transparent=True, bbox_inches='tight')
       
    
    mean1= calculateMean(temp1)
    mean2= calculateMean(temp2)
    mean3= calculateMean(temp3)
    
    return mean1, mean2, mean3
    
   # iterations(mean1, mean2, mean3)
        
def calculateValue(sum_value, length):
    mean= sum_value/length
    return mean
        
def calculateMean(temp):
    sum_value_row=0
    sum_value_col=0
    for i in range(len(temp)):
        sum_value_row=sum_value_row+temp[i][0]
        sum_value_col=sum_value_col+temp[i][1]
    mean_row= calculateValue(sum_value_row, len(temp)) 
    mean_col = calculateValue(sum_value_col, len(temp))
    
    mean_arr=[]
    mean_arr.append(mean_row)
    mean_arr.append(mean_col)
    return mean_arr

    
def fsqrt(a,arr):
    dist = math.sqrt(math.pow(arr[0]-a[0],2) + math.pow(arr[1]-a[1],2))
    return dist


arr_mean1= [6.2, 3.2]
arr_mean2= [6.6, 3.7]
arr_mean3= [6.5, 3.0]       
mean1, mean2, mean3= Calculate(arr_mean1, arr_mean2, arr_mean3)  
   
    
    
    
            
            