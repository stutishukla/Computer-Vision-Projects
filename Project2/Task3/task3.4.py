import math
import cv2
import numpy as np

UBIT = 'stutishu'
np.random.seed(sum(map(ord, UBIT)))

def ConvertImageDataIntoPixelData(imageData):
    imageHeight = len(imageData)
    if(imageHeight is 0):
        return
    imageWidth = len(imageData[0])
    if(imageWidth is 0):
        return
    PixelData = list()
    for height in range(imageHeight):
        for width in range(imageWidth):
            PixelData.append(
                Pixel(imageData[height][width][0], imageData[height][width][1], imageData[height][width][2]))
    return PixelData

def ConvertPixelDataIntoImageData(pixeldata , imageWidth , imageHeight):
    imageData = [[[0,0,0] for width in range(imageWidth)] for height in range(imageHeight)]
    pixelcounter = 0
    for height in range(imageHeight):
        for width in range(imageWidth):
            imageData[height][width][0] = pixeldata[pixelcounter].Red
            imageData[height][width][1] = pixeldata[pixelcounter].Blue
            imageData[height][width][2] = pixeldata[pixelcounter].Green
            pixelcounter += 1
    
    return np.asarray(imageData)


class Pixel:
    def __init__(self,red,blue,green):
        self.Red = red
        self.Blue = blue
        self.Green = green

    def DistanceFrom(self,pixel):
        redDiff = math.pow(self.Blue - pixel.Red,2)
        blueDiff = math.pow(self.Blue - pixel.Blue,2)
        greenDiff = math.pow(self.Green - pixel.Green,2)
        return math.sqrt(redDiff + blueDiff + greenDiff)

class KmeanCluster:
    def __init__(self,Classes):
        self.NoOfClasses = Classes
        self.MeanValues = [ Pixel(np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)) for pixel in range(self.NoOfClasses)]
        #elf.MeanValues = list()
        self.Clusters = [[] for cluster in range(self.NoOfClasses)]

    def initializeMeans(self,means):
        self.MeanValues = means

    def classifyPixels(self,PixelData):
        for pixel in PixelData:
            DistanceFromMeanPixels = list()
            for mean in self.MeanValues:
                DistanceFromMeanPixels.append(pixel.DistanceFrom(mean))

            clusterIndex = DistanceFromMeanPixels.index(min(DistanceFromMeanPixels))
            self.Clusters[clusterIndex].append(pixel)
        return self.Clusters

    def returnClosetsPixel(self,pixel):
        DistanceFromMeanPixels = list()
        for mean in self.MeanValues:
            DistanceFromMeanPixels.append(pixel.DistanceFrom(mean))

        clusterIndex = DistanceFromMeanPixels.index(min(DistanceFromMeanPixels))
        return self.MeanValues[clusterIndex]

    def UpdatePixeldata(self,PixelData):
        updatedPixelData = list()
        for pixel in PixelData:
            updatedPixelData.append(self.returnClosetsPixel(pixel))
        return updatedPixelData


imageData = cv2.imread('/Users/stutishukla/Downloads/data/baboon.jpg', cv2.IMREAD_COLOR)
tempimageWidth= len(imageData)
tempimageHeight = len(imageData[0])
temp=len(imageData[0][0])

#Converting imageData into Pixels
PixelData = ConvertImageDataIntoPixelData(imageData)


#Initializing Cluster
clusterlist = [3,5,10,20]

for k in clusterlist:
    NoOfClusters = k
    cluster = KmeanCluster(NoOfClusters)
    updatedPixelData = cluster.UpdatePixeldata(PixelData)
    updatedImageData = ConvertPixelDataIntoImageData(updatedPixelData,tempimageWidth,tempimageHeight)
    cv2.imwrite('/Users/stutishukla/Downloads/task3_images/task3_baboon_K_new_'+str(NoOfClusters)+'.jpg',updatedImageData)
    print("Done for K :" + str(k))

#Save updatedImageData here
