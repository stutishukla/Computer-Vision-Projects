#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 20:39:49 2018

@author: stutishukla
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:22:50 2018

@author: stutishukla
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
print(cv2.__version__)

img1 = cv2.imread('/Users/stutishukla/Downloads/data/tsucuba_left.png',cv2.IMREAD_COLOR)          # queryImage
img2 = cv2.imread('/Users/stutishukla/Downloads/data/tsucuba_right.png',cv2.IMREAD_COLOR)          # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

img1=cv2.drawKeypoints(img1, kp1, None)
img2=cv2.drawKeypoints(img2, kp2, None)


#img1=np.asarray(img1)
cv2.imwrite('/Users/stutishukla/Downloads/task2_images/task1_sift1.jpg',img1)
cv2.imwrite('/Users/stutishukla/Downloads/task2_images/task1_sift2.jpg',img2)

