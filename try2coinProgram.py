# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:32:18 2020

@author: renny
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:24:12 2020

@author: renny
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('estimateCoin.png')
cv.imshow("originalImg",img)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
cv.imshow("grayImg",gray)
cv.imshow("threshImg",thresh)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
cv.imshow("noiseremovedopeningImg",opening)
# sure background area
sure_bg = cv.dilate(thresh,kernel,iterations=5)
cv.imshow("sure_bgImg",sure_bg)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,3)
cv.imshow("dist_transformImg",dist_transform)
ret, sure_fg = cv.threshold(dist_transform,0.4*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
cv.imshow("sure_fgImg",sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
cv.imshow("unknownImg",unknown)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
cv.imshow("markersImg",markers)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv.watershed(img,markers)
#m = cv.convertScaleAbs(markers)
#m = cv.threshold(m, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

img[markers == -1] = [0,255,0]
#_, contours, _ = cv.findContours(img[markers == -1], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv.imshow("markersImg",img)
cv.waitKey(0)