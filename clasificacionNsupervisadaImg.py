# -*- coding: utf-8 -*-
"""
Created on Sun May  1 19:55:54 2022

@author: manci
"""

import cv2
import numpy as np
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

imagen='clasificacion (banano - manzana)/banano25.jpg'

img= cv2.imread(imagen)

I=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

umbral,_=cv2.threshold(I, 0, 255, cv2.THRESH_OTSU)

mascar=np.uint8((I<umbral)*255)

output=cv2.connectedComponentsWithStats(mascar, 4,cv2.CV_32S)
cantObj=output[0]
labels=output[1]
stats=output[2]

mascar=(np.argmax(stats[:,4][1:])+1==labels)

mascar=ndimage.binary_fill_holes(mascar).astype(int)

rojo=np.sum(mascar*img[:,:,0]/255)/np.sum(mascar)
verde=np.sum(mascar*img[:,:,1]/255)/np.sum(mascar)

mascar1=np.uint8(mascar*255)

im2,contours,hierarchy=cv2.findContours(mascar, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt=contours[0]
rect=cv2.minAreaRect(cnt)
box= cv2.boxPoints(rect)
box= np.uint8(box)
m,n=mascar1.shape