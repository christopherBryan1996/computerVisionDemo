# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 22:05:40 2022

@author: manci
"""

import cv2 as cv
import numpy as np

img=cv.imread('imagenProceso/iphone.jpg') 

imG=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

canny=cv.Canny(imG,25,150)

#delitacion
kernel = np.ones((5,5),np.uint8 )
border=cv.dilate(canny, kernel)

contours,_=cv.findContours(border, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

objetos=border.copy()

cv.drawContours(objetos, [max(contours,key=cv.contourArea)], -1,255,thickness=-1)

objetos=objetos/255

seg=np.zeros(img.shape)

seg[:,:,0]=objetos*img[:,:,0]+255*(objetos==0)
seg[:,:,1]=objetos*img[:,:,1]+255*(objetos==0)
seg[:,:,2]=objetos*img[:,:,2]+255*(objetos==0)

seg=np.uint8(seg)

cv.imshow('original',img)
cv.imshow('segmetada',seg)