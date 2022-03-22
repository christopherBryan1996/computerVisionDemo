# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:14:03 2022

@author: manci
"""

import cv2 as cv
import numpy as np

img= cv.imread('imagenProceso/cuadro.jpg')
#para usar una imagen de internet con url
# cap = cv.VideoCapture('http://photos1.blogger.com/blogger/7026/1810/1600/cuadrodelado.jpg')
# ret,img2=cap.read()

imG=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# img2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

m,n,_=img.shape
#m,n,_=img2.shape

pts_src= np.array([[41,12],[193,50],[40,304],[198,271]])

pts_dst= np.array([[0,0],[n,0],[0,m],[n,m]])

h,_=cv.findHomography(pts_src,pts_dst)

im2=cv.warpPerspective(img,h,(n,m))

cv.imshow('Original',img)
cv.imshow('Correccion prespectiva',im2)