# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 22:43:25 2022

@author: manci
"""
import cv2 as cv
import numpy as np 

img=cv.imread('imagenProceso/bananos.jpg')

cv.imshow('original',img)

kernel_3x3=np.ones((3,3),np.float32)/(3*3)
#le ponemos el menos 1para que tenga el mismo tamaño que la imgen
output= cv.filter2D(img,-1,kernel_3x3)
cv.imshow('promedio 3x3',output)

kernel_5x5=np.ones((5,5),np.float32)/(5*5)
#le ponemos el menos 1para que tenga el mismo tamaño que la imgen
output= cv.filter2D(img,-1,kernel_5x5)
cv.imshow('promedio 5x5',output)

output= cv.GaussianBlur(img, (3,3), 0)
cv.imshow('Gauss dev= 3*3', output)

output= cv.GaussianBlur(img, (11,11), 0)
cv.imshow('Gauss dev= 11*11', output)

output= cv.GaussianBlur(img, (111,111), 0)
cv.imshow('Gauss dev= 111*111', output)