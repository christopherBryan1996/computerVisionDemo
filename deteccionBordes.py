# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 20:21:19 2022

@author: manci
"""

import cv2 as cv
import numpy as np

a = np.zeros((100,50))
b= np.ones((100,50))
#axis=1 es a columnas 
#img=np.uint8(255*np.concatenate((a,b),axis=1))
#si ponemos 0 ya lo convierte a escala de grises
img=cv.imread('imagenProceso/clasificacion-peces-dieta-1280x720x80xX.jpg',0)

gx=cv.Sobel(img, cv.CV_64F, 1, 0,5)
gy=cv.Sobel(img, cv.CV_64F, 0, 1,5)
#ang nos da valores de 0 a pi
mag,ang=cv.cartToPolar(gx, gy)

#les ponemos valores de 0 a 255
gx=cv.convertScaleAbs(gx)
gy=cv.convertScaleAbs(gy)
mag=cv.convertScaleAbs(mag)

#vamos a convetir radiales a grados
ang=(180/np.pi)*ang

imgFill=cv.GaussianBlur(img, (5,5), 0)
lap=cv.convertScaleAbs(cv.Laplacian(imgFill, cv.CV_64F,5))

#                  umbral min, umbral maximo
canny=cv.Canny(img,55,150)

cv.imshow('gx',gx)
cv.imshow('gy',gy)
cv.imshow('mag',mag)
cv.imshow('Lap',lap)
cv.imshow('canny',canny)
