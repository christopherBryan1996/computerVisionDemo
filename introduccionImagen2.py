# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 20:32:11 2022

@author: manci
"""

import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from scipy import ndimage
#tomamos la imagen y la pasamos cv2
img= cv2.imread('imagenProceso/000078.jpg')
#trasformamos la imagen en hsv
hvs = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#pasamos la imagen a escala de grises
I=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#sacaremos el umbrar con el algoritmo otsu
#  imagen 0= negro 255=blanco
umbral,_=cv2.threshold(I, 0, 255, cv2.THRESH_OTSU)

mascara=np.uint8((I<umbral)*255)

output =cv2.connectedComponentsWithStats(mascara,4,cv2.CV_32S)
cantObj=output[0]
label=output[1]
stats=output[2]

mascara=(np.argmax(stats[:,4][1:])+1==label)
mascara=ndimage.binary_fill_holes(mascara).astype(int)

mascara1= np.uint8(mascara*255)

contours,_=cv2.findContours(mascara1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt=contours[0]

P=cv2.arcLength(cnt, True)
A=cv2.contourArea(cnt)
#A1=np.sum(mascara1/255)

#convex hull 
hull=cv2.convexHull(cnt)
puntosCovex=hull[:,0,:]
m,n=mascara1.shape
ar=np.zeros((m,n))
mascaraConvex=cv2.fillConvexPoly(ar, puntosCovex, 1)

#Bounding box rotado
rect=cv2.minAreaRect(cnt)
box=np.int0(cv2.boxPoints(rect))

m,n=mascara1.shape
ar=np.zeros((m,n))
mascaraReact=cv2.fillConvexPoly(ar,box,1)

#el menos uno es para mostrar todos los obejos si es que hay mas de uno
contours,_=cv2.findContours(mascara1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1,(0,0,255),1)

dato=I.flatten()

rojo=img[:,:,0].flatten()
verde=img[:,:,1].flatten()
azul=img[:,:,2].flatten()
plt.hist(rojo, bins=1000,histtype='stepfilled',color='red')
plt.hist(verde, bins=1000,histtype='stepfilled',color='green')
plt.hist(azul, bins=1000,histtype='stepfilled',color='blue')
plt.show()

segColor=np.zeros((m,n,3)).astype('uint8')
segColor[:,:,0]=np.uint8(img[:,:,0]*mascara)
segColor[:,:,1]=np.uint8(img[:,:,1]*mascara)
segColor[:,:,2]=np.uint8(img[:,:,2]*mascara)

segGrey=np.zeros((m,n))
segGrey[:,:]=np.uint8(I*mascara)

cv2.imshow('img',segColor)