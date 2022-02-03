# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 20:10:51 2022

@author: manci
"""
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from scipy import ndimage
img =cv2.imread('imagenProceso/bananos.jpg')
img2='imagenProceso/bananos.jpg'
img3='https://www.hogarmania.com/archivos/202006/clasificacion-peces-dieta-1280x720x80xX.jpg'
img4=cv2.imread(img3)
img5 = cv2.imread('imagenProceso/clasificacion-peces-dieta-1280x720x80xX.jpg')
img6=cv2.imread('imagenProceso/000078.jpg')
#solo toma las que allan pasado por cv2 
#cv2.imshow('banana', img4)
#cv2.waitKey(0)
#cv2.destroyALLWindows()

hvs=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hvs2=cv2.cvtColor(img5, cv2.COLOR_BGR2HSV)

I=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
I2=cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
I3=cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
#cv2.imshow('banana',I2)

#para sacar un umbral mas acorde a la imagen 
umbral,_=cv2.threshold(I, 0, 255, cv2.THRESH_OTSU)
umbral2,_=cv2.threshold(I2, 0, 255, cv2.THRESH_OTSU)
umbral3,_=cv2.threshold(I3,0,255,cv2.THRESH_OTSU)

mascara= np.uint8((I<umbral)*255)
mascara2= np.uint8((I2<umbral2)*255)
mascara3=np.uint8((I3<umbral3)*255)


#dato=binari.flatten()
#plt.hist(dato,bins=100)
#plt.show()

output=cv2.connectedComponentsWithStats(mascara3,4,cv2.CV_32S)
cabtObj=output[0]
labels=output[1]
stats=output[2]
mascara3_1=(np.argmax(stats[:,4][1:])+1==labels)
mascara3_1=ndimage.binary_fill_holes(mascara3_1).astype(int)


rojo=img5[:,:,0].flatten()
verde=img5[:,:,1].flatten()
azul=img5[:,:,2].flatten()
plt.hist(rojo, bins=1000,histtype='stepfilled',color='red')
plt.hist(verde, bins=1000,histtype='stepfilled',color='green')
plt.hist(azul, bins=1000,histtype='stepfilled',color='blue')
plt.show()

cv2.imshow('imagen',mascara3)









