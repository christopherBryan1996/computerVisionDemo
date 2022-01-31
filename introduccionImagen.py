# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 20:10:51 2022

@author: manci
"""
import numpy as np
import cv2 
import matplotlib.pyplot as plt
img =cv2.imread('imagenProceso/bananos.jpg')
img2='imagenProceso/bananos.jpg'
img3='https://www.hogarmania.com/archivos/202006/clasificacion-peces-dieta-1280x720x80xX.jpg'
img4=cv2.imread(img3)
img5 = cv2.imread('imagenProceso/clasificacion-peces-dieta-1280x720x80xX.jpg')

#solo toma las que allan pasado por cv2 
#cv2.imshow('banana', img4)
#cv2.waitKey(0)
#cv2.destroyALLWindows()

hvs=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hvs2=cv2.cvtColor(img5, cv2.COLOR_BGR2HSV)

I=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
I2=cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)

#cv2.imshow('banana',I2)

umbral=240
binari= np.uint8((I<umbral)*255)
binari2= np.uint8((I2<umbral)*255)

#dato=binari.flatten()
#plt.hist(dato,bins=100)
#plt.show()

rojo=img5[:,:,0].flatten()
verde=img5[:,:,1].flatten()
azul=img5[:,:,2].flatten()
plt.hist(rojo, bins=1000,histtype='stepfilled',color='red')
plt.hist(verde, bins=1000,histtype='stepfilled',color='green')
plt.hist(azul, bins=1000,histtype='stepfilled',color='blue')
plt.show()










