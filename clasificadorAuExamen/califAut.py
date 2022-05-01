# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 21:17:48 2022

@author: manci
"""

import cv2
import numpy as np
#tomamos la imagen
img=cv2.imread('foto_formato.jpg',0)
#detectamos todos los bordes de la imagen
canny = cv2.Canny(img,20,150)
#tamaño de dilatacion
kernerl=np.ones((5,5),np.uint8)
#dilatamos los bordes
bordes=cv2.dilate(canny,kernerl)
#cv2.RETR_TREE nos ayuda a clasificar los objetos optenidos 
#cv2.CHAIN_APPROX_SIMPLE dibuja el contorno
contour,_=cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

objetos= bordes.copy()
#tomamos los bordes grades y los rellenamos 
cv2.drawContours(objetos, [max(contour,key=cv2.contourArea)], -1, 255,thickness=-1)

output =cv2.connectedComponentsWithStats(objetos,4,cv2.CV_32S)

numObj=output[0]
label=output[1]
stats=output[2]
#tomamos el objeto de nuestro interes 
mascara=np.uint8(255*(np.argmax(stats[:,4][1:])+1==label))
#lo alinearemos para sacar bien las vertices 
contours,_=cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt=contours[0]

hull=cv2.convexHull(cnt)
puntosConvex=hull[:,0,:]
#m filas n columnas
m,n=mascara.shape
ar=np.zeros((m,n))
mascaraConvex= np.uint8(cv2.fillConvexPoly(ar,puntosConvex,1))

#para tomar las verices de la imagen automaticamente

vertices=cv2.goodFeaturesToTrack(mascaraConvex, 4, 0.01, 26)


x=vertices[:,0,0]
y=vertices[:,0,1]

vertices=vertices[:,0,:]

#ordenamos
xo=np.sort(x)
yo=np.sort(y)

xn=np.zeros([1,4])
yn=np.zeros([1,4])

xn=(x==xo[2])*n+(x==xo[3])*n
yn=(y==yo[2])*m+(y==yo[3])*m

verticesN=np.zeros([4,2])
verticesN[:,0]=xn
verticesN[:,1]=yn

vertices=np.int64(vertices)
verticesN=np.int64(verticesN)


h,_=cv2.findHomography(vertices, verticesN)

im2=cv2.warpPerspective(img, h,(n,m))

#creamos imagen de interes
roi=im2[:,np.uint64(0.25*n):np.uint64(0.84*n)]
opciones=['A','B','C','D','E']
respuestas=[]

for i in range(0,26):
    #para visualizar 
    #pregunta.append(roi[np.uint64(i*(m/26)):np.uint64((i+1)*(m/26)),:])
    #trabajaremos sobre cada pregunta
    pregunta=roi[np.uint64(i*(m/26)):np.uint64((i+1)*(m/26)),:]
    
    sumI=[]
    for j in range(0,5):
        _,col=pregunta.shape
        sumI.append(np.sum(pregunta[:,np.uint64(j*(col/5)):np.uint64((j+1)*(col*5))]))
    print(sumI)
    respuestas.append(opciones[np.argmin(sumI)])



