# -*- coding: utf-8 -*-
"""
Created on Sun May  1 19:55:54 2022

@author: manci
"""

import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

imagen='clasificacion (banano - manzana)/manzana2.jpg'

img= cv2.imread(imagen)

I=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
