#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:45:18 2024

@author: ankit
"""

# Gabor Filters can be used for edge detection, texture analysis and feature extraction etc.

# The Gabor filter, named after Dennis Gabor, is a linear filter.

# These filters have been shown to possess optimal localization properties in both 
# spatial and frequency domains and thus are well-suited for texture segmentation problems. 

# They are bandpass filters because they allow certain band of frequencies and reject other types.

# A Gabor filter can be viewed as a sinusoidal signal of particular frequency and orientation, 
# modulated by a Gaussian wave

# Consider an example of an elephant that has patterns or stripes on its skin at different orientations. 
# Now to highlight or extract out all those patterns we are going to use a bank of 16 Gabor filters 
# at an orientation of 11.250 (i.e. if the first filter is at 00, then the second will be at 11.250, 
# the third will be at 22.50, and so on.

# When the input image is convolved with all the Gabor filters the patterns are easily highlighted.

# When a Gabor filter is applied to an image, it gives the highest response at edges 
# and at points where texture changes. When we say that a filter responds to a particular feature, 
# we mean that the filter has a distinguishing value at the spatial location of that feature.

# 

# https://medium.com/@anuj_shah/through-the-eyes-of-gabor-filter-17d1fdb3ac97

# https://dsp.stackexchange.com/questions/14714/understanding-the-gabor-filter-function/14715#14715

# Think of gabor as a gaussian  function in 2D. It can be multidimensional too.   

# gabor is a function of various parameters:
    
# g(x,y,sigma,theta,lambda, gamma, phi) = exp[- (x^2+gamma^2*y^2)/(2*sigma^2)]*exp[i*(2*pi*x/lambda+phi)] 
  
 

# where
# x,y = size of kernel
# sigma = standard deviation
# theta = angle
# lambda = wavelength
# gamma = aspect ratio 
# phi = phase offset

# If gamma=1 then we have a kernel which looks similar to circular kernel
# if gamma is close to zero then kernel looks similar to elliptical in one direction or straight line

######################################################################################################


import numpy as np
import cv2
import matplotlib.pyplot as plt

ksize = 5 # change to 50
sigma = 3 # change to 30
theta = 1*np.pi/4
lamda = 1*np.pi/4
gamma = 0.5
phi = 0

kernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma,phi, ktype=cv2.CV_32F)
# CV_32F means once the kernel is generated, we store the numbers in float 32 representation

#plt.imshow(kernel)

# ksize depends on image size and feature size. If number of pixels are less then we use less ksize

# Gabor kernel are amazing for textures. If we have 2 different regions with very similar gray level 
# but we can't extract separate them. 

img = cv2.imread('images/original_images/6.png')
#img = cv2.imread('BSE_Image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)

kernel_resized = cv2.resize(kernel, (400, 400)) # Resize image
#cv2.imshow('Kernel', kernel_resized)
#cv2.imshow('Original Img.', img)
#cv2.imshow('Filtered', fimg)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
