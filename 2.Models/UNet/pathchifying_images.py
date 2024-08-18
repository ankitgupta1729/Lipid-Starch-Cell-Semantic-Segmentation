#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 21:50:26 2024

@author: ankit
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from patchify import patchify

large_image_stack = cv2.imread('2.Models/UNet/images/train_images/6.png')
large_mask_stack = cv2.imread('2.Models/UNet/images/train_masks/6.png')

img = cv2.imread("2.Models/UNet/images/train_images/6.png")
patches_img = patchify(img, (128,128,3), step=128)

for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        single_patch_img = patches_img[i, j, 0, :, :, :]
        if not cv2.imwrite('2.Models/UNet/images/patches_128/trained_images/' + 'image_' + '_'+ str(i)+str(j)+'.png', single_patch_img):
            raise Exception("Could not write the image")