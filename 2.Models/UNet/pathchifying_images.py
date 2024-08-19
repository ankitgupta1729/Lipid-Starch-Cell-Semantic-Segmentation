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
import glob
import os

image_dir="images/train_images/"
mask_dir="images/train_masks/"
patch_dir_train="images/patches_128/train_images/"
patch_dir_mask="images/patches_128/train_masks/"

for directory_path in glob.glob(image_dir):
    for image_path in glob.glob(os.path.join(directory_path, "*.png")):
        image_name=image_path.split('/')[-1].split('.')[0]
        img = cv2.imread(image_path)
        patches_img = patchify(img, (128,128,3), step=128)
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, 0, :, :, :]
                if not cv2.imwrite(patch_dir_train+'pathched128_'+image_name+ '_'+ str(i)+str(j)+'.png', single_patch_img):
                    raise Exception("Could not write the image")

for directory_path in glob.glob(mask_dir):
    for image_path in glob.glob(os.path.join(directory_path, "*.png")):
        image_name=image_path.split('/')[-1].split('.')[0]
        img = cv2.imread(image_path)
        patches_img = patchify(img, (128,128,3), step=128)
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, 0, :, :, :]
                if not cv2.imwrite(patch_dir_mask +'pathched128_'+image_name+ '_'+ str(i)+str(j)+'.png', single_patch_img):
                    raise Exception("Could not write the image")
