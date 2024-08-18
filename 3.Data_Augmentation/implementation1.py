#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 10:28:21 2024

@author: ankit
"""

# Data Augmentation would be helpful if we have 2000 images and want to generate 5000 images but
# if we have less than 100 or 200 images then we should have to use traditional machine learning

"""
Image shifts via the width_shift_range and height_shift_range arguments.
Image flips via the horizontal_flip and vertical_flip arguments.
Image rotations via the rotation_range argument
Image brightness via the brightness_range argument.
Image zoom via the zoom_range argument.
"""

from keras.preprocessing.image import ImageDataGenerator
from skimage import io

# Construct an instance of the ImageDataGenerator class
# Pass the augmentation parameters through the constructor. 

datagen = ImageDataGenerator(
        rotation_range=45,     #Random rotation between 0 and 45 degrees
        width_shift_range=0.2,   #% shift 20%
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='wrap', cval=125)    #Also try nearest, constant, reflect, wrap

# fill mode='constant' means when we shift the image to new place then old place got black,
# so we have to fill those pixels, we can fill it with gray pixels with value as cval=125 

######################################################################
#Loading a single image for demonstration purposes.
#Using flow method to augment the image

# Loading a sample image  
#Can use any library to read images but they need to be in an array form
#If using keras load_img convert it to an array first
# x = io.imread('3.Data_Augmentation/images/train_images/6.png')  #Array with shape (1059, 1024, 3)

# # Reshape the input image because ...
# #x: Input data to datagen.flow must be Numpy array of rank 4 or a tuple.
# #First element represents the number of images
# x = x.reshape((1, ) + x.shape)  #Array with shape (1, 256, 256, 3)

# i = 0
# for batch in datagen.flow(x, batch_size=16,  
#                           save_to_dir='3.Data_Augmentation/images/augmented/', 
#                           save_prefix='aug', 
#                           save_format='png'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely  


####################################################################
#Multiple images.
#Manually read each image and create an array to be supplied to datagen via flow method
dataset = []

import numpy as np
from skimage import io
import os
from PIL import Image

image_directory = '3.Data_Augmentation/images/train_images/'
SIZE = 128
dataset = []

my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    if (image_name.split('.')[1] == 'png'):
        image = io.imread(image_directory + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE,SIZE))
        dataset.append(np.array(image))

x = np.array(dataset)

#Let us save images to get a feel for the augmented images.
#Create an iterator either by using image dataset in memory (using flow() function)
#or by using image dataset from a directory (using flow_from_directory)
#from directory can beuseful if subdirectories are organized by class
   
# Generating and saving 10 augmented samples  
# using the above defined parameters.  
#Again, flow generates batches of randomly augmented images
   
i = 0
for batch in datagen.flow(x, batch_size=16,  
                          save_to_dir='3.Data_Augmentation/images/augmented_train_images/', 
                          save_prefix='aug', 
                          save_format='png'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely  

##########################


image_directory = '3.Data_Augmentation/images/train_masks/'
SIZE = 128
dataset = []

my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    if (image_name.split('.')[1] == 'png'):
        image = io.imread(image_directory + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE,SIZE))
        dataset.append(np.array(image))

x = np.array(dataset)

#Let us save images to get a feel for the augmented images.
#Create an iterator either by using image dataset in memory (using flow() function)
#or by using image dataset from a directory (using flow_from_directory)
#from directory can beuseful if subdirectories are organized by class
   
# Generating and saving 10 augmented samples  
# using the above defined parameters.  
#Again, flow generates batches of randomly augmented images
   
i = 0
for batch in datagen.flow(x, batch_size=16,  
                          save_to_dir='3.Data_Augmentation/images/augmented_train_masks/', 
                          save_prefix='aug', 
                          save_format='png'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely  



#####################################################################
#Multiclass. Read dirctly from the folder structure using flow_from_directory

# i = 0
# for batch in datagen.flow_from_directory(directory='monalisa_einstein/', 
#                                          batch_size=16,  
#                                          target_size=(256, 256),
#                                          color_mode="rgb",
#                                          save_to_dir='augmented', 
#                                          save_prefix='aug', 
#                                          save_format='png'):
#     i += 1
#     if i > 31:
#         break 

# #Creates 32 images for each class. 
        
# #Once data is augmented, you can use it to fit a model via: fit.generator
# #instead of fit()
# #model = 
# #fit model on augmented data
# #model.fit_generator(datagen.flow(x))