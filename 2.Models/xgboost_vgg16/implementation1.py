#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 15:25:42 2024

@author: ankit
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import cv2
import pickle

from keras.models import Model
#from keras.layers import Conv2D
import os
from keras.applications.vgg16 import VGG16


print(os.listdir("2.Models/SegmentationModels/images/"))

#Resizing images is optional, CNNs are ok with large images
SIZE_X = 128 #Resize images (height  = X, width = Y)
SIZE_Y = 128

#Capture training image info as a list
train_images = []

for directory_path in glob.glob("2.Models/xgboost_vgg16//images/train_images/"):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        #train_labels.append(label)
#Convert list to array for machine learning processing        
train_images = np.array(train_images)

#Capture mask/label info as a list
train_masks = [] 
for directory_path in glob.glob("2.Models/xgboost_vgg16//images/train_masks/"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X),interpolation=cv2.INTER_NEAREST)
        #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        train_masks.append(mask)
        #train_labels.append(label)
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)

#Use customary x_train and y_train variables
X_train = train_images
y_train = train_masks
y_train = np.expand_dims(y_train, axis=3) #May not be necessary.. leftover from previous code 


#Load VGG16 model wothout classifier/fully connected layers
#Load imagenet weights that we are going to use as feature generators
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE_X, SIZE_Y, 3))

#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()  #Trainable parameters will be 0

#After the first 2 convolutional layers the image dimension changes. 
#So for easy comparison to Y (labels) let us only take first 2 conv layers
#and create a new model to extract features
#New model with only first 2 conv layers
new_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)
new_model.summary()

#Now, let us apply feature extractor to our training data
features=new_model.predict(X_train)

#Plot features to view them
square = 8
ix=1
for _ in range(square):
    for _ in range(square):
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(features[0,:,:,ix-1], cmap='gray')
        ix +=1
plt.show()

#Reassign 'features' as X to make it easy to follow
X=features
X = X.reshape(-1, X.shape[3])  #Make it compatible for Random Forest and match Y labels

#Reshape Y to match X
Y = y_train.reshape(-1)

#Combine X and Y into a dataframe to make it easy to drop all rows with Y values 0
#In our labels Y values 0 = unlabeled pixels. 
dataset = pd.DataFrame(X)
dataset['Label'] = Y
print(dataset['Label'].unique())
print(dataset['Label'].value_counts())

##If we do not want to include pixels with value 0 
##e.g. Sometimes unlabeled pixels may be given a value 0.
dataset = dataset[dataset['Label'] != 0]

def change_lable(x):
    if x==38:
        return 0
    elif x==75:
        return 1
    else:
        return 2

dataset['Label']=dataset['Label'].map(lambda x:change_lable(x))

print(dataset['Label'].unique())
print(dataset['Label'].value_counts())


#Redefine X and Y for Random Forest
X_for_training = dataset.drop(labels = ['Label'], axis=1)
X_for_training = X_for_training.values  #Convert to array
Y_for_training = dataset['Label']
Y_for_training = Y_for_training.values  #Convert to array

#Note: You can work with pandas dataframes instead of arrays by please beware that
#xboost drops columns with zero values which creates a mismatch between column names
#for training and future testing datasets. So it is safe to work with arrays. 

#RANDOM FOREST, if interested in using it instead of xgboost. 
#from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(n_estimators = 10, random_state = 42)

#XGBOOST
import xgboost as xgb
model = xgb.XGBClassifier()

# Train the model on training data
model.fit(X_for_training, Y_for_training) 

#Save model for future use
filename = 'model_XG.sav'
pickle.dump(model, open(filename, 'wb'))

################################################################
#Start segmenting future images

#Load model.... 
loaded_model = pickle.load(open(filename, 'rb'))

#Test on a different image
#READ EXTERNAL IMAGE...
test_img = cv2.imread('2.Models/xgboost_vgg16/images/test_images/S12.png', cv2.IMREAD_COLOR)       
test_img = cv2.resize(test_img, (SIZE_Y, SIZE_X))
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
test_img = np.expand_dims(test_img, axis=0)

X_test_feature = new_model.predict(test_img)
X_test_feature = X_test_feature.reshape(-1, X_test_feature.shape[3])
prediction = loaded_model.predict(X_test_feature)

#View and Save segmented image
prediction_image = prediction.reshape(mask.shape)
plt.imshow(prediction_image, cmap='jet')
plt.imsave('2.Models/xgboost_vgg16/images/segmented/predicted_image.jpg', prediction_image, cmap='gray')

#########################################################
#Check accuracy and IoU
######################################################
#READ EXTERNAL IMAGE...
validation_img = cv2.imread('2.Models/xgboost_vgg16/images/test_images/S12.png', cv2.IMREAD_COLOR)       
validation_img = cv2.resize(validation_img, (SIZE_Y, SIZE_X))
validation_img = cv2.cvtColor(validation_img, cv2.COLOR_RGB2BGR)
validation_img = np.expand_dims(validation_img, axis=0)

X_validation_feature = new_model.predict(validation_img)
X_validation_feature = X_validation_feature.reshape(-1, X_validation_feature.shape[3])
prediction_validation = loaded_model.predict(X_validation_feature)

#Load ground truth image (Mask)
truth = cv2.imread('2.Models/xgboost_vgg16/images/test_masks/S12.png', 0).reshape(-1)

##########################################
#Pixel accuracy - not a good metric for semantic segmentation
#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(truth, prediction_validation))

##############################################

#IOU for each class is..
# IOU = true_positive / (true_positive + false_positive + false_negative).

#Using built in keras function
from keras.metrics import MeanIoU
num_classes = 5
IOU_keras = MeanIoU(num_classes=num_classes)  
IOU_keras.update_state(truth, prediction_validation)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(num_classes, num_classes)
print(values)
class1_IoU = values[1,1]/(values[1,1] + values[1,2] + values[1,3] + values[1,4] + values[2,1]+ values[3,1]+ values[4,1])
class2_IoU = values[2,2]/(values[2,2] + values[2,1] + values[2,1] + values[2,3] + values[1,2]+ values[3,2]+ values[4,2])
class3_IoU = values[3,3]/(values[3,3] + values[3,1] + values[3,2] + values[3,4] + values[1,3]+ values[2,3]+ values[4,3])
class4_IoU = values[4,4]/(values[4,4] + values[4,1] + values[4,2] + values[4,3] + values[1,4]+ values[2,4]+ values[3,4])

print("IoU Class 1 =", class1_IoU)
print("IoU Class 2 =", class2_IoU)
print("IoU Class 3 =", class3_IoU)
print("IoU Class 4 =", class4_IoU)

# Change hyperparameters to fine tune the model and verify IoU.



