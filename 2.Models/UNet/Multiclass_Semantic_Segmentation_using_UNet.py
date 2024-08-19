#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 21:09:43 2024

@author: ankit
"""

from simple_multi_unet_model import multi_unet_model #Uses softmax 

from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from sklearn.preprocessing import LabelEncoder
import collections
import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.utils import class_weight
from keras.metrics import MeanIoU
import random

print("Current Working Directory:",os.getcwd())
#Resizing images, if needed
SIZE_X = 128
SIZE_Y = 128
n_classes=4 #Number of classes for segmentation

#Capture training image info as a list
train_images = []

for directory_path in glob.glob("images/train_images/"):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, 0)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)
       
#Convert list to array for machine learning processing        
train_images = np.array(train_images)

#Capture mask/label info as a list
train_masks = [] 
for directory_path in glob.glob("images/train_masks/"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        train_masks.append(mask)
        
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)



###############################################
#Encode labels... but multi dim array so need to flatten, encode and reshape

labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)


print(collections.Counter(train_masks.reshape(-1)))
print(collections.Counter(train_masks_reshaped_encoded))

# Labels: 0=0=background, 38=1=cell, 75=2=lipid(black), 113=3=starch(white)
# Counter({0: 1964502, 38: 1362116, 113: 210059, 75: 67803})
# Counter({0: 1964502, 1: 1362116, 3: 210059, 2: 67803})

#################################################
train_images = np.expand_dims(train_images, axis=3)
train_images = normalize(train_images, axis=1)

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

#Create a subset of data for quick testing
#Picking 10% for testing and remaining for training

X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0)

X_train=X1
y_train=y1
print(collections.Counter(y_train.reshape(-1)))
#Further split training data t a smaller subset for quick testing of models
#X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.2, random_state = 0)

print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 


train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))



test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

###############################################################

class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(train_masks_reshaped_encoded),y=train_masks_reshaped_encoded)
print("Class weights are...:", class_weights)

# class_weight={}
# class_weight[0]=class_weights[0]
# class_weight[1]=class_weights[1]
# class_weight[2]=class_weights[2]
# class_weight[3]=class_weights[3]

# print(class_weight)

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_weighted = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        focal_loss = alpha_weighted * tf.pow(tf.abs(y_true - y_pred), self.gamma)
        return tf.reduce_mean(focal_loss)
class TverskyLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.5, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        tversky_coeff = (tp + 1e-6) / (tp + self.alpha * fn + self.beta * fp + 1e-6)
        return 1 - tversky_coeff
class DiceLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)
        dice = numerator / (denominator + 1e-6)
        return 1 - dice
def dice_coef(y_true, y_pred, smooth=1):
    # flatten
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # one-hot encoding y with 3 labels : 0=background, 1=label1, 2=label2
    y_true_f = K.one_hot(K.cast(y_true_f, np.uint8), 3)
    y_pred_f = K.one_hot(K.cast(y_pred_f, np.uint8), 3)
    # calculate intersection and union exluding background using y[:,1:]
    intersection = K.sum(y_true_f[:,1:]* y_pred_f[:,1:], axis=[-1])
    union = K.sum(y_true_f[:,1:], axis=[-1]) + K.sum(y_pred_f[:,1:], axis=[-1])
    # apply dice formula
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def dice_loss(y_true, y_pred):
    return 1-dice_coef

def iou_metric(y_true, y_pred):
    aog = tf.abs(tf.transpose(y_true)[2] - tf.transpose(y_true)[0] + 1) * tf.abs(tf.transpose(y_true)[3] - tf.transpose(y_true)[1] + 1)
    aop = tf.abs(tf.transpose(y_pred)[2] - tf.transpose(y_pred)[0] + 1) * tf.abs(tf.transpose(y_pred)[3] - tf.transpose(y_pred)[1] + 1)
    overlap_0 = tf.maximum(tf.transpose(y_true)[0], tf.transpose(y_pred)[0])
    overlap_1 = tf.maximum(tf.transpose(y_true)[1], tf.transpose(y_pred)[1])
    overlap_2 = tf.minimum(tf.transpose(y_true)[2], tf.transpose(y_pred)[2])
    overlap_3 = tf.minimum(tf.transpose(y_true)[3], tf.transpose(y_pred)[3])
    intersection = (overlap_2 - overlap_0 + 1) * (overlap_3 - overlap_1 + 1)
    union = aog + aop - intersection
    iou = intersection / union
    iou = tf.keras.backend.clip(iou, 0.0 + tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
    return iou   

adam = optimizers.Adam(learning_rate=0.0001)


#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=adam, loss=[dice_loss],metrics=[dice_coef])
model.compile(optimizer=Adam(learning_rate=0.0001), loss=FocalLoss(gamma=2.0, alpha=0.25),metrics=['accuracy'])
#model.compile(optimizer=Adam(), loss=FocalLoss(gamma=2.0, alpha=0.25),metrics=[mean_iou])
#model.compile(optimizer=Adam(), loss=FocalLoss(gamma=2.0, alpha=0.25),metrics=[dice_coef])
#model.compile(optimizer=Adam(), loss=FocalLoss(gamma=2.0, alpha=0.25),metrics=[iou_metric])
model.summary()

# we can use loss as focal loss which is better used for imbalanced data like this
# for metrics, we can also use jaccord coeff etc

#If starting with pre-trained weights. 
#model.load_weights('???.hdf5')


len_y_train_cat = len(y_train_cat)

class_weights_mat = np.zeros((len_y_train_cat, n_classes))
class_weights_mat[:, 0] += class_weights[0]
class_weights_mat[:, 1] += class_weights[1]
class_weights_mat[:, 2] += class_weights[2]
class_weights_mat[:, 3] += class_weights[3]

history = model.fit(X_train, y_train_cat, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=500, 
                    validation_data=(X_test, y_test_cat), 
                    #sample_weight=class_weights_mat,
                    shuffle=False)



model.save('unet_50_epochs.hdf5')

#Evaluate the model
	# evaluate model
_, acc = model.evaluate(X_test, y_test_cat)
print("Accuracy is = ", (acc * 100.0), "%")


###
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

##################################
#model = get_model()
model.load_weights('unet_50_epochs.hdf5') 

#IOU
y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)


#Using built in keras function

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

plt.imshow(train_images[0, :,:,0], cmap='gray')
plt.show()
plt.imshow(train_masks[0], cmap='gray')
plt.show()
#######################################################################
#Predict on a few images
#model = get_model()
#model.load_weights('???.hdf5')  

test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()

#####################################################################
