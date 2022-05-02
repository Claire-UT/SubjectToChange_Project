# -*- coding: utf-8 -*-
"""
Team 6- Subject to Change
Members- Laura Homiller, David Spitler, Claire Zhang
-------------
Model should have ~100% accuracy on images used for training, and ~90% accuracy on images used for testing
Seems to vary a lot run to run

Dataset from: https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols

Model adapted from: https://www.kaggle.com/code/achintyatripathi/emnist-letter-dataset-97-9-acc-val-acc-91-78/notebook

INSTRUCTIONS:
    1. If you download the dataset from the link:
        - Download dataset to the same location as this file.
        - Open folder '9' and delete the directory file.
    2. If you don't already have some of the packages, you may also need to install other packages such as tesnsorflow
    3. Run file
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
import cv2


imgs = []
labels=[]
#dim is the number of pixels for the image to resize to. so the final dimensions will be (dimxdim)
dim=28

# IMPORTING THE IMAGES ------------------------------------------------------------
#each image is stored in a folder titled as its label. Manually going into each folder and creating the corresponding label for the image.

path = "dataset/0"
# converts each image into an np.array, and adds it to the imgs np.array. also creates a label
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('0')

path = "dataset/1"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('1')

path = "dataset/2"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('2')

path = "dataset/3"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('3')
    
path = "dataset/4"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('4')
    
path = "dataset/5"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('5')

path = "dataset/6"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('6')
    
path = "dataset/7"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('7')
    
path = "dataset/8"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('8')
    
path = "dataset/9"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('9')
    
path = "dataset/add"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('10')
    
path = "dataset/dec"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('11')
    
path = "dataset/div"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('12')
    
path = "dataset/eq"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('13')
    
path = "dataset/mul"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('14')
    
path = "dataset/sub"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('15')
    
path = "dataset/x"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('16')
    
path = "dataset/y"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('17')
    
path = "dataset/z"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('18')

# USING CNN FROM THE EMNIST MODEL EXAMPLE ----------------------------------------------------------------------

# splitting dataset into train and test dataset
train_images, test_images, train_labels,test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=42)
train_images=np.asarray(train_images)
train_labels=np.asarray(train_labels)
test_images=np.asarray(test_images)
test_labels=np.asarray(test_labels)
print(f"Train Image Shape: {train_images.shape}, Train Label Shape: {train_labels.shape}")

# scaling wrt 255
train_images=train_images/255
test_images=test_images/255

# Formatting train and test data into correct shape for the model
numberImages=np.size(train_images, 0)
train_images_height = 28
train_images_width = 28
train_images= train_images.reshape(numberImages, train_images_height, train_images_width, 1)

test_images_number = test_images.shape[0]
test_images_height = 28
test_images_width = 28
test_images_size = test_images_height*test_images_width
test_images = test_images.reshape(test_images_number, test_images_height, test_images_width, 1)
print(f"Train Image Shape: {train_images.shape}, Train Label Shape: {train_labels.shape}")

# Number of classes to sort into. 6 math symbols + 3 variables + 10 digits = 19 classes
number_of_classes = 19

# What is does to_categorical do: https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
# Converts a class vector (integers) to binary class matrix.
y1 = tf.keras.utils.to_categorical(train_labels, number_of_classes)
y2 = tf.keras.utils.to_categorical(test_labels, number_of_classes)

# creating a CNN ((Convolutional Neural Network))
train_x,test_x,train_y,test_y = train_test_split(train_images,y1,test_size=0.2,random_state = 42)
model = tf.keras.Sequential([ 
    tf.keras.layers.Conv2D(32,3,input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(number_of_classes,activation='softmax')
])
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#Adding Callback API's to save best weights
MCP = ModelCheckpoint('Best_points_Math.h5',verbose=1,save_best_only=True,monitor='val_accuracy',mode='max')
ES = EarlyStopping(monitor='val_accuracy',min_delta=0,verbose=0,restore_best_weights = True,patience=3,mode='max')
RLP = ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.2,min_lr=0.0001)

history = model.fit(train_x,train_y,epochs=10,validation_data=(test_x,test_y),callbacks=[MCP,ES,RLP])

# Save model, and then load into file where it is integrated into vision
# Reference: https://www.tensorflow.org/guide/keras/save_and_serialize
model.save('Math CNN Model')




