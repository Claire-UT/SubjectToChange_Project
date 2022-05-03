# -*- coding: utf-8 -*-
"""
Team 6- Subject to Change
Members- Laura Homiller, David Spitler, Claire Zhang
-------------
Model should have ~90% accuracy on images used for training, and ~80% accuracy on images used for testing

Letters Dataset from: https://www.nist.gov/itl/products-and-services/emnist-dataset
http://yann.lecun.com/exdb/mnist/
Math Dataset from: https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols

How to import EMNIST data: https://pypi.org/project/emnist/
https://github.com/rcgopi100/EMNISTHandwrittenLetters/blob/main/EMINST.ipynb

Model adapted from: https://www.kaggle.com/code/achintyatripathi/emnist-letter-dataset-97-9-acc-val-acc-91-78/notebook

INSTRUCTIONS:
    1. Use 'pip install emnist' to obtain the EMNIST data set (can take a while)
    2. Download the 'Math; dataset from the link above
        - Download dataset to the same location as this file.
        - Open folder '9' and delete the directory file.
    2. If you don't already have it, you may also need to install other packages such as tesnsorflow
    3. Run the file. The model will be saved as 'EMNIST Math CNN Model'
"""

from emnist import extract_training_samples
from emnist import extract_test_samples
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
import cv2

# IMPORTING THE EMNIST IMAGES ------------------------------------------------------------
train_images, train_labels = extract_training_samples('byclass')
test_images, test_labels = extract_test_samples('byclass')
print(f"Train Image Shape: {train_images.shape}, Train Label Shape: {train_labels.shape}")

# keeping only a quarter of the dataset because there is too much for my laptop to process.
train_images_number= int(1/4*814255)
train_images=train_images[0:train_images_number]
train_labels=train_labels[0:train_images_number]
test_images=test_images[0:train_images_number]
test_labels=test_labels[0:train_images_number]
print(f"Train Image Shape: {train_images.shape}, Train Label Shape: {train_labels.shape}")

# IMPORTING THE MATH IMAGES ------------------------------------------------------------
#each image is stored in a folder titled as its label. Manually going into each folder and creating the corresponding label for the image.

imgs = []
labels=[]
#dim is the number of pixels for the image to resize to. so the final dimensions will be (dimxdim)
dim=28
    
path = "dataset/add"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('62')
    
path = "dataset/dec"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('63')
    
path = "dataset/div"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('64')
    
path = "dataset/eq"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('65')
    
path = "dataset/mul"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('66')
    
path = "dataset/sub"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('67')

MATH_train_images, MATH_test_images, MATH_train_labels,MATH_test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=42)
MATH_train_images=np.asarray(MATH_train_images)
MATH_train_labels=np.asarray(MATH_train_labels)
MATH_test_images=np.asarray(MATH_test_images)
MATH_test_labels=np.asarray(MATH_test_labels)
print(f"Train Image Shape: {train_images.shape}, Train Label Shape: {train_labels.shape}")
print(f"Math Train Image Shape: {MATH_train_images.shape}, Math Train Label Shape: {MATH_train_labels.shape}")

# appending math data set to EMNIST dataset
train_labels= np.append(train_labels, MATH_train_labels)
test_labels= np.append(test_labels, MATH_test_labels)
test_images=np.concatenate((test_images, MATH_test_images),axis=0)
train_images=np.concatenate((train_images, MATH_train_images),axis=0)

train_images=np.asarray(train_images)
train_labels=np.asarray(train_labels)
test_images=np.asarray(test_images)
test_labels=np.asarray(test_labels)

print(f"Train Image Shape: {train_images.shape}, Train Label Shape: {train_labels.shape}")
#---- visualize data
fig,axes = plt.subplots(3,5,figsize=(10,8))
for i,ax in enumerate(axes.flat):
    ax.imshow(train_images[i])
print(f"Train Image Shape: {train_images.shape}, Train Label Shape: {train_labels.shape}")

#----- Normalise and reshape data
# Normalising wrt to RGB, which has 255 values.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Tells number of images, because train_images.shape returns row, col. [0] gives row length
train_images_number = train_images.shape[0]
# Formatting train and test data into correct shape for the model
train_images_height = 28
train_images_width = 28
train_images_size = train_images_height*train_images_width
train_images = train_images.reshape(train_images_number, train_images_height, train_images_width, 1)

test_images_number = test_images.shape[0]
test_images_height = 28
test_images_width = 28
test_images_size = test_images_height*test_images_width
test_images = test_images.reshape(test_images_number, test_images_height, test_images_width, 1)
print(f"Train Image Shape: {train_images.shape}, Train Label Shape: {train_labels.shape}")

# Number of classes to sort into. 26 lower case letters + 26 upper case letters + 10 digits = 62 classes
number_of_classes = 68

# What is does to_categorical do: https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
# Converts a class vector (integers) to binary class matrix.
y1 = tf.keras.utils.to_categorical(train_labels, number_of_classes)
y2 = tf.keras.utils.to_categorical(test_labels, number_of_classes)

# creating a CNN ((Convolutional Neural Network))
train_x,test_x,train_y,test_y = train_test_split(train_images,y1,test_size=0.7,random_state = 42)
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
MCP = ModelCheckpoint('Best_points_EMNISTMath.h5',verbose=1,save_best_only=True,monitor='val_accuracy',mode='max')
ES = EarlyStopping(monitor='val_accuracy',min_delta=0,verbose=0,restore_best_weights = True,patience=3,mode='max')
RLP = ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.2,min_lr=0.0001)

history = model.fit(train_x,train_y,epochs=10,validation_data=(test_x,test_y),callbacks=[MCP,ES,RLP])

# Save model, and then load into file where it is integrated into vision
# Reference: https://www.tensorflow.org/guide/keras/save_and_serialize
model.save('EMNIST Math CNN Model')