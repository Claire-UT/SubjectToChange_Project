# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 09:55:20 2022

@author: clair
"""
#Data From: https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols
#Model Adapted From EMNIST Model Ref: https://www.kaggle.com/code/achintyatripathi/emnist-letter-dataset-97-9-acc-val-acc-91-78/notebook
#And from: https://www.kaggle.com/code/sagyamthapa/my-code?scriptVersionId=57999577


import tensorflow as tf
#for the CNN
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Dense,Dropout,MaxPooling2D,Conv2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
import cv2


imgs = []
labels=[]

# IMPORTING THE IMAGES ------------------------------------------------------------
#each image is stored in a folder titled as its label. Manually going into each folder and creating the corresponding label for the image.
#imgs.append(np.array(Image.open(os.path.join(path,f))))

path = "dataset/0"
# converts each image into an np.array, and adds it to the imgs np.array. also creates a label
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('0')

path = "dataset/1"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('1')

path = "dataset/2"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('2')

path = "dataset/3"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('3')
    
path = "dataset/4"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('4')
    
path = "dataset/5"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('5')

path = "dataset/6"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('6')
    
path = "dataset/7"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('7')
    
path = "dataset/8"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('8')
    
path = "dataset/9"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('9')
    
path = "dataset/add"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('10')
    
path = "dataset/dec"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('11')
    
path = "dataset/div"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('12')
    
path = "dataset/eq"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('13')
    
path = "dataset/mul"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('14')
    
path = "dataset/sub"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('15')
    
path = "dataset/x"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('16')
    
path = "dataset/y"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('17')
    
path = "dataset/z"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('18')
    
# FORMATTING DATA SET FOR NEURAL NETWORK------------------------------------------------------------
print(imgs[0][0][0])
# splitting dataset into train and test dataset
train_images, test_images, train_labels,test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=42)
train_images=np.asarray(train_images)
train_labels=np.asarray(train_labels)
test_images=np.asarray(test_images)
test_labels=np.asarray(test_labels)
print(type(train_images))
print(f"Train Image Shape: {train_images.shape}, Train Label Shape: {train_labels.shape}")
# #visualize the data
# import matplotlib.pyplot as plt 
# fig,axes = plt.subplots(3,5,figsize=(10,8))
# for i,ax in enumerate(axes.flat):
#     ax.imshow(train_images[i])

# scaling wrt 255
# train_images=train_images/255
for image in train_images:
    for i in range(len(image)):
        image[i] = image[i] / 255
for image in train_images:
    for i in range(len(image)):
        image[i] = image[i] / 255
        
numberImages=np.size(train_images, 0)
# print("numberImages: " + str(numberImages))
train_images_height = 28
train_images_width = 28
train_images= train_images.reshape(numberImages, train_images_height, train_images_width, 1)
print(f"Train Image Shape: {train_images.shape}, Train Label Shape: {train_labels.shape}")

# Transform labels
number_of_classes = 19

# #https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
y1 = tf.keras.utils.to_categorical(train_labels, number_of_classes)
y2 = tf.keras.utils.to_categorical(test_labels, number_of_classes)

# creating a CNN ((Convolutional Neural Network))
train_x,test_x,train_y,test_y = train_test_split(train_images,y1,test_size=0.7,random_state = 42)
print('split into data')
model = tf.keras.Sequential([ 
    tf.keras.layers.Conv2D(32,3,input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(number_of_classes,activation='softmax')
])
print('build model')
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
print('done')

#Adding Callback API's to save best weights and change lr
MCP = ModelCheckpoint('Best_points_Math.h5',verbose=1,save_best_only=True,monitor='val_accuracy',mode='max')
ES = EarlyStopping(monitor='val_accuracy',min_delta=0,verbose=0,restore_best_weights = True,patience=3,mode='max')
RLP = ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.2,min_lr=0.0001)

history = model.fit(train_x,train_y,epochs=10,validation_data=(test_x,test_y),callbacks=[MCP,ES,RLP])
print('done 1')

# Save model, and then load into file where it is integrated into vision
# Reference: https://www.tensorflow.org/guide/keras/save_and_serialize
model.save('Math Model')


# ALTERNATIVE -------------------------------

# train_images_number = train_images.shape[0]
# # print(train_images_number)
# train_images_height = 28
# train_images_width = 28
# train_images_size = train_images_height*train_images_width

# train_images = train_images.reshape(train_images_number, train_images_height, train_images_width, 1)
# data_dir = pathlib.Path('dataset')

# #split data into train and test data
# batch_size = 32
# img_height = 100
# img_width = 100
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   color_mode="grayscale",
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   color_mode="grayscale",
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)


# val_ds=tfds.as_numpy(val_ds)

# print(val_ds)

# class_names = train_ds.class_names
# print('1....')
# train_np = np.stack(list(train_ds))
# test_np = np.stack(list(val_ds))
# print('2....')
# print(type(train_np), train_np.shape)
# print(type(test_np), test_np.shape)

# # print(list(train_ds.as_numpy_iterator()))
# print(type(test_np))
# print(f"Train Image Shape: {train_ds.shape}, Train Label Shape: {class_names.shape}")
# print(class_names)

# num_classes = 16
# img_channels = 3




