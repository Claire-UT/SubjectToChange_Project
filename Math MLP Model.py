# -*- coding: utf-8 -*-
"""
Team 6- Subject to Change
Members- Laura Homiller, David Spitler, Claire Zhang
-------------
Model should have ~30% accuracy

Dataset from: https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols

Model adapted from: https://thedatafrog.com/en/articles/handwritten-digit-recognition-scikit-learn/

INSTRUCTIONS:
    1. If you download the dataset from the link:
        - Download dataset to the same location as this file.
        - Open folder '9' and delete the directory file.
    2. If you don't already have some of the packages, you may also need to install other packages such as tesnsorflow, sklearn, pillow, pickle
    3. Run file
"""

from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
import cv2
from sklearn.neural_network import MLPClassifier #Create neural network for classifier
from sklearn.metrics import accuracy_score, classification_report #Find accuracy of model
import matplotlib.pyplot as plt
import pickle

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
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('0')

path = "dataset/1"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('1')

path = "dataset/2"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('2')

path = "dataset/3"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('3')
    
path = "dataset/4"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('4')
    
path = "dataset/5"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('5')

path = "dataset/6"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('6')
    
path = "dataset/7"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('7')
    
path = "dataset/8"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('8')
    
path = "dataset/9"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('9')
    
path = "dataset/add"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('10')
    
path = "dataset/dec"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('11')
    
path = "dataset/div"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('12')
    
path = "dataset/eq"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('13')
    
path = "dataset/mul"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('14')
    
path = "dataset/sub"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('15')
    
path = "dataset/x"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('16')
    
path = "dataset/y"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('17')
    
path = "dataset/z"
for f in os.listdir(path):
    image=Image.open(os.path.join(path,f))
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.bitwise_not(image)
    image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
    imgs.append(image2)
    labels.append('18')

# USING MLPClassifier, THE MODEL USED IN OUR LIGHTNING TALK -----------------------------------------------------------------
train_images, test_images, train_labels,test_labels = train_test_split(imgs, labels, test_size=0.50, random_state=42)
train_images=np.asarray(train_images)
train_labels=np.asarray(train_labels)
test_images=np.asarray(test_images)
test_labels=np.asarray(test_labels)

#Plot the first 16 digits with their associated target value
def plot_multi(i):
    '''Plots 16 digits, starting with digit i'''
    nplots = 16
    fig = plt.figure(figsize=(15,15))
    for j in range(nplots):
        plt.subplot(4,4,j+1)
        plt.imshow(train_images[i+j])
        plt.title(train_labels[i+j])
        plt.axis('off')
    plt.show()
plot_multi(0)

print(f"Train Image Shape: {train_images.shape}, Train Label Shape: {train_labels.shape}")

# scaling wrt 255
train_images = train_images / 255.0
test_images = test_images / 255.0

            
train_images = train_images.reshape((len(train_images), -1))
test_images = test_images.reshape((len(test_images), -1))
print(f"Train Image Shape: {train_images.shape}, Train Label Shape: {train_labels.shape}")

#Create neural network using Multi-layer Perceptron (MLP) classifier
# "This implementation works with data represented as dense numpy arrays" https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic', alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.1, verbose=True)

mlp.fit(train_images,train_labels) #Train the neural network
#Loss is average difference between actual and predicted value for the iteration

predictions = mlp.predict(test_images) #Test trained network on unused test data

#Display 4 random test samples and show their predicted digit value
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes,test_images, predictions):
    ax.set_axis_off()
    image = image.reshape(dim, dim)
    ax.imshow(image, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

#Output total accuracy of prediction vs actual value for all test data
accuracy = float(accuracy_score(test_labels, predictions))
accuracypercent = round((accuracy * 100),2)
print()
print('The model correctly predicted digits with '+ str(accuracypercent) +'% accuracy')
print(classification_report(test_labels,predictions))

# save model
with open('Math MLP Model.pkl','wb') as f:
    pickle.dump(mlp,f)

