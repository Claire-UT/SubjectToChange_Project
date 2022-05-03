# -*- coding: utf-8 -*-
"""
Team 6- Subject to Change
Members- Laura Homiller, David Spitler, Claire Zhang

This file contains the functions used for predicting characters. 
A single image must be formatted as a numpy array of size 28x28 storing the grayscale value of each pixel.  
Each 'predict' function is capable of handling an 'image' parameter that is a 2d array (single image) or 3d array (multiple images)
    The EMNIST_model is trained by running 'EMNIST_Model.py'
    The math_CNN_model is trained by running 'Math CNN Model.py'
    The math_MLP_model is trained by running 'Math MLP Model.py'
    The EMNISTmath_model is trained by running 'EMNIST_Math_CNN Model.py'

This file also has 'test' functions that can be called to show the accuracy of each model on its own test data
To use the test functions, you must have the EMNIST dataset and 'Math' dataset.
To obtain the EMNIST dataset, use pip install emnist
To obtain the Math dataset, visit https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols
    -Download the dataset in the same location as this python file.
    -Open folder '9' and delete the directory file.
    
INSTRUCTIONS
    For using the 'predict' functions:
        1. Import the relevant function into your file.
        2. Call the function with correct parameters. 
                - 'image' is either a 2D array (28x28) or 3D array (?x28x28) storing the grayscale value of each pixel
                - '...model' pass in the correct model
                - 'images_height' = 28, unless trained on new dimensions
                - 'images_width' = 28, unless trained on new dimensions
        3. The function will return the predicted text as a list
        
    For using the test functions:
        1. Donwload the datasets
        2. Uncomment the DEMOING TESTS section at the end of this file
        3. Run the file
        4. The function will output the plot of images examined, the predicted characters, the actual characters, and the accuracy of the example.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import pickle

# PREDICTION FUNCTIONS -----------------------------------------------------------------------------------
def EMNIST_predict (image, EMNIST_model, images_height, images_width):
    mapping = {    
        # digits
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        
        # capital lettes
        10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 
        20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
        30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
        
        # lowercase letters
        36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k',
        47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v',
        58: 'w', 59: 'x', 60: 'y', 61: 'z',
    }

    #converting image to a np array, and coverting to float
    image=np.array(image)
    image = image.astype(float)
    
    #if converting only one character, need to manually add another dimension
    if image.ndim ==2:
        image=[image]
    
    #normalizing 3d array wrt to 255, so all grayscale values become 0 to 1.
    image=np.array(image)      
    image = image / 255.0
    
    # reshaping array to match shape required by the model
    numberImages=np.size(image, 0)
    image=np.array(image)
    image= image.reshape(numberImages, images_height, images_width, 1)
    
    # predicting value
    prediction=EMNIST_model.predict(image)
    converted = np.argmax(prediction, axis=-1)
    
    #associating the label to an alphanumeric character
    predicted_text=list(mapping[int(i)] for i in converted)
    # print(predicted_text)
    return predicted_text


def MathCNN_predict (image, math_CNN_model, images_height, images_width):
    mapping = {    
        # digits
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    
        # other characters
        10: '+', 11: '.', 12: '/', 13: '=', 14: '*', 15: '-', 16: 'x', 17: 'y', 18: 'z',
        }

    #converting image to a np array, and coverting to float
    image=np.array(image)
    image = image.astype(float)
    
    #if converting only one character, need to manually add another dimension
    if image.ndim ==2:
        image=[image]
    
    #normalizing 3d array wrt to 255, so all grayscale values become 0 to 1.
    image=np.array(image)      
    image = image / 255.0
    
    # reshaping array to match shape required by the model
    numberImages=np.size(image, 0)
    image=np.array(image)
    image= image.reshape(numberImages, images_height, images_width, 1)
    
    # predicting value
    prediction=math_CNN_model.predict(image)
    converted = np.argmax(prediction, axis=-1)
    
    #associating the label to an 'math' character
    predicted_text=list(mapping[int(i)] for i in converted)
    # print(predicted_text)
    return predicted_text


def MathMLP_predict (image, math_MLP_model, images_height, images_width):
    mapping = {    
        # digits
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    
        # other characters
        10: '+', 11: '.', 12: '/', 13: '=', 14: '*', 15: '-', 16: 'x', 17: 'y', 18: 'z',
        }

    #converting image to a np array, and coverting to float
    image=np.array(image)
    image = image.astype(float)
    
    #if converting only one character, need to manually add another dimension
    if image.ndim ==2:
        image=[image]
    
    #normalizing 3d array wrt to 255, so all grayscale values become 0 to 1.
    image=np.array(image)      
    image = image / 255.0
    
    # reshaping array to match shape required by the model
    numberImages=np.size(image, 0)
    image=np.array(image)
    image= image.reshape((len(image), -1))
    
    # predicting value
    prediction=math_MLP_model.predict(image)
    prediction=np.asarray(prediction)
    
    #associating the label to an 'math' character
    predicted_text=list(mapping[int(i)] for i in prediction)
    print(predicted_text)
    return predicted_text

def EMNISTMathCNN_predict (image, EMNISTmath_model, images_height, images_width):
    mapping = {    
        # digits
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        
        # capital lettes
        10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 
        20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
        30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
        
        # lowercase letters
        36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k',
        47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v',
        58: 'w', 59: 'x', 60: 'y', 61: 'z',
        
        # math characters
        62: '+', 63: '.', 64: '/', 65: '=', 66: '*', 67: '-',
        }

    #converting image to a np array, and coverting to float
    image=np.array(image)
    image = image.astype(float)
    
    #if converting only one character, need to manually add another dimension
    if image.ndim ==2:
        image=[image]
    
    #normalizing 3d array wrt to 255, so all grayscale values become 0 to 1.
    image=np.array(image)      
    image = image / 255.0
    
    # reshaping array to match shape required by the model
    numberImages=np.size(image, 0)
    image=np.array(image)
    image= image.reshape(numberImages, images_height, images_width, 1)
    
    # predicting value
    prediction=EMNISTmath_model.predict(image)
    converted = np.argmax(prediction, axis=-1)
    
    #associating the label to an 'math' character
    predicted_text=list(mapping[int(i)] for i in converted)
    # print(predicted_text)
    return predicted_text

# TEST FUNCTIONS -----------------------------------------------------------------------------------
# shows the accuracy of predictions from EMNIST's own set of test images
def EMNIST_test (EMNIST_model, images_height, images_width):
    print(" ")
    print("EMNIST Test")    
    from emnist import extract_test_samples
    
    # Data to test model on:
    test_images, test_labels = extract_test_samples('byclass')

    mapping = {    
         # digits
         0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
         
         # capital lettes
         10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 
         20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
         30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
         
         # lowercase letters
         36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k',
         47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v',
         58: 'w', 59: 'x', 60: 'y', 61: 'z',
    }
    
    # Looking at only the first 15 images
    test_images=test_images[0:15]
    test_labels=test_labels[0:15]

    #normalizing 3d array wrt to 255, so all grayscale values become 0 to 1.
    test_images = test_images / 255.0
    
    # reshaping array to match shape required by the model
    test_images_number = test_images.shape[0]
    test_images = test_images.reshape(test_images_number, images_height, images_width, 1)

    # predicting value
    prediction=EMNIST_model.predict(test_images)
    converted = np.argmax(prediction, axis=-1)
    
    #associating the predicted and test labels to a alphanumeric character
    predicted_text=list(mapping[int(i)] for i in converted)
    print(f"Predicted Characters: {predicted_text}")
    
    actual_text=list(mapping[int(i)] for i in test_labels)
    print(f"Actual Characters:    {actual_text}")

    #displaying the actual images the model is predicting
    fig,axes = plt.subplots(3,5,figsize=(10,8))
    for i,ax in enumerate(axes.flat):
        ax.imshow(test_images[i])
    
    # finding the percent correct in the example test_images
    correct=0
    for i in range(0,len(test_labels)):
        if converted[i] == test_labels[i]:
            correct+=1
    example_accuracy=correct/len(test_labels) * 100
    print(f"Accuracy in Example: {example_accuracy}%")
    
def importMathData ():
    imgs = []
    labels=[]

    #Importing data
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
        
    return imgs, labels

# shows the accuracy of predictions from Math  dataset's own set of test images, using the CNN model
def MathCNN_test (math_CNN_model, images_height, images_width):
    print(" ")
    print("Math CNN Test")    
    mapping = {    
        # digits
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        
        # other characters
        10: '+', 11: '.', 12: '/', 13: '=', 14: '*', 15: '-', 16: 'x', 17: 'y', 18: 'z',
    }

    #loading data
    imgs, labels = importMathData()
    
    #splittinginto train and test
    train_images, test_images, train_labels,test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=42)
    test_images=np.asarray(test_images)
    test_labels=np.asarray(test_labels)
    
    #normalizing wrt 255
    test_images=test_images/255
    
    # predicting the selected test_images
    output=math_CNN_model.predict(test_images[0:15])
    converted = np.argmax(output, axis=-1)
    predicted_text=list(mapping[int(i)] for i in converted)
    print(f"Predicted Characters: {predicted_text}")

    actual=np.asarray(test_labels[0:15])
    actual_text=list(mapping[int(i)] for i in actual)
    print(f"Actual Characters:    {actual_text}")
    
    #plotting the selected test_images
    fig,axes = plt.subplots(3,5,figsize=(10,8))
    for i,ax in enumerate(axes.flat):
        ax.imshow(test_images[i])

    # finding the percent correct in the example test_images
    correct=0
    for i in range(0,len(actual_text[0:15])):
        if predicted_text[i] == actual_text[i]:
            correct+=1
    example_accuracy=correct/len(actual_text) * 100
    print(f"Accuracy in Example: {example_accuracy}%")
    

# shows the accuracy of predictions from Math  dataset's own set of test images, using the MLPClassifier model
def MathMLP_test (math_MLP_model, images_height, images_width):
    dim=images_height
    
    print(" ")
    print("Math MLP Test")   
    mapping = {    
        # digits
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        
        # other characters
        10: '+', 11: '.', 12: '/', 13: '=', 14: '*', 15: '-', 16: 'x', 17: 'y', 18: 'z',
    }

    #loading data
    imgs,labels= importMathData()
    
    #splitting into train and test
    train_images, test_images, train_labels,test_labels = train_test_split(imgs, labels, test_size=0.50, random_state=42)
    test_images=np.asarray(test_images[0:15])
    test_labels=np.asarray(test_labels[0:15])
    
    #normalizing wrt to 255
    test_images=test_images/255
    
    test_images = test_images.reshape((len(test_images), -1))
    
    # predicting the selected test_images
    predictions = math_MLP_model.predict(test_images)
    predicted_text=list(mapping[int(i)] for i in predictions)
    print(f"Predicted Characters: {predicted_text}")

    actual_text=list(mapping[int(i)] for i in test_labels)
    print(f"Actual Characters:    {actual_text}")
    
    #plotting the selected test_images
    test_images_number = test_images.shape[0]
    test_images = test_images.reshape(test_images_number, images_height, images_width, 1)
    fig,axes = plt.subplots(3,5,figsize=(10,8))
    for i,ax in enumerate(axes.flat):
        ax.imshow(test_images[i])

    # finding the percent correct in the example test_images
    correct=0
    for i in range(0,len(test_labels)):
        if predictions[i] == test_labels[i]:
            correct+=1
    example_accuracy=correct/len(test_labels) * 100
    print(f"Accuracy in Example: {example_accuracy}%")
    
def EMNISTMathCNN_test (EMNISTmath_model, images_height, images_width):
    print(" ")
    print("EMNIST Math CNN Test")    
    mapping = {    
        # digits
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
            
        # capital lettes
        10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 
        20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
        30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
        
        # lowercase letters
        36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k',
        47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v',
        58: 'w', 59: 'x', 60: 'y', 61: 'z',
            
        # math characters
        62: '+', 63: '.', 64: '/', 65: '=', 66: '*', 67: '-',
        }
    #loading math data
    imgs = []
    labels=[]
    
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
        
    #randomizing math dataset
    MATHtrain_images, MATHtest_images, MATHtrain_labels, MATHtest_labels = train_test_split(imgs, labels, test_size=0.1, random_state=40)
    
    # loading EMNIST data:
    from emnist import extract_test_samples
    test_images, test_labels = extract_test_samples('byclass')
    
    #randomizing EMNIST dataset
    train_images, test_images, train_labels,test_labels = train_test_split(test_images, test_labels, test_size=0.1, random_state=40)
    #keeping only 10 EMNIST images for testing
    test_images=np.asarray(test_images[0:10])
    test_labels=np.asarray(test_labels[0:10])
    
    #appending 5 Math images to EMNIST images for testing
    test_labels= np.append(test_labels, MATHtest_labels[10:15])
    test_images=np.concatenate((test_images, MATHtest_images[10:15]),axis=0)
    
    #normalizing wrt 255
    test_images=test_images/255
        
    # predicting the selected test_images
    output=EMNISTmath_model.predict(test_images[0:15])
    converted = np.argmax(output, axis=-1)
    predicted_text=list(mapping[int(i)] for i in converted)
    print(f"Predicted Characters: {predicted_text}")

    actual=np.asarray(test_labels[0:15])
    actual_text=list(mapping[int(i)] for i in actual)
    print(f"Actual Characters:    {actual_text}")
        
    #plotting the selected test_images
    fig,axes = plt.subplots(3,5,figsize=(10,8))
    for i,ax in enumerate(axes.flat):
        ax.imshow(test_images[i])

    # finding the percent correct in the example test_images
    correct=0
    for i in range(0,len(actual_text[0:15])):
        if predicted_text[i] == actual_text[i]:
            correct+=1
    example_accuracy=correct/len(actual_text) * 100
    print(f"Accuracy in Example: {example_accuracy}%")
        
#DEMOING TESTS ---------------------------------------------------------------------------------------------
# importing all models
EMNIST_model = keras.models.load_model('EMNIST Model')
math_CNN_model = keras.models.load_model('Math CNN Model')
with open('Math MLP Model.pkl', 'rb') as f:
    math_MLP_model = pickle.load(f)
EMNISTmath_model = keras.models.load_model('EMNIST Math CNN Model')

# calling all test functions
images_height=28
images_width=28

# EMNIST_test (EMNIST_model, images_height, images_width)
# MathCNN_test (math_CNN_model, images_height, images_width)
# MathMLP_test(math_MLP_model, images_height, images_width)
# EMNISTMathCNN_test(EMNISTmath_model, images_height, images_width)