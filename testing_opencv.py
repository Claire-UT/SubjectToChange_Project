# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 21:59:22 2022

@author: laura
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets #Pre-made data set to learn on from sklearn library
from sklearn.model_selection import train_test_split #Split training and test data
from sklearn.neural_network import MLPClassifier #Create neural network for classifier
from sklearn.metrics import accuracy_score, classification_report #Find accuracy of model


from DigitsDemo import *

# word = cv2.imread('hello.jpg',1)
# # word = cv2.resize(word,(0,0),fx=1,fy=1)
# black_low = np.array([0,0,0])
# black_high = np.array([105,105,105])

# wordBlack = cv2.inRange(word,black_low,black_high)

# # cv2.imshow(wordBlack)
# # cv2.imshow('Hello',word)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# print(wordBlack.shape)

# cv2.imwrite('hello2.png', wordBlack)

# im = cv2.imread('hello.jpg')
# imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(imgray, 127, 255, 0)
# im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(im, contours, -1, (0,255,0), 3)


def show_image(image):
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
# initialize camera
cam = cv2.VideoCapture(0)    

result, image = cam.read()

good = False
while not good:
    result, image = cam.read()
    show_image(image)
    if input("good?") == 'y':
        good = True

# image = cv2.imread('alphabet.jpg')
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, im = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# res_img = cv2.drawContours(image, contours, -1, (0,255,75), 2)
# show_image(res_img)

blank = np.zeros(image.shape, dtype='uint8')
new_img = cv2.drawContours(blank,contours, -1, (0,0,0))
cropped_images = []
predictions = []
resized = []

# dimension of image for machine learning:
dim = (8,8)

# make neural network
mlp = makeNN()

for contour in contours:
    (x,y,w,h) = cv2.boundingRect(contour)
    # for now, try just making it square by using the max of width, height
    w,h = max(w,h), max(w,h)
    crp = im[y:y+h,x:x+w]
    #resize to correct dimensions
    resized_image = cv2.resize(crp, dim,interpolation = cv2.INTER_AREA)
    cropped_images.append((im[y:y+h,x:x+w],[x,y,w,h]))
    resized.append(resized_image)
    predictions.append(predictNum(resized_image, mlp))
    


# cropped = image[y:y+h,x:x+w]
# cv2.rectangle(image, (x,y), (x+w,y+h), (255, 0, 0), 2)
# show_image(cropped)

#TEST ON DIGITS NETWORK

