import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from tensorflow import keras
import tensorflow as tf

np.set_printoptions(threshold=np.inf)

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

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, im = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

blank = np.zeros(image.shape, dtype='uint8')
new_img = cv2.drawContours(blank,contours, -1, (0,0,0))
cropped_images = []
predictions = []
resized = []

# dimension of image for machine learning:
dim = (28,28)

for contour in contours:
    (x,y,w,h) = cv2.boundingRect(contour)
    # for now, try just making it square by using the max of width, height
    w,h = max(w,h), max(w,h)
    crp = im[y:y+h,x:x+w]
    #resize to correct dimensions
    resized_image = cv2.resize(thresh, dim,interpolation = cv2.INTER_AREA)
    cropped_images.append((im[y:y+h,x:x+w],[x,y,w,h]))
    resized.append(resized_image)

cropped = thresh[y:y+h,x:x+w]
resized_cropped = cv2.resize(cropped, dim,interpolation = cv2.INTER_AREA)
# print(cropped.astype('int'))
# show_image(cropped)

# print(resized_image.astype('int'))
# show_image(resized_image)

resized_cropped.astype('int')
# print(resized_cropped)
# print(type(resized_cropped))
show_image(resized_cropped)

# EMNIST PREDICTIONS ----------------------------------------
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

#adding in neural network:
model = keras.models.load_model('EMNIST Model')
# print(resized_cropped)
resized_cropped = resized_cropped / 255.0
# !IMPORTANT! below line arbitrarily adds the image into an ndarray of images, since this is the input the model accepts. Assumed that in the future, resized will already contain multiple images (3d ndarray)

resized_cropped = np.array([resized_cropped])
# print('normallized: ')
# print(resized_cropped)
# print(resized_cropped.shape)

numberImages=np.size(resized_cropped, 0)
# print("numberImages: " + str(numberImages))
train_images_height = 28
train_images_width = 28
resized_cropped= resized_cropped.reshape(numberImages, train_images_height, train_images_width, 1)

            
predictions=model.predict(resized_cropped)
# print('done')
converted = np.argmax(predictions, axis=-1)
print(converted)
print(list(mapping[i] for i in converted))

# MATH PREDICTIONS ---------------------------------------
mapping = {    
    # digits
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    
    # other characters
    10: '+', 11: '.', 12: '/', 13: '=', 14: '*', 15: '-', 16: 'x', 17: 'y', 18: 'z',
}

#adding in neural network:
model = keras.models.load_model('Math Model')
# print(resized_cropped)
resized_cropped = resized_cropped / 255.0
# !IMPORTANT! below line arbitrarily adds the image into an ndarray of images, since this is the input the model accepts. Assumed that in the future, resized will already contain multiple images (3d ndarray)

resized_cropped = np.array([resized_cropped])
# print('normallized: ')
# print(resized_cropped)
# print(resized_cropped.shape)

numberImages=np.size(resized_cropped, 0)
# print("numberImages: " + str(numberImages))
train_images_height = 28
train_images_width = 28
resized_cropped= resized_cropped.reshape(numberImages, train_images_height, train_images_width, 1)

            
predictions=model.predict(resized_cropped)
# print('done')
converted = np.argmax(predictions, axis=-1)
print(converted)
print(list(mapping[i] for i in converted))


