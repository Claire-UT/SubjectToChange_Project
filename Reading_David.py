import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from tensorflow import keras
import tensorflow as tf

# Import required packages
import cv2
import numpy as np

def show_image(image):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cam = cv2.VideoCapture(0)

while True:

    #Show the video feed
    result, img = cam.read()
    cv2.imshow('LiveStream', img)    

    # Preprocessing the image starts
    
    #Smooth the image
    kernel = np.ones((5,5),np.float32)/25
    smooth1 = cv2.filter2D(img,-1,kernel)
    smooth = cv2.filter2D(smooth1,-1,kernel)
     
    # Convert the image to gray scale
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
     
    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    #thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    
    #Blur threshold to get rid of any dots
    blur = cv2.medianBlur(thresh1, 3)
     
    # Specify structure shape and kernel size.
    # Kernel size increases or decreases the area
    # of the rectangle to be detected.
    # A smaller value like (10, 10) will detect
    # each word instead of a sentence.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
     
    # Applying dilation on the threshold image
    dilation = cv2.dilate(blur, rect_kernel, iterations = 1)
    dilation2 = cv2.dilate(blur, rect_kernel, iterations = 3)
    
    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_NONE)
     
    # Creating a copy of image
    im2 = img.copy()
    
    #Test Data File Dimensions to be resized to
    dim = (28,28)
     
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
    # Looping through the identified contours
    # Then rectangular part is cropped and padded with white space to make square
    # This is resized to 28x28 and also shown in 280x280
    alltext=[]
    train_images_height = 28
    train_images_width = 28
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        old_image_height = h
        old_image_width = w
        new_image_width = max(w,h)
        new_image_height = max(w,h)
        mask = np.full((new_image_height,new_image_width), 255, dtype=np.uint8)
        
        # compute center offset
        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2
        
        invthresh = cv2.bitwise_not(thresh1)
        cropped = invthresh[y:y+h,x:x+w]
    
        # copy img image into center of result image
        mask[y_center:y_center+old_image_height,x_center:x_center+old_image_width] = cropped
        
        #Resize square, padded value to input dim and larger viewable 280x280 size
        resized_cropped = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
        Big = cv2.resize(mask, (280,280),interpolation = cv2.INTER_AREA)
        
        #converting resized_cropped to a np array, and coverting to float before normaling wrt 255.
        resized_cropped=np.array(resized_cropped)
        resized_cropped = resized_cropped.astype(float)
        
        for row in resized_cropped:
            for i in range(len(row)):
                row[i] = row[i] / 255.00
            
        # numberImages=np.size(resized_cropped, 0)
        numberImages=1
        resized_cropped= resized_cropped.reshape(numberImages, train_images_height, train_images_width, 1)
        predictions=model.predict(resized_cropped)
        alltext.append(resized_cropped)
        
        converted = np.argmax(predictions, axis=-1)
        print(converted)
        print(mapping.get(converted[0]))
        # print(resized_cropped.astype('int'))
        #cv2.imshow('28x28 Image', resized_cropped)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    # cv2.imshow('280x280 Image', Big)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        alltext.append(mapping.get(converted[0]))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
alltext=np.array(alltext)
# print(alltext)
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()


# # MATH PREDICTIONS WITH EMNIST MODEL---------------------------------------
# mapping = {    
#     # digits
#     0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    
#     # other characters
#     10: '+', 11: '.', 12: '/', 13: '=', 14: '*', 15: '-', 16: 'x', 17: 'y', 18: 'z',
# }

# #adding in neural network:
# model = keras.models.load_model('Math Model')
# # print(resized_cropped)
# resized_cropped = resized_cropped / 255.0
# # !IMPORTANT! below line arbitrarily adds the image into an ndarray of images, since this is the input the model accepts. Assumed that in the future, resized will already contain multiple images (3d ndarray)

# resized_cropped = np.array([resized_cropped])
# # print('normallized: ')
# # print(resized_cropped)
# # print(resized_cropped.shape)

# numberImages=np.size(resized_cropped, 0)
# # print("numberImages: " + str(numberImages))
# train_images_height = 28
# train_images_width = 28
# resized_cropped= resized_cropped.reshape(numberImages, train_images_height, train_images_width, 1)

            
# predictions=model.predict(resized_cropped)
# # print('done')
# converted = np.argmax(predictions, axis=-1)
# print(converted)
# print(list(mapping[i] for i in converted))

# # MATH PREDICTIONS WITH MLPClassifier MODEL---------------------------------------
# import pickle 
# mapping = {    
#     # digits
#     0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    
#     # other characters
#     10: '+', 11: '.', 12: '/', 13: '=', 14: '*', 15: '-', 16: 'x', 17: 'y', 18: 'z',
# }


# #adding in neural network:
# with open('Math model.pkl', 'rb') as f:
#     model = pickle.load(f)
    
# # print(resized_cropped)
# # for image in resized_cropped:
# #     for row in image:
# #         for i in range(len(row)):
# #             row[i] = float(row[i] / 255)

# resized_cropped = resized_cropped / 255.0
            
# # !IMPORTANT! below line arbitrarily adds the image into an ndarray of images, since this is the input the model accepts. Assumed that in the future, resized will already contain multiple images (3d ndarray)

# resized_cropped = np.array([resized_cropped])
# # print('normallized: ')
# # print(resized_cropped)
# # print(resized_cropped.shape)

# # numberImages=np.size(resized_cropped, 0)
# # print("numberImages: " + str(numberImages))
# train_images_height = 28
# train_images_width = 28

# resized_cropped= resized_cropped.reshape((len(resized_cropped), -1))
# # resized_cropped= resized_cropped.reshape(numberImages, train_images_height, train_images_width, 1)
            
# predictions=model.predict(resized_cropped)
# # print('done')
# # converted = np.argmax(predictions, axis=-1)
# # print(converted)
# predictions=np.asarray(predictions)
# print(type(predictions))
# print(list(mapping[int(i)] for i in predictions))


# # TESTING THE EMNIST MODEL-------------------------------------------
# from emnist import extract_test_samples
# # Loading EMNIST model back in
# # Ref: https://www.tensorflow.org/guide/keras/save_and_serialize
# model = keras.models.load_model('EMNIST Model')
# # Data to test model on:
# test_images, test_labels = extract_test_samples('byclass')

# test_images=test_images[0:15]
# test_labels=test_labels[0:15]
# # print(test_images)
# test_images = test_images / 255.0
# # test_images=np.ceil(test_images)
# test_images=np.rint(test_images)
# test_images_number = test_images.shape[0]
# test_images_height = 28
# test_images_width = 28
# test_images_size = test_images_height*test_images_width

# # print(type(test_images))
# # print(test_images[0])
# print(type(test_images[0][0][0]))
# test_images = test_images.reshape(test_images_number, test_images_height, test_images_width, 1)

# number_of_classes=62
# # y2 = tf.keras.utils.to_categorical(test_labels, number_of_classes)

# # Model.predict info: https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
# #maybe individual, maybe list
# output=model.predict(test_images)
# # print(output)
# #37 outputs, each one corresponds to a class. the highest value is the one with the highest probability
# converted = np.argmax(output, axis=-1)
# # print(converted)
# print(list(mapping[i] for i in converted))
# print(test_labels[0])

# import matplotlib.pyplot as plt 
# fig,axes = plt.subplots(3,5,figsize=(10,8))
# for i,ax in enumerate(axes.flat):
#     ax.imshow(test_images[i])
# # print(test_images[7])

# Convert back from categorical to original labels: https://www.google.com/search?q=tensorflow+convert+back+from+categorical&rlz=1C1VDKB_enUS981US981&oq=tensorflow+convert+back+from+categorical&aqs=chrome..69i57j33i160l2.7491j0j7&sourceid=chrome&ie=UTF-8
# This line takes a list of lists (each inside list is the binary representation of each category) and converts it back to a list of the original labels
# need to import numpy as np
# original= np.argmax(list_of_lists, axis=-1)

# # TESTING THE MATH MODEL-------------------------------------------
# # Model.predict info: https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict

# from sklearn.model_selection import train_test_split
# import numpy as np
# import os
# from PIL import Image
# import cv2

# imgs = []
# labels=[]

# #Importing data
# path = "dataset/0"
# # converts each image into an np.array, and adds it to the imgs np.array. also creates a label
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('0')

# path = "dataset/1"
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('1')

# path = "dataset/2"
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('2')

# path = "dataset/3"
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('3')
    
# path = "dataset/4"
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('4')
    
# path = "dataset/5"
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('5')

# path = "dataset/6"
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('6')
    
# path = "dataset/7"
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('7')
    
# path = "dataset/8"
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('8')
    
# path = "dataset/9"
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('9')
    
# path = "dataset/add"
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('10')
    
# path = "dataset/dec"
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('11')
    
# path = "dataset/div"
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('12')
    
# path = "dataset/eq"
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('13')
    
# path = "dataset/mul"
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('14')
    
# path = "dataset/sub"
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('15')
    
# path = "dataset/x"
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('16')
    
# path = "dataset/y"
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('17')
    
# path = "dataset/z"
# for f in os.listdir(path):
#     image=Image.open(os.path.join(path,f))
#     image=np.array(image)
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     image2=cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
#     imgs.append(image2)
#     labels.append('18')
    
# mapping = {    
#     # digits
#     0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    
#     # other characters
#     10: '+', 11: '.', 12: '/', 13: '=', 14: '*', 15: '-', 16: 'x', 17: 'y', 18: 'z',
# }

# #splittinginto train and test
# train_images, test_images, train_labels,test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=42)
# train_images=np.asarray(train_images)
# train_labels=np.asarray(train_labels)
# test_images=np.asarray(test_images)
# test_labels=np.asarray(test_labels)

# #loading model
# model = keras.models.load_model('Math Model')


# #n determines which chunk of data to look at.
# n=12 
# output=model.predict(test_images[0+15*n:15+15*n])
# # print(output)
# #37 outputs, each one corresponds to a class. the highest value is the one with the highest probability
# converted = np.argmax(output, axis=-1)
# # print(converted)
# print(list(mapping[i] for i in converted))

# actual=np.asarray(test_labels[0+15*n:15+15*n])
# # print(actual)
# print(list(mapping[int(i)] for i in actual))

# import matplotlib.pyplot as plt 
# fig,axes = plt.subplots(3,5,figsize=(10,8))
# for i,ax in enumerate(axes.flat):
#     ax.imshow(test_images[i+15*n])
