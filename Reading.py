# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 13:22:27 2022

@author: clair
"""

from tensorflow import keras
import tensorflow as tf
from emnist import extract_test_samples
import numpy as np


#FORMAT FOR INPUTTING INTO MODEL:
    # Each image needs to be in a list of lists (28 x 28) each element is an integer from 0 to 255 representing one pixel, needs to be in gray scale. 
    # Can give me list of images (so a list of list of lists)
    
    
mapping = {
    1: 'a',
    2: 'b',
    3: 'c',
    4: 'd',
    5: 'e',
    6: 'f',
    7: 'g',
    8: 'h',
    9: 'i',
    10: 'j',
    11: 'k',
    12: 'l',
    13: 'm',
    14: 'n',
    15: 'o',
    16: 'p',
    17: 'q',
    18: 'r',
    19: 's',
    20: 't',
    21: 'u',
    22: 'v',
    23: 'w',
    24: 'x',
    25: 'y',
    26: 'z',
}


#Loading EMNIST model back in
# Ref: https://www.tensorflow.org/guide/keras/save_and_serialize
model = keras.models.load_model('EMNIST Model')
# print(type(model))

#Data to test model on:
test_images, test_labels = extract_test_samples('letters')
test_images=test_images[0:15]
test_labels=test_labels[0:15]

test_images = test_images / 255.0

test_images_number = test_images.shape[0]
test_images_height = 28
test_images_width = 28
test_images_size = test_images_height*test_images_width

test_images = test_images.reshape(test_images_number, test_images_height, test_images_width, 1)

number_of_classes=37
# y2 = tf.keras.utils.to_categorical(test_labels, number_of_classes)

# Model.predict info: https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
#maybe individual, maybe list
output=model.predict(test_images)
print(output)
#37 outputs, each one corresponds to a class. the highest value is the one with the highest probability
converted = np.argmax(output, axis=-1)
print(converted)
print(list(mapping[i] for i in converted))
print(test_labels)

import matplotlib.pyplot as plt 
fig,axes = plt.subplots(3,5,figsize=(10,8))
for i,ax in enumerate(axes.flat):
    ax.imshow(test_images[i])
    
# Convert back from categorical to original labels: https://www.google.com/search?q=tensorflow+convert+back+from+categorical&rlz=1C1VDKB_enUS981US981&oq=tensorflow+convert+back+from+categorical&aqs=chrome..69i57j33i160l2.7491j0j7&sourceid=chrome&ie=UTF-8
# This line takes a list of lists (each inside list is the binary representation of each category) and converts it back to a list of the original labels
# need to import numpy as np
# original= np.argmax(list_of_lists, axis=-1)