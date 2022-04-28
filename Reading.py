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
    
    # Ref for knowing what the mapping should be like: https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&color_mode=auto&commit=6d7e042d9cc69c34b84ea51c2f314838359b57dc&device=unknown&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f47696666792f41495f454d4e4953542d6368617261637465722d7265636f676e6974696f6e2f366437653034326439636336396333346238346561353163326633313438333833353962353764632f454d4e4953545f6279436c6173735f4750555f2e6970796e62&logged_in=false&nwo=Giffy%2FAI_EMNIST-character-recognition&path=EMNIST_byClass_GPU_.ipynb&platform=android&repository_id=159695117&repository_type=Repository&version=98
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



#Loading EMNIST model back in
# Ref: https://www.tensorflow.org/guide/keras/save_and_serialize
model = keras.models.load_model('EMNIST Model')
# print(type(model))

#Data to test model on:
# test_images, test_labels = extract_test_samples('letters')
test_images, test_labels = extract_test_samples('byclass')

test_images=test_images[0:15]
test_labels=test_labels[0:15]

test_images = test_images / 255.0

test_images_number = test_images.shape[0]
test_images_height = 28
test_images_width = 28
test_images_size = test_images_height*test_images_width

test_images = test_images.reshape(test_images_number, test_images_height, test_images_width, 1)

number_of_classes=62
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