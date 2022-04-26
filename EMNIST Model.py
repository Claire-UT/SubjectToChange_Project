# -*- coding: utf-8 -*-
"""
Team 6- Subject to Change
Members- Laura Homiller, David Spitler, Claire Zhang

Dataset from: https://www.nist.gov/itl/products-and-services/emnist-dataset
http://yann.lecun.com/exdb/mnist/

Project Insporation:
    
"""
#Install EMIST library, import datasets of letters, Matplotlib
#STEP 1: In the kernal use pip install emnist AND pip install tensorflow AND pip install keras (will take A Second.... took me like 40 mins for everything)

#Stuff I actually used:
# https://pypi.org/project/emnist/
# https://github.com/rcgopi100/EMNISTHandwrittenLetters/blob/main/EMINST.ipynb
# EMNIST project: https://www.kaggle.com/code/achintyatripathi/emnist-letter-dataset-97-9-acc-val-acc-91-78/notebook

from emnist import list_datasets
from emnist import extract_training_samples
from emnist import extract_test_samples
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
#for the CNN
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Dense,Dropout,MaxPooling2D,Conv2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from sklearn.model_selection import train_test_split

#---- importing data set
train_images, train_labels = extract_training_samples('letters')
test_images, test_labels = extract_test_samples('letters')
print(f"Train Image Shape: {train_images.shape}, Train Label Shape: {train_labels.shape}")

# keeping only a quarter of the dataset because there is too much to process.
# train_images_number = int(1/4*len(train_images[0]))
train_images_number= int(1/4*124800)
train_images=train_images[0:train_images_number]
train_labels=train_labels[0:train_images_number]
test_images=test_images[0:train_images_number]
test_labels=test_labels[0:train_images_number]

print(f"Train Image Shape: {train_images.shape}, Train Label Shape: {train_labels.shape}")


#---- visualize data
import matplotlib.pyplot as plt 
fig,axes = plt.subplots(3,5,figsize=(10,8))
for i,ax in enumerate(axes.flat):
    ax.imshow(train_images[i])
print(f"Train Image Shape: {train_images.shape}, Train Label Shape: {train_labels.shape}")

#----- Normalise and reshape data ... NOT SURE IF NEEDED....
# Normalising wrt to RGB, which has 255 values
train_images = train_images / 255.0
test_images = test_images / 255.0

# Tells number of images, because train_images.shape returns row, col. [0] gives row length
train_images_number = train_images.shape[0]
# print(train_images_number)
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


# other refs: text recognition w EMNIST project https://jovian.ai/goyalbhavya529/emnist-project
# models w highest accuracy: https://paperswithcode.com/dataset/emnist
# https://github.com/saranshgupta121/HANDWRITTEN-TEXT-RECOGNITION-EMNIST-/blob/master/modeltrain.ipynb


# Transform labels
number_of_classes = 37

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
MCP = ModelCheckpoint('Best_points.h5',verbose=1,save_best_only=True,monitor='val_accuracy',mode='max')
ES = EarlyStopping(monitor='val_accuracy',min_delta=0,verbose=0,restore_best_weights = True,patience=3,mode='max')
RLP = ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.2,min_lr=0.0001)

history = model.fit(train_x,train_y,epochs=10,validation_data=(test_x,test_y),callbacks=[MCP,ES,RLP])
print('done 1')

# Save model, and then load into file where it is integrated into vision
# Reference: https://www.tensorflow.org/guide/keras/save_and_serialize
model.save('EMNIST Model')



# #Let's plot Accuracy vs Val_Accuracy to further evaluation..
# import seaborn as sns
# q = len(history.history['accuracy'])

# plt.figsize=(10,10)
# sns.lineplot(x = range(1,1+q),y = history.history['accuracy'], label='Accuracy')
# sns.lineplot(x = range(1,1+q),y = history.history['val_accuracy'], label='Val_Accuracy')
# plt.xlabel('epochs')
# plt.ylabel('Accuracy')

# train_x2,test_x2,train_y2,test_y2 = train_test_split(train_images,y1,test_size=0.15,random_state = 42)

# history1 = model.fit(train_x2,train_y2,epochs=10,validation_data=(test_x2,test_y2))


# q = len(history1.history['accuracy'])

# plt.figsize=(10,10)
# sns.lineplot(x = range(1,1+q),y = history1.history['accuracy'], label='Accuracy')
# sns.lineplot(x = range(1,1+q),y = history1.history['val_accuracy'], label='Val_Accuracy')
# plt.xlabel('epochs')
# plt.ylabel('Accuracy')