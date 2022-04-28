"""
================================
Recognizing handwritten digits
================================

This example is heavily adapted from examples by 'The Data Frog' and Gael 
Varoquaux. Check out their examples here:
    
Data Frog's: https://thedatafrog.com/en/articles/handwritten-digit-recognition-scikit-learn/
Gael's: https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py

"""

import matplotlib.pyplot as plt
from sklearn import datasets #Pre-made data set to learn on from sklearn library
from sklearn.model_selection import train_test_split #Split training and test data
from sklearn.neural_network import MLPClassifier #Create neural network for classifier
from sklearn.metrics import accuracy_score, classification_report #Find accuracy of model


def makeNN():   
    digits = datasets.load_digits()
    #Digits contains 8x8 arrays of varying gradient squares
    #There are 1797 array "images" meant to look like a number (0-9)
    #They each have an associated "target value", which is the number it looks like
    
    #Plot the first 16 digits with their associated target value
    # def plot_multi(i):
    #     '''Plots 16 digits, starting with digit i'''
    #     nplots = 16
    #     fig = plt.figure(figsize=(15,15))
    #     for j in range(nplots):
    #         plt.subplot(4,4,j+1)
    #         plt.imshow(digits.images[i+j], cmap='binary')
    #         plt.title(digits.target[i+j])
    #         plt.axis('off')
    #     plt.show()
    # plot_multi(0)
    
    #Reduce input array from 2D to 1D to "flatten" the image
    y = digits.target
    x = digits.images.reshape((len(digits.images), -1))
    x.shape #Now 64 elements in array rather than 8x8
    
    #Randomly split data into 50% train and 50% test subsets
    x_train, x_test, y_train, y_test = train_test_split(
        x, digits.target, test_size=0.5, shuffle=True
    )
    
    #Create neural network using Stochastic Gradient Descent (sgd) classifier
    mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic', alpha=1e-4,
                        solver='sgd', tol=1e-4, random_state=1,
                        learning_rate_init=.1, verbose=True)
    #Takes 64 input nodes from array in input layer
    #Sorted by 50 neurons (user defined) in hidden layer
    #Outputs with 10 neurons corresponding to 0-9 digits
    
    mlp.fit(x_train,y_train) #Train the neural network
    #Loss is average difference between actual and predicted value for the iteration
    
    return mlp
    # predictions = mlp.predict(x_test) #Test trained network on unused test data
    
    #Display 4 random test samples and show their predicted digit value
    # _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    # for ax, image, prediction in zip(axes, x_test, predictions):
    #     ax.set_axis_off()
    #     image = image.reshape(8, 8)
    #     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    #     ax.set_title(f"Prediction: {prediction}")
    
    #Output total accuracy of prediction vs actual value for all test data
    # accuracy = float(accuracy_score(y_test, predictions))
    # accuracypercent = round((accuracy * 100),2)
    print('done.')
    # print('The model correctly predicted digits with '+ str(accuracypercent) +'% accuracy')
    # print(classification_report(y_test,predictions))
    

def predictNum(num,mlp):
    return mlp.predict([num.flatten()])