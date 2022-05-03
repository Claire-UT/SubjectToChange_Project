# SubjectToChange_Project: How to use the code

Download all files off the Main branch of the Github Repo 

Open the file ""
=============================
--ABOUT EACH FILE--
-REAL TIME OPTICAL CHARACTER RECOGNITION-
Run main?

-USING & TESTING MODELS-
'Predictions_ModelTests.py' - This file contains the functions used for predicting characters. 
A single image must be formatted as a numpy array of size 28x28 storing the grayscale value of each pixel.  
Each 'predict' function is capable of handling an 'image' parameter that is a 2d array (single image) or 3d array (multiple images).
    For using the 'predict' functions:
        1. Import the relevent function into your file.
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
        3. The function will output the plot of images examined, the predicted characters, the actual characters, and the accuracy of the example.

-MODEL TRAINING-
'EMNIST_Model.py' - install the EMNIST data set, and run the file to train the EMNIST_model.
'Math CNN Model.py' - install the MATH data set, and run the file to train the math_CNN_model.
'Math MLP Model.py' - install the MATH data set, and run the file to train the math_MLP_model.
'EMNIST_Math_CNN Model.py' - install the EMNIST and MATH data sets, and run the file to trian the EMNISTmath_model.

=============================
--DATASETS--
If you want to retrain the models, you will need to download the datasets.
EMNIST data set: use 'pip install emnist' (will take a while)
Math dataset: visit https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols
    -Download the dataset in the same location as the python files
    -Open folder '9' and delete the directory file.
