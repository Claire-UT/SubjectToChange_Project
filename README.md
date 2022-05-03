# SubjectToChange_Project: How to use the code

# QUICK START GUIDE
Download all files off the Main branch of the Github Repo 
The most important files are Demo_main.py, Predictions_ModelTests.py, EMNIST Math CNN Model folder, EMNIST Model folder, Math CNN Model folder, Math MLP Model.pkl
Open the file **Demo_main.py**.

# ABOUT EACH FILE & FUNCTIONALITY
REAL TIME OPTICAL CHARACTER RECOGNITION
Run main?

## USING & TESTING MODELS
**'Predictions_ModelTests.py'** - This file contains the functions used for predicting characters with the already trained models. 
A single image must be formatted as a numpy array of size 28x28 storing the grayscale value of each pixel. 

Each 'predict' function is capable of handling an 'image' parameter that is a 2d array (single image) or 3d array (multiple images).

For using the 'predict' functions:
1. Import the relevant function into your file.
2. Call the function with correct parameters. 
   * 'Image'  is either a 2D array (28x28) or 3D array (?x28x28) storing the grayscale value of each pixel
   * ‘'...model' pass in the correct model
   * 'images_height' = 28, unless trained on new dimensions
   * 'images_width' = 28, unless trained on new dimensions
3. The function will return the predicted text as a list

For using the test functions:
1. Download the datasets
2. Uncomment the DEMOING TESTS section at the end of this file
3. Run the file
4. The function will output the plot of images examined, the predicted characters, the actual characters, and the accuracy of the example.

## DATASETS
If you want to retrain the models, you will need to download the datasets.
1. EMNIST data set: use 'pip install emnist' (will take a while)
2. Math dataset: visit https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols
   * Download the dataset in the same location as the python files
   * Open folder '9' and delete the directory file

## MODEL TRAINING
**'EMNIST_Model.py'** - install the EMNIST data set, and run the file to train the EMNIST_model.
Stores best points in the ‘Best_points.h5’ file
Stores model in the ‘EMNIST Model’ folder

**'Math CNN Model.py'** - install the MATH data set, and run the file to train the math_CNN_model.
Stores best points in the ‘Best_points_Math.h5’ file
Stores model in the ‘Math CNN Model’ folder

**'Math MLP Model.py'** - install the MATH data set, and run the file to train the math_MLP_model.
Stores model in the ‘Math MLP Model.pkl’ file

**'EMNIST_Math_CNN Model.py'** - install the EMNIST and MATH data sets, and run the file to train the EMNISTmath_model.
Stores best points in the ‘Best_points_EMNISTMath.h5’ file
Stores model in the ‘EMNIST Math CNN Model’ folder

## Enable LaTeX OUTPUT
1. pip install pylatex
2. Install a LateX compiler (MiKTeX worked well for Windows)
4. Uncomment lines 6-9 to import required packages
5. Modify path in makePDF function to point to location of pdflatex (or other compiler)
7. Uncomment call to makePDF at end of main to enable output
