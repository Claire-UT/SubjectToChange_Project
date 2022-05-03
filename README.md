# SubjectToChange_Project: How to use the code

# QUICK START GUIDE
Download all files off the Main branch of the Github Repo 

Open the file **Demo_main.py**, and hit run.
This is with a pre-trained model, to train your own models check the Datasets and Model Training sections.

Demo_main.py is currently commented to detect and display the text from “test2.jpg”. This can be changed to any .jpg or .png file. To run Demo_main using the real time camera detection, comment out everything after “FROM IMAGE” and uncomment the “FROM CAMERA” section. 

Note: FROM CAMERA uses cv2.VideoCapture(0), which is usually a laptop's built-in webcam. If using a USB webcam, “cam” may need to be changed to cv2.VideoCapture(1).

# ABOUT EACH FILE & FUNCTIONALITY
## REAL TIME OPTICAL CHARACTER RECOGNITION
Real time OCR will run and work automatically when running Demo_main.py

If you wish to update any settings for your specific environment, general tips are provided below:

*Smooth* - image is currently smoothed twice in pre-processing. If your image is blurry to start with, delete “smooth” and replace the input for “gray” with “smooth1” to only smooth the image once.

*Thresholding* - The image threshold is currently set for a bright environment and thresholds anything above a “0” grayscale value. For darker environments, increase this threshold value. A threshold of “100” is a good place to start.

*Blur* - To increase the area blurred, “3” can be increased to a higher odd number. This is the kernel size that is being blurred.

*Dilation* - Dilation combines disjointed letters like “i” and “j” to ensure they are contoured together. If need be, dilation iterations can be increased from 1 to 2 for letters that are disjointed by large distances.

*Dilation2* - Dilation 2 is used to join words or equations together for contouring the output region. Current iterations are set to 4. Decrease this if words are generally close together, or increase this if letters are generally far apart within the same word. 

*Dim* - Currently, all detected characters are resized to 28x28 pixels. To change this size, change “dim” to equal your desired image size.

If your training data is a black letter on a white background, uncomment “invthresh”, change “cropped” to equal invthresh[y:y+h,x:x+w], and change the 0 in “mask” to be 255.

## USING & TESTING MODELS
**'Predictions_ModelTests.py'** - This file contains the functions used for predicting characters. 
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
* Stores best points in the ‘Best_points.h5’ file
* Stores model in the ‘EMNIST Model’ folder

**'Math CNN Model.py'** - install the MATH data set, and run the file to train the math_CNN_model.
* Stores best points in the ‘Best_points_Math.h5’ file
* Stores model in the ‘Math CNN Model’ folder

**'Math MLP Model.py'** - install the MATH data set, and run the file to train the math_MLP_model.
* Stores model in the ‘Math MLP Model.pkl’ file

**'EMNIST_Math_CNN Model.py'** - install the EMNIST and MATH data sets, and run the file to train the EMNISTmath_model.
* Stores best points in the ‘Best_points_EMNISTMath.h5’ file
* Stores model in the ‘EMNIST Math CNN Model’ folder


## Enable LaTeX OUTPUT
1. pip install pylatex
2. Install a LaTeX compiler (MiKTeX worked well for Windows)
3. Uncomment lines 6-9 in ‘Demo_main.py’ to import required packages
4. Modify path in makePDF function to point to location of pdflatex (or other compiler)
5. Uncomment call to makePDF at end of ‘Demo_main.py’ to enable output

