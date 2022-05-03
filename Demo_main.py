# Import required packages
import cv2
import numpy as np
import keras
from Predictions_ModelTests import *
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Matrix, Alignat
from pylatex.utils import italic
from pdflatex import PDFLaTeX
import os
import platform
import subprocess

def show_image(image):
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def inBound(bound1, bound2, loc):
    return bound1[0] <= loc[0] and bound2[0] >= loc[0] and bound1[1] <= loc[1] and bound2[1] >= loc[1]

def processImage(img):
    # Preprocessing the image starts
    
    #Smooth the image
    kernel = np.ones((5,5),np.float32)/25
    smooth1 = cv2.filter2D(img,-1,kernel)
    smooth = cv2.filter2D(smooth1,-1,kernel)
     
    # Convert the image to gray scale
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
     
    # Performing OTSU threshold
    # ret, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    ret, thresh1 = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    #thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    # show_image(thresh1)
    
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
    
    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_NONE)
    return contours, thresh1

def getWordBounds(image):
    #Blur threshold to get rid of any dots
    blur = cv2.medianBlur(image, 3)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    #Dilate with extra iterations so letters in same word touch
    dilation2 = cv2.dilate(blur, rect_kernel, iterations = 4)
    #Get contours for each word
    wordContours, hierarchy2 = cv2.findContours(dilation2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return wordContours

def cropLetters(contours,img,dim):
    letters = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        old_image_height = h
        old_image_width = w
        new_image_width = max(w,h)
        new_image_height = max(w,h)
        #Make white image of correct dimension
        mask = np.full((new_image_height,new_image_width), 255, dtype=np.uint8)
        
        # compute center offset
        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2
        
        invthresh = cv2.bitwise_not(img)
        cropped = invthresh[y:y+h,x:x+w]
    
        # copy img image into center of result image
        mask[y_center:y_center+old_image_height,x_center:x_center+old_image_width] = cropped
        
        #Resize square, padded value to input dim and larger viewable 280x280 size
        resized_cropped = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)

        letters.append((resized_cropped,[x,y,w,h]))
    return letters

def printWordsToScreen(letters,wordContours, img):
    words = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for word in wordContours:
        collectLetters = ''
        x1,y1,w,h = cv2.boundingRect(word)
        x2, y2 = x1 + w, y1 + h
        #get all letters in the word
        #need to sort all the letters by location first...
        for letter, loc in letters:
            if inBound([x1,y1],[x2,y2],loc):
                collectLetters += letter   
        words.append((collectLetters,[x1,y1,w,h],False))
        img = cv2.putText(img,collectLetters,(x1,y1-10), font, 1, (0, 0, 0), 2, cv2.LINE_4)
        img = cv2.rectangle(img,(x1,y1),(x2,y2),(0, 255, 0),3)
    return img, words
  
def sortLetters(letters):
    #https://stackoverflow.com/questions/38654302/how-can-i-sort-contours-from-left-to-right-and-top-to-bottom
    max_height = max(letters,key=lambda x: x[1][3])
    max_height = max_height[1][3]

    # Sort the contours by y-value
    by_y = sorted(letters, key=lambda x: x[1][1])  # y values

    line_y = by_y[0][1][1]       # first y
    line = 1
    by_line = []
    
    # Assign a line number to each contour
    for letter, loc in by_y:
        if loc[1] > line_y + max_height:
            line_y = loc[1]
            line += 1
        by_line.append([line,letter,loc])
            # This will now sort automatically by line then by x

    letters_sorted = [(letter,loc) for line, letter, loc in sorted(by_line, key = lambda x: (x[0], x[2][0]))]

    return letters_sorted

def sortWords(letters):
    #https://stackoverflow.com/questions/38654302/how-can-i-sort-contours-from-left-to-right-and-top-to-bottom
    max_height = max(letters,key=lambda x: x[1][3])
    max_height = max_height[1][3]

    # Sort the contours by y-value
    by_y = sorted(letters, key=lambda x: x[1][1])  # y values

    line_y = by_y[0][1][1]       # first y
    line = 1
    by_line = []
    
    # Assign a line number to each contour
    for letter, loc, t in by_y:
        #Might have to adjust max_height based on scale
        if loc[1] > line_y + max_height:
            line_y = loc[1]
            print(line_y)
            line += 1
        by_line.append([line,letter,loc,t])
    # Sort by line then by x
    letters_sorted = [(letter,loc,t) for line, letter, loc, t in sorted(by_line, key = lambda x: (x[0], x[2][0]))]

    return letters_sorted
   
def pullEquations(img):
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_range = np.array([  0,  67, 128])
    upper_range = np.array([179, 255, 255])
    mask_color = cv2.inRange(hsvImage,lower_range,upper_range)
    
    blur = cv2.medianBlur(mask_color, 3)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    dilation = cv2.dilate(blur, rect_kernel, iterations = 1)
    
    contours, hierarchy = cv2.findContours(dilation,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    equations = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        cropped = img[y:y+h,x:x+w]
        equations.append((cropped.copy(),[x,y,w,h],True))

        #white out equations (don't use machine learning to predict them)
        img[y:y+h,x:x+w] = [255,255,255]
        
    return img, equations

        
def makePDF(words,title):
    geometry_options = {"tmargin": "5cm", "lmargin": "5cm"}
    doc = Document(geometry_options=geometry_options)

    doc.create(Section(''))
    doc.append(Section('Demo Output'))

    i = 0
    for word in words:
        if word[2]:
            filename = 'eq' + str(i) + '.jpg'
            i += 1
            cv2.imwrite(filename, word[0])
            f = Figure(data=None, position='h!')
            f.add_image(filename=filename, width='10cm')
            doc.append(f)
        else:
            doc.append(word[0] + ' ')

    #Can change name of output doc
    doc.generate_tex('demo_output')
    # Change to directory where pdflatex is located
    #Can also comment this out and only generate TeX file
    subprocess.call("C:\\Users\\laura\\AppData\\Local\\Programs\\MiKTeX\\miktex\\bin\\x64\\pdflatex demo_output.tex")
    os.startfile('demo_output.pdf')
        
##########################################################################
#                              FROM CAMERA                               #
##########################################################################

# cam = cv2.VideoCapture(0)
# EMNIST_model = keras.models.load_model('EMNIST Model')
# Math_model = keras.models.load_model('Math CNN Model')

# while True:

#     #Read from the video feed
#     result, img = cam.read()  

#     #Pull boxed equations from image
#     img, equations = pullEquations(img)

#     #Process image, get contours of individual letters
#     contours, newImg = processImage(img)

#     #Pull contours for words from the image
#     wordContours = getWordBounds(newImg)

#     #Test Data File Dimensions to be resized to
#     dim = (28,28)

#     #crop letters from image
#     letterImages = cropLetters(contours, newImg, dim)

#     #Predict letters based on cropped images
#     letterList =  [letter[0] for letter in letterImages]
#     locationList = [letter[1] for letter in letterImages]

#     letterList = np.asarray(letterList)
#     letterList = cv2.bitwise_not(letterList)
#     predictions = EMNIST_predict(letterList,EMNIST_model,28,28)
#     # predictions = MathCNN_predict(letterList,Math_model,28,28)
#     merged_list = tuple(zip(predictions, locationList))

#     #Sort letters into words and add to full image
#     sortedLetters = sortLetters(merged_list)
#     img, words = printWordsToScreen(sortedLetters,wordContours, img)

#     #Make pdf
#     out = words + equations

#     cv2.imshow('LiveStream', img)  
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # makePDF(sortWords(out),'Demo')    
# cam.release()
# # Destroy all the windows
# cv2.destroyAllWindows()

##########################################################################
#                               FROM IMAGE                               #
##########################################################################

EMNIST_model = keras.models.load_model('EMNIST Model')
Math_model = keras.models.load_model('Math CNN Model')

#Read from the image file
img = cv2.imread('test2.jpg') 

#Pull boxed equations from image
img, equations = pullEquations(img)

#Process image, get contours of individual letters
contours, newImg = processImage(img)

#Pull contours for words from the image
wordContours = getWordBounds(newImg)

#Test Data File Dimensions to be resized to
dim = (28,28)

#crop letters from image
letterImages = cropLetters(contours, newImg, dim)

#Predict letters based on cropped images
letterList =  [letter[0] for letter in letterImages]
locationList = [letter[1] for letter in letterImages]

letterList = np.asarray(letterList)
letterList = cv2.bitwise_not(letterList)
predictions = EMNIST_predict(letterList,EMNIST_model,28,28)
# predictions = MathCNN_predict(letterList,Math_model,28,28)
merged_list = tuple(zip(predictions, locationList))

#Sort letters into words and add to full image
sortedLetters = sortLetters(merged_list)
img, words = printWordsToScreen(sortedLetters,wordContours, img)

#Make pdf
out = words + equations
# makePDF(sortWords(out),'Demo')

show_image(img)

