# Import required packages
import cv2
import numpy as np
import keras
from Predictions_ModelTests import *

def show_image(image):
    cv2.imshow('image',img)
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
    
    
    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_NONE)
    return contours, thresh1

def getWordBounds(image):
    #Blur threshold to get rid of any dots
    blur = cv2.medianBlur(image, 3)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    dilation2 = cv2.dilate(blur, rect_kernel, iterations = 4)
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
        # Big = cv2.resize(mask, (280,280),interpolation = cv2.INTER_AREA)
        
        print(resized_cropped.astype('int'))
        
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
        words.append([collectLetters,[x1,y1]])
        img = cv2.putText(img,collectLetters,(x1,y1-10), font, 1, (0, 0, 0), 2, cv2.LINE_4)
        img = cv2.rectangle(img,(x1,y1),(x2,y2),(0, 255, 0),3)
    return img, words
  
def sortLetters(letters):
    #https://stackoverflow.com/questions/38654302/how-can-i-sort-contours-from-left-to-right-and-top-to-bottom
    max_height = max(letters,key=lambda x: x[1][3])

    # Sort the contours by y-value
    by_y = sorted(letters, key=lambda x: x[1][1])  # y values

    line_y = by_y[0][1]       # first y
    line = 1
    by_line = []
    
    # Assign a line number to each contour
    for x, y, w, h in by_y:
        if y > line_y + max_height:
            line_y = y
            line += 1
            by_line.append((line, x, y, w, h))
            # This will now sort automatically by line then by x
            letters_sorted = [(x, y, w, h) for line, x, y, w, h in sorted(by_line)]
    # for x, y, w, h in contours:
    #     print(f"{x:4} {y:4} {w:4} {h:4}")
   
def pullEquations(img):
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower_red = np.array([160,20,70])
    # upper_red = np.array([190,255,255])
    # mask1 = cv2.inRange(img, lower_red,upper_red)
    # contours, hierarchy = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.imshow('LiveStream',mask1)

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
        
        # invthresh = cv2.bitwise_not(thresh1)
        cropped = img[y:y+h,x:x+w]
        
        equations.append((cropped,[x,y]))

        #white out equations (don't use machine learning to predict them)
        img[y:y+h,x:x+w] = [255,255,255]
        
    return img, equations

        
# #FROM CAMERA
# cam = cv2.VideoCapture(0)
# EMNIST_model = keras.models.load_model('EMIST Model')

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
#     predictions = EMNIST_predict(letterList,EMNIST_model,28,28)
#     merged_list = tuple(zip(predictions, locationList))
    
#     #Sort letters into words and add to full image
#     img, words = printWordsToScreen(predictions,wordContours, img)
    
#     #Make pdf
    
#     cv2.imshow('LiveStream', img)  

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# cam.release()
# # Destroy all the windows
# cv2.destroyAllWindows()

#####################################

#FROM IMAGE

EMNIST_model = keras.models.load_model('EMNIST Model')

#Read from the video feed
img = cv2.imread('DavidTest.jpg') 

#Pull boxed equations from image
# img, equations = pullEquations(img)

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
predictions = EMNIST_predict(letterList,EMNIST_model,28,28)
merged_list = tuple(zip(predictions, locationList))

#Sort letters into words and add to full image
img, words = printWordsToScreen(merged_list,wordContours, img)

#Make pdf

show_image(img)


