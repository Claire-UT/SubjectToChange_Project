# Import required packages
import cv2
import numpy as np
 
# Read image from which text needs to be extracted
img = cv2.imread("DavidTest.jpg")
cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Preprocessing the image starts

#Smooth the image
kernel = np.ones((5,5),np.float32)/25
smooth = cv2.filter2D(img,-1,kernel)
cv2.imshow('Smooth', smooth)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
# Convert the image to gray scale
gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
cv2.imshow('thresh1',thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
 
# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
cv2.imshow('Dilation',dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)
 
# Creating a copy of image
im2 = img.copy()

#Test Data File Dimensions to be resized to
dim = (28,28)
  
# Looping through the identified contours
# Then rectangular part is cropped and passed on
# to pytesseract for extracting text from it
# Extracted text is then written into the text file
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    
    if w < (h):
        x = int(x - ((h-w)/2))
        
    #Make it square by using the max of width, height
    w,h = max(w,h), max(w,h)
    
    #x = int(x - (0.1*x))
    
    #cropped = thresh1[y:y+ int(h*.9),x:x+ int(w*.9)]
    invthresh = cv2.bitwise_not(thresh1)
    cropped = invthresh[y:y+h,x:x+w]
    cv2.imshow('Pre-re dim',cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    resized_cropped = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)
    Big = cv2.resize(cropped, (280,280),interpolation = cv2.INTER_AREA)
    
    # # Drawing a rectangle on copied image
    # rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
     
    # # Cropping the text block for giving input to OCR
    # cropped = im2[y:y + h, x:x + w]
    
    cv2.imshow('Cropped', Big)
    print(resized_cropped.astype('int'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

