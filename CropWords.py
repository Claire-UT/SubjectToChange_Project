# Import required packages
import cv2
import numpy as np
 
#Read image from which text needs to be extracted
# img = cv2.imread("DavidTest.jpg")
# cv2.imshow('Original', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def show_image(image):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cam = cv2.VideoCapture(0)
result, img = cam.read()

good = False
while not good:
    result, image = cam.read()
    show_image(img)
    if input("good?") == 'y':
        good = True

# Preprocessing the image starts

#Smooth the image
kernel = np.ones((5,5),np.float32)/25
smooth1 = cv2.filter2D(img,-1,kernel)
smooth = cv2.filter2D(smooth1,-1,kernel)
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
#thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow('thresh1',thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()

blur = cv2.medianBlur(thresh1, 3)
cv2.imshow('blur',blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
 
# Applying dilation on the threshold image
dilation = cv2.dilate(blur, rect_kernel, iterations = 1)
dilation2 = cv2.dilate(blur, rect_kernel, iterations = 3)
cv2.imshow('Dilation',dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Dilation2',dilation2)
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
# Then rectangular part is cropped and padded with white space to make square
# This is resized to 28x28 and also shown in 280x280
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
    
    invthresh = cv2.bitwise_not(thresh1)
    cropped = invthresh[y:y+h,x:x+w]

    # copy img image into center of result image
    mask[y_center:y_center+old_image_height,x_center:x_center+old_image_width] = cropped
    
    #Resize square, padded value to input dim and larger viewable 280x280 size
    resized_cropped = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
    Big = cv2.resize(mask, (280,280),interpolation = cv2.INTER_AREA)
    
    #print(resized_cropped.astype('int'))
    #cv2.imshow('28x28 Image', resized_cropped)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imshow('280x280 Image', Big)
    cv2.waitKey(0)
    cv2.destroyAllWindows()