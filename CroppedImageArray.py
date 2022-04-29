import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=np.inf)

def show_image(image):
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
# initialize camera
cam = cv2.VideoCapture(0)    

result, image = cam.read()

good = False
while not good:
    result, image = cam.read()
    show_image(image)
    if input("good?") == 'y':
        good = True

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, im = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

blank = np.zeros(image.shape, dtype='uint8')
new_img = cv2.drawContours(blank,contours, -1, (0,0,0))
cropped_images = []
predictions = []
resized = []

# dimension of image for machine learning:
dim = (28,28)

for contour in contours:
    (x,y,w,h) = cv2.boundingRect(contour)
    # for now, try just making it square by using the max of width, height
    w,h = max(w,h), max(w,h)
    crp = im[y:y+h,x:x+w]
    #resize to correct dimensions
    resized_image = cv2.resize(thresh, dim,interpolation = cv2.INTER_AREA)
    cropped_images.append((im[y:y+h,x:x+w],[x,y,w,h]))
    resized.append(resized_image)

cropped = thresh[y:y+h,x:x+w]
resized_cropped = cv2.resize(cropped, dim,interpolation = cv2.INTER_AREA)
# print(cropped.astype('int'))
# show_image(cropped)

# print(resized_image.astype('int'))
# show_image(resized_image)

print(resized_cropped.astype('int'))
show_image(resized_cropped)



