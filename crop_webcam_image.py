import cv2 as cv
import numpy as np 

# using this to make sure that the image actually displays
import matplotlib.pyplot as plt

# load open cv's cascade, open the image,change it to grayscale, and crop the image out
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
image = cv.imread('test.jpeg')
test_image_gray = cv.cvtColor(image, cv.COLOR_BGR2GREY)
plt.imshow(test_image_gray, cmap='gray')

def convertToRGB(image) 
   return cv.cvtColor(image, cv.COLOR_BGR2RGB)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# Display the output
cv2.imshow('img', img)
cv2.waitKey()

# ultimately, crop the image out right here