import cv2 as cv
import numpy as np 
import flask_app.app as app

# using this to make sure that the image actually displays
import matplotlib.pyplot as plt

# load open cv's cascade, open the image,change it to grayscale, and crop the image out
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# camera = app.success()
# image = cv.imread(camera)
image = cv.imread('flask_app/image.bmp')
gray_img = cv.cvtColor(image, cv.IMREAD_GRAYSCALE)

faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

cropped_img = gray_img[y:y+h, x:x+w]
plt.imshow(cropped_img)
cv.waitKey(0)

plt.savefig('./cropped_results/out.png', dpi=100)
plt.show()
plt.draw()
