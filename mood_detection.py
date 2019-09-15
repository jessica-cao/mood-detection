import os
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.python.keras import layers
from skimage.color import rgb2lab
import numpy as np
from PIL import Image

model_directory = "/posture/model/first_run.h5"

# loading images
train = []
label = []
test_directory = '/facesdb/test'
train_directory = "/facesdb/training"
for people in os.listdir(test_directory):
    for file in os.listdir(os.path.join(test_directory, people)):
        count = 0
        for img in os.listdir(os.path.join(test_directory, people, file)):
            if count < 7:
                new_img = Image.open(test_directory + '/' + file, 'r')
                resized_img = new_img.resize((64, 64), Image.ANTIALIAS)
                train.append(resized_img)
                label.append(count)
                count += 1

train = np.array(train, dtype=float)
train = 1.0/255*train
np.random.shuffle(train)

# the ML model
model = tf.keras.Sequential()
model.add(layers.InputLayer(input_shape=(28, 28, 3)))
model.add(layers.Conv2D(16, (2, 2), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(32, (2, 2),  padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (2, 2),  padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(62, activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop', loss='mse')

model.load_weights("/mirflickr/model/dog.h5")