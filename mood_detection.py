import os
from pathlib import Path
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow as tf
from tensorflow.python.keras import layers, utils
from skimage.color import rgb2lab
import numpy as np
from PIL import Image

# model_directory = "/facesdb/model/first_run.json"

# loading images
train = []
label = []
train_dir = Path('facesdb/training')
for person in os.listdir(train_dir):
    count = 0
    for img_dir in os.listdir(train_dir / Path(person) / Path('bmp')):
        if count < 7:
            new_img = Image.open(train_dir / Path(person) / Path('bmp') / Path(img_dir), 'r')
            print(train_dir / Path(person) / Path('bmp') / Path(img_dir))
            resized_img = new_img.resize((64, 64), Image.ANTIALIAS)
            resized_img = np.array(resized_img)
            train.append(resized_img)
            label.append(count)
            count += 1

train = np.array(train)
print(train)
label = np.array(label)
train = 1.0/255*train

num_classes = 7
label = utils.to_categorical(label, num_classes)

# the ML model
model = tf.keras.Sequential()
model.add(layers.InputLayer(input_shape=(64, 64, 3)))
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
model.add(layers.Dense(7, activation='softmax'))
model.summary()

model.load_weights("/facesdb/model/first_run.json")

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(train, label, batch_size=25, epochs=150, verbose=1)

# Save weights
model.save_weights("/facesdb/model/first_run.json")

test = []
test_label = []
test_dir = Path('facesdb/test')
for person in os.listdir(test_dir):
    count = 0
    for img_dir in os.listdir(test_dir / Path(person) / Path('bmp')):
        if count < 7 and img_dir == 'oliver.bmp':
            new_img = Image.open(test_dir / Path(person) / Path('bmp') / Path(img_dir), 'r')
            print(test_dir / Path(person) / Path('bmp') / Path(img_dir))
            resized_img = new_img.resize((64, 64), Image.ANTIALIAS)
            resized_img = np.array(resized_img)
            test.append(resized_img)
            test_label.append(count)
            count += 1

test = np.array(test, dtype=float)
test_label = np.array(test_label)
test = 1.0/255*test

test_label = utils.to_categorical(test_label, num_classes)

score = model.evaluate(test, test_label, verbose=1)
print('\n', 'Test accuracy:', score[1])
