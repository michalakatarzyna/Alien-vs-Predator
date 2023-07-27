!pip install opendatasets
!pip install gradio

import os
import glob
import random
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, util, color
from PIL import Image
import cv2

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential, load_model, Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ProgbarLogger
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.datasets import mnist
from keras.applications import ResNet50
from keras.utils import plot_model

import opendatasets as od
import gradio as gr

!pip install --upgrade tensorflow
!pip install --upgrade gradio

"""#Dataset
Use datasets from the kaggle.
"""

dataset_url = 'kaggle.com/datasets/pmigdal/alien-vs-predator-images/data'
od.download(dataset_url)

train_path = '/content/alien-vs-predator-images/data/train'
valid_path = '/content/alien-vs-predator-images/data/validation'

class_names = sorted(os.listdir(train_path))

class_counts = {}

for class_name in class_names:
    class_dir = os.path.join(train_path, class_name)
    file_count = len(os.listdir(class_dir))
    class_counts[class_name] = file_count

for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

"""##Check the quality of the data"""

class_counts = []

for class_name in class_names:
    class_dir = os.path.join(train_path, class_name)
    file_count = len(os.listdir(class_dir))
    class_counts.append(file_count)

plt.bar(class_names, class_counts)
plt.xlabel('Class')
plt.ylabel('Number of Examples')
plt.title('Class Distribution')
plt.show()

num_examples = 5
random.seed(42)

for class_name in class_names:
    class_dir = os.path.join(train_path, class_name)
    file_names = os.listdir(class_dir)
    random_files = random.sample(file_names, num_examples)

    for file_name in random_files:
        image_path = os.path.join(class_dir, file_name)
        image = Image.open(image_path)
        plt.imshow(image)
        plt.title(class_name)
        plt.axis('off')
        plt.show()

"""##Data preprocessing"""

train_files = os.listdir(train_path)
train, test = train_test_split(train_files, test_size=0.2, random_state=42)

valid_files = os.listdir(valid_path)
valid, test = train_test_split(valid_files, test_size=0.5, random_state=42)

"""##Model definition, training, and evaluation"""

batch_size = 32
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
train_data = train_generator.flow_from_directory(
    train_path,
    target_size=(128, 128),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

valid_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
valid_data = valid_generator.flow_from_directory(
    valid_path,
    target_size=(128, 128),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

"""###grayscale"""

target_size = (224, 224,)
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
    )

train_gray = train_generator.flow_from_directory(
    train_path,
    batch_size=batch_size,
    target_size=target_size,
    class_mode='categorical',
    color_mode='grayscale'
)


valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
    )


valid_gray = valid_datagen.flow_from_directory(
    valid_path,
    batch_size=batch_size,
    target_size = target_size,
    class_mode='categorical',
    color_mode='grayscale'
    )

train_gray[0][0].shape

batch_size = 32
epochs = 10
num_classes = 2

input_shape = (224, 224, 1)

model = Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history_gray = model.fit(
    train_gray,
    steps_per_epoch=train_gray.samples // batch_size,
    epochs=epochs,
    validation_data=valid_gray,
    validation_steps=valid_gray.samples // batch_size)

augmented_images, _ = train_gray.next()

augmented_image = augmented_images[0]

plt.imshow(augmented_image)
plt.axis('off')
plt.show()

plt.plot(history_gray.history['loss'])
plt.plot(history_gray.history['val_loss'])
plt.title('Model Loss Plot')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

plt.plot(history_gray.history['accuracy'])
plt.plot(history_gray.history['val_accuracy'])
plt.title('Accuracy Plot')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

"""###rgb"""

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
)

train_rgb = train_datagen.flow_from_directory(
    train_path,
    batch_size=batch_size,
    target_size=target_size,
    class_mode='categorical',
    color_mode='rgb'
)

valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
)

valid_rgb = valid_datagen.flow_from_directory(
    valid_path,
    batch_size=batch_size,
    target_size=target_size,
    class_mode='categorical',
    color_mode='rgb'
)

batch_size = 32
epochs = 10
num_classes = 2
input_shape = (224, 224, 3)

model_2 = keras.Sequential()
model_2.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model_2.add(layers.MaxPooling2D(pool_size=(2, 2)))
model_2.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_2.add(layers.MaxPooling2D(pool_size=(2, 2)))
model_2.add(layers.Flatten())
model_2.add(layers.Dropout(0.5))
model_2.add(layers.Dense(num_classes, activation='softmax'))

model_2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_2.summary()

history_rgb = model_2.fit(
    train_rgb,
    steps_per_epoch=train_rgb.samples // batch_size,
    epochs=epochs,
    validation_data=valid_rgb,
    validation_steps=valid_rgb.samples // batch_size
)

augmented_images, _ = train_rgb.next()


augmented_image = augmented_images[0]

plt.imshow(augmented_image)
plt.axis('off')
plt.show()

plt.plot(history_rgb.history['loss'])
plt.plot(history_rgb.history['val_loss'])
plt.title('Model loss plot')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

plt.plot(history_rgb.history['accuracy'])
plt.plot(history_rgb.history['val_accuracy'])
plt.title('Accuracy plot')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

"""###rotate"""

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    fill_mode='nearest'
)

train_rot = train_datagen.flow_from_directory(
    train_path,
    batch_size=batch_size,
    target_size=target_size,
    class_mode='categorical',
    color_mode='rgb'
)

valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0
)

valid_rot = valid_datagen.flow_from_directory(
    valid_path,
    batch_size=batch_size,
    target_size=target_size,
    class_mode='categorical',
    color_mode='rgb'
)

batch_size = 32
epochs = 10
num_classes = 2
input_shape = (224, 224, 3)

model_3 = keras.Sequential()
model_3.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model_3.add(layers.MaxPooling2D(pool_size=(2, 2)))
model_3.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_3.add(layers.MaxPooling2D(pool_size=(2, 2)))
model_3.add(layers.Flatten())
model_3.add(layers.Dropout(0.5))
model_3.add(layers.Dense(num_classes, activation='softmax'))

model_3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_3.summary()

history_rot = model_3.fit(
    train_rot,
    steps_per_epoch=train_rot.samples // batch_size,
    epochs=epochs,
    validation_data=valid_rot,
    validation_steps=valid_rot.samples // batch_size
)

augmented_images, _ = train_rot.next()
augmented_image = augmented_images[0]

plt.imshow(augmented_image)
plt.axis('off')
plt.show()

plt.plot(history_rot.history['loss'])
plt.plot(history_rot.history['val_loss'])
plt.title('Model loss plot')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

plt.plot(history_rot.history['accuracy'])
plt.plot(history_rot.history['val_accuracy'])
plt.title('Accuracy plot')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

"""###flip"""

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
    horizontal_flip=True,
    vertical_flip=True,
)

train_flip = train_datagen.flow_from_directory(
    train_path,
    batch_size=batch_size,
    target_size=target_size,
    class_mode='categorical',
    color_mode='rgb'
)

valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0
)

valid_flip = valid_datagen.flow_from_directory(
    valid_path,
    batch_size=batch_size,
    target_size=target_size,
    class_mode='categorical',
    color_mode='rgb'
)

batch_size = 32
epochs = 10
num_classes = 2
input_shape = (224, 224, 3)

model_4 = keras.Sequential()
model_4.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model_4.add(layers.MaxPooling2D(pool_size=(2, 2)))
model_4.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_4.add(layers.MaxPooling2D(pool_size=(2, 2)))
model_4.add(layers.Flatten())
model_4.add(layers.Dropout(0.5))
model_4.add(layers.Dense(num_classes, activation='softmax'))

model_4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_4.summary()

history_flip = model_4.fit(
    train_flip,
    steps_per_epoch=train_flip.samples // batch_size,
    epochs=epochs,
    validation_data=valid_flip,
    validation_steps=valid_flip.samples // batch_size
)

augmented_images, _ = train_flip.next()
augmented_image = augmented_images[0]

plt.imshow(augmented_image)
plt.axis('off')
plt.show()

plt.plot(history_flip.history['loss'])
plt.plot(history_flip.history['val_loss'])
plt.title('Wykres straty modelu')
plt.ylabel('Strata')
plt.xlabel('Epoka')
plt.legend(['Trening', 'Walidacja'], loc='upper right')
plt.show()

plt.plot(history_flip.history['accuracy'])
plt.plot(history_flip.history['val_accuracy'])
plt.title('Wykres dokładności')
plt.ylabel('Dokładność')
plt.xlabel('Epoka')
plt.legend(['Trening', 'Walidacja'], loc='lower right')
plt.show()

"""###brightness"""

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
    brightness_range=[0.5,2.0]
)

train_bright = train_datagen.flow_from_directory(
    train_path,
    batch_size=batch_size,
    target_size=target_size,
    class_mode='categorical',
    color_mode='rgb'
)

valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0
)

valid_bright = valid_datagen.flow_from_directory(
    valid_path,
    batch_size=batch_size,
    target_size=target_size,
    class_mode='categorical',
    color_mode='rgb'
)

batch_size = 32
epochs = 10
num_classes = 2
input_shape = (224, 224, 3)

model_5 = keras.Sequential()
model_5.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model_5.add(layers.MaxPooling2D(pool_size=(2, 2)))
model_5.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_5.add(layers.MaxPooling2D(pool_size=(2, 2)))
model_5.add(layers.Flatten())
model_5.add(layers.Dropout(0.5))
model_5.add(layers.Dense(num_classes, activation='softmax'))

model_5.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_5.summary()

history_bright = model_5.fit(
    train_bright,
    steps_per_epoch=train_bright.samples // batch_size,
    epochs=epochs,
    validation_data=valid_bright,
    validation_steps=valid_bright.samples // batch_size
)

augmented_images, _ = train_bright.next()
augmented_image = augmented_images[0]

plt.imshow(augmented_image)
plt.axis('off')
plt.show()

plt.plot(history_bright.history['loss'])
plt.plot(history_bright.history['val_loss'])
plt.title('Model loss plot')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

plt.plot(history_bright.history['accuracy'])
plt.plot(history_bright.history['val_accuracy'])
plt.title('Accuracy plot')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

"""###ResNet50"""

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(num_classes, activation='softmax')(x)
model_6 = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers:
    layer.trainable = False

model_6.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_6.summary()

history_RN50 = model_6.fit(
    train_rgb,
    steps_per_epoch=train_rgb.samples // batch_size,
    epochs=epochs,
    validation_data=valid_rgb,
    validation_steps=valid_rgb.samples // batch_size
)

history_model_1 = history_gray.history
history_model_2 = history_rgb.history
history_model_3 = history_rot.history
history_model_4 = history_flip.history
history_model_5 = history_bright.history
history_model_6 = history_RN50.history


accuracy_values = [
    history_model_1['accuracy'][-1],
    history_model_2['accuracy'][-1],
    history_model_3['accuracy'][-1],
    history_model_4['accuracy'][-1],
    history_model_5['accuracy'][-1],
    history_model_6['accuracy'][-1]
]

model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6']

plt.figure(figsize=(12, 20))


plt.subplot(3, 1, 1)
plt.plot(history_model_1['accuracy'], label='Model 1', color='blue')
plt.plot(history_model_2['accuracy'], label='Model 2', color='orange')
plt.plot(history_model_3['accuracy'], label='Model 3', color='green')
plt.plot(history_model_4['accuracy'], label='Model 4', color='red')
plt.plot(history_model_5['accuracy'], label='Model 5', color='purple')
plt.plot(history_model_6['accuracy'], label='Model 6', color='brown')

plt.title('Accuracy of models')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')


plt.subplot(3, 1, 2)
plt.plot(history_model_1['loss'], label='Model 1', color='blue')
plt.plot(history_model_2['loss'], label='Model 2', color='orange')
plt.plot(history_model_3['loss'], label='Model 3', color='green')
plt.plot(history_model_4['loss'], label='Model 4', color='red')
plt.plot(history_model_5['loss'], label='Model 5', color='purple')
plt.plot(history_model_6['loss'], label='Model 6', color='brown')

plt.title('Loss of models')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.subplot(3, 1, 3)
plt.bar(model_names, accuracy_values, color=['blue', 'orange', 'green', 'red', 'purple', 'brown'])


plt.ylabel('Accuracy')
plt.ylim(0, 1.0)

plt.tight_layout()
plt.show()

"""##Evaluating the best model"""

test_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
)

test_gray = test_datagen.flow_from_directory(
    valid_path,
    batch_size=batch_size,
    target_size=target_size,
    class_mode='categorical',
    color_mode='grayscale'
)

test_loss, test_accuracy = model.evaluate(
    test_gray,
    steps=test_gray.samples // batch_size
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

model_path = '/content/model.h5'
model.save(model_path)


model.save(model_path)

"""#Gradio Interface"""

def preprocess(image):
    image_array = np.array(image)
    target_size = (224, 224)
    processed_image = tf.image.resize(image_array, target_size)
    processed_image = tf.expand_dims(processed_image, axis=0)
    processed_image = processed_image / 255.0

    return processed_image

model = tf.keras.models.load_model('model.h5')

def predict(image):
    processed_image = preprocess(image)
    predictions = model_2.predict(processed_image)

    class_labels = ["Predator", "Alien"]
    predicted_percentages = [round(percentage * 100, 2) for percentage in predictions[0]]
    results = [f"{label}: {percentage}%" for label, percentage in zip(class_labels, predicted_percentages[::-1])]

    return results

iface = gr.Interface(
    fn=predict,
    inputs="image",
    outputs="text",
    title="Model Prediction",
    description="Model for making predictions."
)

iface.launch(share=True)
