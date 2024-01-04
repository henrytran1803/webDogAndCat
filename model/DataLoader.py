import os
import numpy as np
import pandas as pd
import PIL
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


def loaddata(TRAINING_DIR, VALIDATION_DIR,TEST_DIR):
    # Instantiate the ImageDataGenerator class for both training and validation
    train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
    directory=TRAINING_DIR,
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150)
    )

    validation_datagen = ImageDataGenerator(rescale=1/255)

    validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                batch_size=20,
                                                                class_mode='binary',
                                                                target_size=(150, 150))
    test_datagen = ImageDataGenerator(rescale=1/255)

    test_generator = test_datagen.flow_from_directory(
        directory=TEST_DIR,
        batch_size=20,
        class_mode='binary',
        target_size=(150, 150)
    )
    return train_generator, validation_generator, test_generator