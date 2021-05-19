import os
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import general

tf.keras.backend.clear_session()


def build_network(input_shape, embedding_size):
    '''
    Define the neural network to learn image similarity
    Input :
            input_shape : shape of input images
            embeddingsize : vectorsize used to encode our picture
    '''

    # Convolutional Neural Network
    network = Sequential()
    network.add(
        layers.Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    network.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    network.add(layers.Dropout(0.2))

    network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    network.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    network.add(layers.Dropout(0.2))

    network.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    network.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    network.add(layers.Dropout(0.2))

    network.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    network.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    network.add(layers.Dropout(0.2))

    network.add(layers.Flatten())
    network.add(layers.Dense(units=4096, activation='relu'))
    network.add(layers.Dropout(0.2))
    network.add(layers.Dense(units=1024, activation='relu'))
    network.add(layers.Dropout(0.2))
    network.add(layers.Dense(units=512, activation='relu'))
    network.add(layers.Dropout(0.2))
    network.add(layers.Dense(units=embedding_size))

    return network


def get_features(images):
    img_rows = general.img_row
    img_cols = general.img_col
    embedding_size = general.embedding_size

    fix_size_img = (img_rows, img_cols)
    input_shape = (img_rows, img_cols, 3)

    inputs = []

    for image_path in images:
        face_image = Image.open(image_path)

        # resize the image
        face_numpy = np.array(face_image, 'uint8')
        face_fix_size = cv2.resize(face_numpy, fix_size_img)
        input_test = np.array(face_fix_size)

        inputs.append(input_test)

    inputs = np.array(inputs).reshape(-1, img_rows, img_cols, 3)

    model = build_network(input_shape, embedding_size)

    # predict with not weight
    checkpoint_path = general.model_triplet_path

    # Loads the weights
    model.load_weights(checkpoint_path)
    outputs = model.predict(inputs)

    # compute averaging the vectors
    sum_vec = outputs[0]
    for i in range(1, len(outputs)):
        sum_vec += outputs[i]

    average = sum_vec / len(outputs)
    vector_feature = np.array(average)

    return vector_feature
