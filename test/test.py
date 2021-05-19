import os
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics.pairwise import cosine_similarity

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

    network.summary()

    return network


img_row = general.img_row
img_col = general.img_col
fix_size_img = (img_row, img_col)
input_shape = (img_row, img_col, 3)
embedding_size = general.embedding_size

path_test = '../data/data_test_training/notme'
image_paths = [os.path.join(path_test, f) for f in os.listdir(path_test)]

inputs = []
vectors = []

for image_path in image_paths:
    face_image = Image.open(image_path)

    # resize the image
    face_numpy = np.array(face_image, 'uint8')
    face_fix_size = cv2.resize(face_numpy, fix_size_img)
    input_test = np.array(face_fix_size)

    inputs.append(input_test)
    # vectors.append(model.predict(input_test))
inputs = np.array(inputs).reshape(-1, img_row, img_col, 3)

model = build_network(input_shape, embedding_size)

# predict with not weight
checkpoint_path = '../model_triplet_1505_1/model.ckpt'

# Loads the weights
model.load_weights(checkpoint_path)
outputs = model.predict(inputs)

vector_1 = outputs[0]
vector_2 = outputs[1]
vector_3 = outputs[2]
vector_4 = outputs[3]
vector_5 = outputs[4]

inputs = []
vectors = []
path_test = '../data/data_test_training/me'
image_paths = [os.path.join(path_test, f) for f in os.listdir(path_test)]
for image_path in image_paths:
    face_image = Image.open(image_path)
    # resize the image
    face_numpy = np.array(face_image, 'uint8')
    face_fix_size = cv2.resize(face_numpy, fix_size_img)
    input_test = np.array(face_fix_size)

    inputs.append(input_test)
    # vectors.append(model.predict(input_test))
inputs = np.array(inputs).reshape(-1, img_row, img_col, 3)
outputs = model.predict(inputs)
sum_vec = outputs[0]
for i in range(1, len(outputs)):
    sum_vec += outputs[i]

sum_vec = sum_vec/len(outputs)
me_1 = np.array(sum_vec)

print(cosine_similarity(vector_1.reshape(1, -1), me_1.reshape(1, -1))[0, 0], '===not me')
print(cosine_similarity(vector_2.reshape(1, -1), me_1.reshape(1, -1))[0, 0], '===not me')
print(cosine_similarity(vector_3.reshape(1, -1), me_1.reshape(1, -1))[0, 0], '===not me')
print(cosine_similarity(vector_4.reshape(1, -1), me_1.reshape(1, -1))[0, 0], '===not me')
print(cosine_similarity(vector_5.reshape(1, -1), me_1.reshape(1, -1))[0, 0], '===not me')

print("=====================================================================")

# # =============================================================================================================
inputs = []
vectors = []
path_test = '../data/data_test_training/me_2'
image_paths = [os.path.join(path_test, f) for f in os.listdir(path_test)]
for image_path in image_paths:
    face_image = Image.open(image_path)
    # resize the image
    face_numpy = np.array(face_image, 'uint8')
    face_fix_size = cv2.resize(face_numpy, fix_size_img)
    input_test = np.array(face_fix_size)

    inputs.append(input_test)
    # vectors.append(model.predict(input_test))
inputs = np.array(inputs).reshape(-1, img_row, img_col, 3)
outputs = model.predict(inputs)
sum_vec = outputs[0]
for i in range(1, len(outputs)):
    sum_vec += outputs[i]

sum_vec = sum_vec/len(outputs)
me_2 = np.array(sum_vec)

print(cosine_similarity(vector_1.reshape(1, -1), me_2.reshape(1, -1))[0, 0], '===not me')
print(cosine_similarity(vector_2.reshape(1, -1), me_2.reshape(1, -1))[0, 0], '===not me')
print(cosine_similarity(vector_3.reshape(1, -1), me_2.reshape(1, -1))[0, 0], '===not me')
print(cosine_similarity(vector_4.reshape(1, -1), me_2.reshape(1, -1))[0, 0], '===not me')
print(cosine_similarity(vector_5.reshape(1, -1), me_2.reshape(1, -1))[0, 0], '===not me')
print(cosine_similarity(vector_5.reshape(1, -1), me_2.reshape(1, -1))[0, 0], '===not me')
print(cosine_similarity(me_1.reshape(1, -1), me_2.reshape(1, -1))[0, 0], '===me')
print("=====================================================================")
