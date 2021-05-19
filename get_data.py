import numpy as np
import os
import cv2
from PIL import Image

import general


size_img_1 = general.img_row
size_img_2 = general.img_col


# get the image array and id array as data training and validation
def get_matrix(path):
    faces = []
    id_class = 0
    fix_size_img = (size_img_1, size_img_2)
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]

    for image_path in image_paths:
        list_image = [os.path.join(image_path, f) for f in os.listdir(image_path)]
        each_face = []

        # each face class
        for each_image in list_image:
            face_image = Image.open(each_image)

            # resize the image
            face_numpy = np.array(face_image, 'uint8')
            face_fix_size = cv2.resize(face_numpy, fix_size_img)
            each_face.append(face_fix_size)

        faces.append(np.array(each_face))
        id_class += 1

    list_id = np.arange(id_class)
    return list_id, np.array(faces), id_class


def create_random_list(lengh, size):
    result = []
    for i in range(lengh):
        chose = np.random.randint(lengh)
        while chose in result:
            chose = np.random.randint(lengh)

        result.append(chose)
        if len(result) == size:
            return np.array(result)


def get_data(image_paths):
    ids, data, number_classes = get_matrix(image_paths)
    data_full = []

    for each_per in range(len(data)):

        faces_each_per = data[each_per]
        length = len(faces_each_per)

        # Get random index to choose negative
        length_negative_each_class = (length - 1) // number_classes + 1

        random_class = create_random_list(number_classes, number_classes)
        random_class = random_class.tolist()
        random_class.remove(ids[each_per])

        count_class = 0
        count_face_in_class = 0

        for index_face in range(length - 1):
            anchor_positive_negative = [np.array(faces_each_per[index_face]).reshape((size_img_1, size_img_2, 3)),
                                        np.array(faces_each_per[index_face + 1]).reshape((size_img_1, size_img_2, 3))]

            # anchor
            # positive
            # negative
            class_negative = data[random_class[count_class]]

            list_index_choose_negative = create_random_list(len(class_negative), length_negative_each_class)

            anchor_positive_negative.append(np.array(class_negative[list_index_choose_negative[count_face_in_class]])
                                            .reshape(size_img_1, size_img_2, 3))

            data_full.append(np.array(anchor_positive_negative))

            count_face_in_class += 1
            if count_face_in_class == len(list_index_choose_negative):
                count_face_in_class = 0
                count_class += 1
                if count_class == len(random_class):
                    break

    return np.array(data_full)


def get_triplets_batch(n):
    data_full = get_data(general.images_path)
    length = len(data_full)
    random_index = create_random_list(length, n)
    idxs_a, idxs_p, idxs_n = [], [], []
    for index in random_index:
        a, p, n = data_full[index][0], data_full[index][1], data_full[index][2]
        idxs_a.append(a)
        idxs_p.append(p)
        idxs_n.append(n)
    return np.array(idxs_a), np.array(idxs_p), np.array(idxs_n)
