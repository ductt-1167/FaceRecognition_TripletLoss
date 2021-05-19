import cv2
import numpy as np
import re
import os
import shutil
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import general
import extract_feature


def convert_str_array(list_str):
    result = []
    for vector_str in list_str:
        vector = np.zeros(general.embedding_size)
        middle = re.sub("\[", "", vector_str)
        middle = re.sub("]", "", middle)
        count = 0
        for number in middle.split():
            vector[count] = float(number)
            count += 1

        result.append(vector)

    return np.array(result)


def compute_similarity():
    pass


# load face cascade and eyes cascade
cascade_face_path = general.cascade_face_path

file = pd.read_csv(general.user_information)
list_id_information = file['id'].values
list_name_information = file['Name'].values

file = pd.read_csv(general.feature_data)
list_id_feature = file['id'].values
list_vector_feature = file['vector_feature'].values
vector_feature_array = convert_str_array(list_vector_feature)

# capture object from camera
cam = cv2.VideoCapture(0)

# using haar cascade
face_cascade = cv2.CascadeClassifier(cascade_face_path)

font = cv2.FONT_HERSHEY_SIMPLEX
sample_num = 0
total_sample = 50  # total_sample is the number of picture for each user to check

folder_cache = "data/data_cache"

os.mkdir(folder_cache)

while True:
    # capture video frame by frame
    ret, frame = cam.read()

    #  the frame, converted to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        sample_num += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 0)

        # save face images
        cv2.imwrite(folder_cache + '/' + 'pic_' + str(sample_num) + ".jpg", frame[y:y + h, x:x + w])

        cv2.putText(frame, "Please gentle movement and not look up", (x, y), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # display the result
    cv2.imshow('Login face', frame)

    # the 'q' button is set as quitting button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if sample_num > total_sample:
        break

cam.release()
cv2.destroyAllWindows() 

list_images = [os.path.join(folder_cache, f) for f in os.listdir(folder_cache)]
vector_feature = extract_feature.get_features(list_images)

id_predict = []
predict = []
p = 0.9     # rating index
for index in range(len(vector_feature_array)):
    print(list_id_feature[index], cosine_similarity(vector_feature.reshape(1, -1), vector_feature_array[index].reshape(1, -1))[0, 0])
    if cosine_similarity(vector_feature.reshape(1, -1), vector_feature_array[index].reshape(1, -1))[0, 0] >= p:
        id_predict.append(list_id_feature[index])
        predict.append(cosine_similarity(vector_feature.reshape(1, -1), vector_feature_array[index].reshape(1, -1))[0, 0])

if len(id_predict) == 0:
    print("Sorry, Can not recognize you!")
else:
    max_predict = max(predict)
    print("Welcome ", list_name_information[list_id_information.tolist().index(id_predict[predict.index(max_predict)])],
          'with ', max_predict*100, ' %')
    # for i in range(len(id_predict)):
    #     index = list_id_information.tolist().index(id_predict[i])
    #     print(list_name_information[index], 'with ', predict[i]*100, ' %')

# remove folder cache
shutil.rmtree(folder_cache)

