import cv2
import csv
import os
import pandas as pd

import general
import extract_feature


# load face cascade and eyes cascade
data_information = general.user_information
cascade_face_path = general.cascade_face_path

file = pd.read_csv(data_information)
list_id = file['id'].values


# register information
def register_information():
    name_user = input('Enter your name: ')
    id_user = input('Enter your id: ')

    # check the ID is exist?
    while int(id_user) in list_id:
        print("This ID is used! Please create a different ID!")
        name_user = input('Enter your name: ')
        id_user = input('Enter your ID: ')

    # save data user in csv file
    data = name_user + ',' + id_user
    data = data.split(',')
    with open(data_information, 'a+') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(data)

    return id_user


# register for user
id = register_information()

# capture object from camera
cam = cv2.VideoCapture(0)

# using haar cascade
face_cascade = cv2.CascadeClassifier(cascade_face_path)

font = cv2.FONT_HERSHEY_SIMPLEX
sample_num = 0
total_sample = 50  # total_sample is the number of picture for each user to train

folder_faces = "data/data_register/user_" + str(id)

try:
    os.mkdir(folder_faces)

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
            cv2.imwrite(folder_faces + '/' + str(id) + '_' + str(sample_num) + ".jpg", frame[y:y + h, x:x + w])

            cv2.putText(frame, "Please gentle movement and not look up", (x, y), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # display the result
        cv2.imshow('Register face', frame)

        # the 'q' button is set as quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if sample_num > total_sample:
            break

    cam.release()
    cv2.destroyAllWindows()

    # get feature vector
    list_images = [os.path.join(folder_faces, f) for f in os.listdir(folder_faces)]
    vector_feature = extract_feature.get_features(list_images)

    # save data in file csv
    with open(general.feature_data, 'a+') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([str(id), vector_feature])

except:
    print("Exist this user!")




