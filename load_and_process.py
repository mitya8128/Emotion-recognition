import pandas as pd
import cv2
from skimage import io
import numpy as np
import os

detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
yale_dataset_path = 'yalefaces'
dataset_path = 'fer2013/fer2013/fer2013.csv'
image_size=(48,48)

def load_fer2013():
        data = pd.read_csv(dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'),image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        return faces, emotions

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def load_yale_face_db():
    face_detection = cv2.CascadeClassifier(detection_model_path)
    imageData = []
    imageLabels = []
    yale_dataset = []

    for i in yale_dataset_path:
        imgRead = io.imread(i, as_grey=True)
        imageData.append(imgRead)

        #labelRead = int(os.path.split(i)[1].split(".")[0].replace("subject", "")) - 1
        #imageLabels.append(labelRead)

        for i in imageData:
            facePoints = face_detection.detectMultiScale(i)
            x, y = facePoints[0][:2]
            cropped = i[y: y + 48, x: x + 48]
            yale_dataset.append(cropped)

    return yale_dataset