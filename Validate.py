import numpy as np
import os
import cv2
from keras import regularizers
from tensorflow import keras
import matplotlib.pyplot as plt


def combine_gen(*gens):
    while True:
        for g in gens:
            yield next(g)


def predict_image(img1):
    img2 = cv2.resize(img1, (640, 360))
    img2 = img2[..., ::-1].astype(np.float32)
    X = np.expand_dims(img2, axis=0)
    val = model.predict(X)
    return val


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


if __name__ == '__main__':

    model = keras.Sequential()

    # Convolutional layer and maxpool layer 1
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(360, 640, 3)))
    model.add(keras.layers.MaxPool2D(2, 2))

    # Convolutional layer and maxpool layer 2
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    #Convolutional layer and maxpool layer 3
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    # Convolutional layer and maxpool layer 4
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    # This layer flattens the resulting image array to 1D array
    model.add(keras.layers.Flatten())

    # Hidden layer with 512 neurons and Rectified Linear Unit activation function
    model.add(keras.layers.Dense(256, activation='relu',
                                 kernel_regularizer=regularizers.l2(0.01)))

    # Here we use sigmoid activation function which makes our model output to lie between 0 and 1
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.load_weights(r'Models/weights_w_wood2')

    print("Validating")

    imagesF = load_images_from_folder(r'C:\Users\skriv\PycharmProjects\MDB1\Validation data\Flipped')
    imagesNF = load_images_from_folder(r'C:\Users\skriv\PycharmProjects\MDB1\Validation data\NotFlipped')

    Wrong = 0
    Correct = 0
    Idk = 0
    for imageNF in imagesNF:
        res = predict_image(imageNF)
        if res == 1:
            Correct += 1
        elif res == 0:
            Wrong += 1
        else:
            Idk += 1

    for imageF in imagesF:
        res = predict_image(imageF)
        if res == 0:
            Correct += 1
        elif res == 1:
            Wrong += 1
        else:
            Idk += 1

    print(f"Wrong = {Wrong}, Correct = {Correct} and Idk = {Idk}")

