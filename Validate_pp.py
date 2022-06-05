import numpy as np
import os
import cv2
from keras import regularizers
from tensorflow import keras
import matplotlib.pyplot as plt


class Bounds:
    max_h = 0
    max_w = 0
    min_h = 1080
    min_w = 1920
    Threshold = 50

    def __init__(self, bin_img):
        PointSet = np.argwhere(bin_img).tolist()
        Hset, Wset = map(list,zip(*PointSet))
        self.max_h = max(Hset)
        self.max_w = max(Wset)
        self.min_h = min(Hset)
        self.min_w = min(Wset)

    def get_img(self, img):
        size = max(self.min_h - self.max_h, self.min_w - self.max_w)
        center = [(self.min_h + self.max_h)/2, (self.min_w + self.max_w)/2]
        h2 = int(center[0] - size)
        h1 = max(0, int(center[0] + size))
        w2 = int(center[1] - size)
        w1 = int(center[1] + size)
        return img[h1:h2, w1:w2]


def process(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_bw = cv2.adaptiveThreshold(im_gray, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 99, 49)
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    mask = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    b = Bounds(mask)
    return b.get_img(im_gray)


def combine_gen(*gens):
    while True:
        for g in gens:
            yield next(g)


def predict_image(img1):
    img2 = cv2.resize(img1, (360, 360))
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB).astype(np.float32)
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
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(360, 360, 3)))
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

    model.load_weights(r'Models/weights_pp_hr')

    print("Validating")

    imagesF = load_images_from_folder(r'C:\Users\skriv\PycharmProjects\MDB1\Validation data\Flipped')
    imagesNF = load_images_from_folder(r'C:\Users\skriv\PycharmProjects\MDB1\Validation data\NotFlipped')

    Wrong = 0
    Correct = 0
    Idk = 0
    for imageNF in imagesNF:
        im = process(imageNF)
        #plt.imshow(im)
        plt.show()
        res = predict_image(im)
        if res == 1:
            Correct += 1
        elif res == 0:
            Wrong += 1
        else:
            Idk += 1

    for imageF in imagesF:
        res = predict_image(process(imageF))
        if res == 0:
            Correct += 1
        elif res == 1:
            Wrong += 1
        else:
            Idk += 1

    print(f"Wrong = {Wrong}, Correct = {Correct} and Idk = {Idk}")

