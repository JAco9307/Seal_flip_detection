import numpy as np
import cv2
from keras import regularizers
from tensorflow import keras
import matplotlib.pyplot as plt
import time

def combine_gen(*gens):
    while True:
        for g in gens:
            yield next(g)


if __name__ == '__main__':
    shape = (256, 256)
    model = keras.Sequential()

    # Convolutional layer and maxpool layer 1
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
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

    # Output layer with single neuron which gives 0 for Cat or 1 for Dog
    # Here we use sigmoid activation function which makes our model output to lie between 0 and 1
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.load_weights(r'Models/weights1')

    model.summary()

    def predict_image(img1):
        img1 = img1[..., ::-1]
        #plt.imshow(img1)
        #plt.axis('off')

        img2 = cv2.resize(img1, shape)
        img2 = img2[..., ::-1].astype(np.float32)

        X = np.expand_dims(img2, axis=0)
        val = model.predict(X)

        print(val)
        if val == 1:
            plt.title("Not Flipped", fontsize=20)
        elif val == 0:
            plt.title("Flipped", fontsize=20)
        #plt.show()

    while True:
        inp = input('Waiting for input')
        if inp == 'exit':
            break
        print("Processing")
        start = time.time()
        cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        result, image = cam.read()
        cam.release()
        if result:
            predict_image(image)

        else:
            print("No image detected. Please! try again")
        end = time.time()
        print(end - start)
