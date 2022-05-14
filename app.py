import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


if __name__ == '__main__':
    train = ImageDataGenerator(rescale=1/255)
    test = ImageDataGenerator(rescale=1/255)

    train_dataset = train.flow_from_directory("C:/Users/skriv/PycharmProjects/MDB1/Training_W/Train_white",
                                              target_size=(256, 256), batch_size=10, class_mode='binary')
    test_dataset = test.flow_from_directory("C:/Users/skriv/PycharmProjects/MDB1/Training_W/Test_white",
                                            target_size=(256, 256), batch_size=10, class_mode='binary')

    model = keras.Sequential()

    # Convolutional layer and maxpool layer 1
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(keras.layers.MaxPool2D(2, 2))

    # Convolutional layer and maxpool layer 2
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    # Convolutional layer and maxpool layer 3
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    # Convolutional layer and maxpool layer 4
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    # This layer flattens the resulting image array to 1D array
    model.add(keras.layers.Flatten())

    # Hidden layer with 512 neurons and Rectified Linear Unit activation function
    model.add(keras.layers.Dense(256, activation='relu'))

    # Output layer with single neuron which gives 0 for Cat or 1 for Dog
    # Here we use sigmoid activation function which makes our model output to lie between 0 and 1
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=0.001), metrics='accuracy')

    model.load_weights(r'Models/weights1')

    model.summary()

    def predict_image(img1):
        img2 = cv2.resize(img1, (256, 256))
        img2 = img2[..., ::-1].astype(np.float32)

        img1 = img1[..., ::-1]

        plt.imshow(img1)
        plt.show()

        # Y = image.img_to_array(img2)

        X = np.expand_dims(img2, axis=0)
        val = model.predict(X)
        print(val)
        if val == 1:
            plt.xlabel("Not Flipped", fontsize=30)
        elif val == 0:
            plt.xlabel("Flipped", fontsize=30)

    while True:
        inp = input('Waiting for input')
        if inp == 'exit':
            break
        print("Processing")
        cam = cv2.VideoCapture(0)
        result, image = cam.read()
        if result:
            predict_image(image)

        else:
            print("No image detected. Please! try again")

