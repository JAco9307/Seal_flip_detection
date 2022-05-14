import numpy as np
import scipy
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

    # model.fit_generator(train_dataset, steps_per_epoch=20, epochs=5, validation_data=test_dataset)
    history = model.fit(train_dataset, steps_per_epoch=10, batch_size=30, epochs=25, verbose=1, validation_data=test_dataset, validation_steps=800)

    def predict_image(filename):
        img1 = image.load_img(filename, target_size=(256, 256))

        img2 = image.load_img(filename, target_size=(540, 960))

        plt.imshow(img2)

        Y = image.img_to_array(img1)

        X = np.expand_dims(Y, axis=0)
        val = model.predict(X)
        print(val)
        if val == 1:

            plt.xlabel("Not Flipped", fontsize=30)

        elif val == 0:

            plt.xlabel("Flipped", fontsize=30)

    predict_image(r"C:/Users/skriv/PycharmProjects/MDB1/Training_w/Test_white/Flipped/TWTF_1.jpg")
    predict_image(r"C:/Users/skriv/PycharmProjects/MDB1/Training_w/Test_white/NotFlipped/TWTN_1.jpg")


