import numpy as np
import scipy
from keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


if __name__ == '__main__':
    bs = 16
    train = ImageDataGenerator(rescale=1/255)
    test = ImageDataGenerator(rescale=1/255)

    train_dataset = train.flow_from_directory(r"Training_W/Train_white",
                                              target_size=(256, 256), batch_size=bs, class_mode='binary')
    test_dataset = test.flow_from_directory(r"Training_W/Test_white",
                                            target_size=(256, 256), batch_size=bs, class_mode='binary')

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

    # from tensorflow.keras.optimizers import SGD
    # opt = SGD(lr=0.01)
    # model.compile(loss="categorical_crossentropy", optimizer=opt)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    history = model.fit(train_dataset,
                        steps_per_epoch=(216 // bs),
                        validation_steps=(107 // bs),
                        batch_size=bs,
                        epochs=150,
                        verbose=1,
                        validation_data=test_dataset
                        #callbacks=[es]
                        )

    model.summary()

    model.save_weights(r'Models/weights1')


