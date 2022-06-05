import numpy as np
import scipy
from keras import regularizers
from keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


def combine_gen(*gens):
    while True:
        for g in gens:
            yield next(g)


if __name__ == '__main__':
    bs = 10
    shape = (360, 360)
    inc_wood = False
    train = ImageDataGenerator(rescale=1 / 255)
    test = ImageDataGenerator(rescale=1 / 255)

    train_dataset = train.flow_from_directory(r"Training_p/Train_white",
                                              target_size=shape, batch_size=bs, class_mode='binary')

    test_dataset = test.flow_from_directory(r"Training_p/Test_white",
                                            target_size=shape, batch_size=bs, class_mode='binary')

    if inc_wood:
        train_dataset_W = train.flow_from_directory(r"Training_W_W/Train_white",
                                                    target_size=shape, batch_size=bs, class_mode='binary')
        train_dataset = combine_gen(train_dataset, train_dataset_W)

        test_dataset_W = test.flow_from_directory(r"Training_W_W/Test_white",
                                                  target_size=shape, batch_size=bs, class_mode='binary')
        test_dataset = combine_gen(test_dataset, test_dataset_W)

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

    # Output layer with single neuron which gives 0 for Cat or 1 for Dog
    # Here we use sigmoid activation function which makes our model output to lie between 0 and 1
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=0.005), metrics='accuracy')

    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20) this sucks

    history = model.fit(train_dataset,
                        steps_per_epoch=(len(train_dataset) // bs),
                        validation_steps=(len(test_dataset) // bs),
                        batch_size=bs,
                        epochs=300,
                        verbose=1,
                        validation_data=test_dataset
                        )

    model.summary()

    model.save_weights(r'Models/weights_pp_hr')


