import tensorflow as tf
import numpy as np
from tensorflow.python import keras
import os
import cv2
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing import image
import matplotlib.pyplot as plt


if __name__ == '__main__':
    train_dataset = train.flow_from_directory("C:/Users/skriv/PycharmProjects/MDB1/Samples_images")
    test_dataset = train.flow_from_directory("C:/Users/skriv/PycharmProjects/MDB1/Test_images")
