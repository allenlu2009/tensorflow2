"""
This file was created by Corentin Cheron chronc@tcd.ie
Compute the mean and std of the dataset if absent
Create hdf5 storage if absent
Return train image generator with augmentation and val image generator without
"""

import os
import h5py

import numpy as np

import tensorflow as tf
from tensorflow import keras
#import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator


mean = np.array([[[122.4626756, 114.25840613, 101.37467571]]],
                dtype=np.float32)
std = np.array([[[70.63153376, 68.6114437, 71.93088608]]],
               dtype=np.float32)


def create_hdf5_from_folder(folder, hdf5_path, parameters):
    print("creating hdf5 file:{}".format(hdf5_path))
    tmp_generator = ImageDataGenerator().flow_from_directory(
        folder, shuffle=False, batch_size=1, target_size=parameters.input_size)
    n_classes = tmp_generator.num_classes
    imgs = np.empty([tmp_generator.n] + list(tmp_generator.image_shape), dtype=np.float32)
    classes = np.zeros((tmp_generator.n,  n_classes), dtype=np.float32)
    i = 0
    for img, cla in tmp_generator:
        imgs[i] = img[0]
        classes[i] = cla[0] # cla is already a binary matrix
        i += 1
        print("loading hdf5 array {}/{}\r".format(i, tmp_generator.n), end="")
        if i >= tmp_generator.n:
            break

    dataset = h5py.File(hdf5_path, 'w')
    dataset.create_dataset('X', data=imgs)
    dataset.create_dataset('Y', data=classes)
    dataset.close()
    print("done")


def get_normalized_image_generators(parameters):
    # 1. load data
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    #train_images, test_images = train_images/255.0, test_images/255.0

    #train_path_hdf5 = "data/train_raw.hdf5"
    #val_path_hdf5 = "data/val_raw.hdf5"
    mean, std = None, None
    #global mean, std
    if mean is None or std is None:
        #tmp_generator = ImageDataGenerator().flow_from_directory(
        #    "./data/train",
        #    shuffle=False,
        #    batch_size=1,
        #    target_size=parameters.input_size)

        #print("loading image to compute normalization...")
        sums = np.zeros((3), dtype=np.int64)
        nump = 0
        for i in range(100000//1000):
            print("mean {}/100\r".format(i + 1), end="")
            #img, val = next(tmp_generator)
            img = train_images[i]
            sums += img.astype(np.int64).sum(axis=(0,1,2))
            nump += img.shape[0] * img.shape[1] * img.shape[2]
        mean = (sums.astype(np.float64) / nump).reshape(1, 1, 3)
        print("mean={}".format(mean))

        # compute std
        sums = np.zeros((3), dtype=np.float64)
        nump = 0
        for i in range(100000//1000):
            print("std {}/100\r".format(i + 1), end="")
            #img, val = next(tmp_generator)
            img = train_images[i]
            sums += np.square((img.astype(np.float64) - mean)).sum(axis=(0,1,2))
            nump += img.shape[0] * img.shape[1] * img.shape[2]
        std = np.sqrt((sums.astype(np.float64) / nump)).reshape(1, 1, 3)
        print("std={}".format(std))

    #if not os.path.exists(train_path_hdf5):
    #    create_hdf5_from_folder("data/train/", train_path_hdf5, parameters)

    #if not os.path.exists(val_path_hdf5):
    #    create_hdf5_from_folder("data/val/", val_path_hdf5, parameters)

    #print("loading hdf5 file:{}".format(train_path_hdf5))
    #h5f1 = h5py.File(train_path_hdf5, 'r')
    #train_x = h5f1['X']  # [:] -> force loading
    #train_y = h5f1['Y']
    #print("done")
    train_x = train_images
    train_y = train_labels
    strength = parameters.augmentation_strength
    train_data_generator = ImageDataGenerator(featurewise_std_normalization=True,
                                         featurewise_center=True,
                                         horizontal_flip=True,
                                         width_shift_range=0.15 * strength,
                                         height_shift_range=0.15 * strength,
                                         shear_range=.2 * strength,
                                         zoom_range=.15 * strength,
                                         fill_mode='reflect')
    train_data_generator.mean, train_data_generator.std = mean, std
    train_generator = train_data_generator.flow(train_x, train_y,
                                                batch_size=parameters.batch_size)


    #print("loading hdf5 file:{}".format(val_path_hdf5))
    #h5f2 = h5py.File(val_path_hdf5, 'r')
    #val_x = h5f2['X']  # [:] -> force loading
    #val_y = h5f2['Y']
    #print("done")
    val_x = test_images
    val_y = test_labels
    val_generator = ImageDataGenerator(featurewise_std_normalization=True,
                                       featurewise_center=True)
    val_generator.mean, val_generator.std = mean, std
    val_generator = val_generator.flow(val_x, val_y,
                                       batch_size=parameters.batch_size)
    return train_generator, val_generator


if __name__ == "__main__":

    parameters = {
        'batch_size': 128,
        'momentum' : 0.9,
        'initial_learning_rate' : 0.01,
        'augmentation_strength' : 0.7,
        'nb_layers' : 2,
        'nb_filter_inc_per_layer' : 32,
        'conv_repetition' : 1,
        'dropout_rate' : 0.12,
        'filter_size' : (3, 3),
        'padding' : "same",
        'kernel_regularizer' : 0.0005,
        'kernel_initialization' : 'he_normal'
    }

    parameters = ObjFromDict(parameters)
    model = mirrornet(parameters)
    model.summary()
