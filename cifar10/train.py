from __future__ import print_function

import pysnooper
import argparse
import h5py
import os
import numpy as np
import random

import tensorflow as tf
from tensorflow import keras
#import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.callbacks import TensorBoard

#from tf.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, PReLU
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import top_k_categorical_accuracy as top5

from dataset_loading import get_normalized_image_generators

"""The metrics used while training the models."""
#metrics = ['accuracy', top5, tf.losses.log_loss]
metrics = ['sparse_categorical_accuracy']


"""Utility function to store function and parameters in a dict."""
class ObjFromDict(object):
    def __init__(self, dict):
        self.__dict__.update(dict)


def sgdMomentum090(lr):
    return SGD(lr=lr, momentum=.90)


def sgdMomentum095(lr):
    return SGD(lr=lr, momentum=.95)


def sgdMomentum095nesterov(lr):
    return SGD(lr=lr, momentum=.95, nesterov=True)


def adadeltaDefault(lr):
    return Adadelta(lr=lr)


def rmsprop_szegedy(lr):
    return RMSprop(lr=lr, rho=.9, decay=0.94, clipvalue=2.)


def prelu25():
    return PReLU(
        alpha_initializer=initializers.Constant(value=0.25))

class BaseParameters():
    """The reference parameters.
    They are loaded first and modified by following options:
    independent testing or sequence.
    """
    batch_size = 128
    nb_epochs = 71
    input_size = (32, 32)
    pretrained = False

    initial_learning_rate = 0.005
    lr_patience = 1
    lr_update = .5
    min_lr = .00001
    patience_stop = 5
    model_name = 'mirrornet'
    augmentation_strength = 0.7
    optimizer = sgdMomentum095
    dropout_rate = 0.12
    filter_size = (3, 3)
    nb_layers = 2
    conv_repetition = 1
    nb_filter_inc_per_layer = 32
    padding = "same"
    activation = "relu"
    kernel_initialization = 'he_normal'
    kernel_regularizer = 0.0005
    loss_function = "categorical_crossentropy"
    retrain = False  # do not retrain base model (that have pretrained weights)


class PossibleParameters():
    """List of parameters to test with the independent option
    Each of the following will be test independently by updating the
    corresponding value from the reference parameters.
    """

    batch_size = [128, 256]
    dropout_rate = [0., .25, .5]
    padding = ["valid", "same"]
    filter_size = [(2,2), (4, 4), (5, 5)]
    nb_layers = [3, 4]
    conv_repetition = [2, 3]
    nb_filter_inc_per_layer = [16, 48, 96]
    initial_learning_rate = [.00501] * 2
    activation = [prelu25(), "tanh"]
    kernel_initialization = ['glorot_uniform', 'lecun_uniform']
    loss_function = ["categorical_hinge"]
    optimizer = [ sgdMomentum095nesterov]
    initial_learning_rate = [.01, 0.001, .0001]
    kernel_regularizer = [0., 5e-4, 1e-3]
    augmentation_strength = [0., 1.3]


#@pysnooper.snoop('./pysnooper.log')
def train(parameters):

    # for tiny imagenet dataset
    #img_rows = 32
    #img_cols = 32
    #input_shape = (img_rows, img_cols, 3)

    # normalize image based on mean and std read from data/xxx.hdf5
    # if not readable, generate it
    # corresponding mnist train_image/255, test_image/255

    # data loading and augmentation
    train_generator, val_generator  = get_normalized_image_generators(parameters)

    # import ipdb; ipdb.set_trace()
    # define model
    # the following "hack" loads the function parameters.model_name from the
    # file model.py. It allows defining the model by a string.
    model = getattr(__import__("model"), parameters.model_name)(parameters)
    if hasattr(parameters, "load") and parameters.load:
        weight_file = get_model_weights_file(parameters.load)
        print("Loading weights from: {}".format(parameters.load))
        model.load_weights(weight_file)

    model.summary()

    #model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    #model1.compile(
    #    loss=tf.keras.losses.categorical_crossentropy,
    #    optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    sgd = optimizers.SGD(lr=parameters.initial_learning_rate, momentum=0.9)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=metrics)

    tbCallBack = TensorBoard(log_dir='./logs',
                             histogram_freq=0,
                             #batch_size=32,
                             write_graph=True,
                             write_grads=True,
                             write_images=True,
                             embeddings_freq=0,
                             embeddings_layer_names=None,
                             embeddings_metadata=None)


    
    #model.fit(train_generator, epochs=60)
    model.fit_generator(
        train_generator,
        epochs=parameters.nb_epochs,
        steps_per_epoch=train_generator.n // parameters.batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.n // parameters.batch_size,
        callbacks=[tbCallBack])
    #callbacks=callbacks, shuffle='batch', workers=8,
    #use_multiprocessing=False
    
    
#@pysnooper.snoop('./main.log')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True,
                        help='Name this run, required.')
    parser.add_argument('--random', default=False, action='store_true',
                        help="Run N (10) random parameters sets")
    parser.add_argument('--model', help="The name of the model (function name "
                                        "in model.py)")
    parser.add_argument('--pretrained', default=False, action='store_true',
                        help="If available, load ImageNet weights")
    parser.add_argument('--retrain', default=False, action='store_true',
                        help="If available, load previous weights (useful to "
                             "continue an interrupted training")
    parser.add_argument('--sequence', default=False, action='store_true', help=
                        "Run the sequence specified in train.py")
    parser.add_argument('--load', type=str, help='load previous weights')
    parser.add_argument('--independent', default=False, action='store_true',
                        help="Run all parameters sets, changing all the variab"
                             "les, one at a time")
    parser.add_argument('--nrand', type=int, default=10, help='Number of run')
    args = parser.parse_args()
    print(args)

    if args.random:
        for run_id in range(1, args.nrand + 1):
            parameters = {k: v for k, v in BaseParameters.__dict__.items()
                          if not k.startswith('__')}
            parameters['selectedParameters'] = {}
            for key, values in PossibleParameters.__dict__.items():
                if key.startswith('__'):
                    continue
                param = random.choice(values)
                parameters[key] = param
                parameters['selectedParameters'][key] = param
            parameters = ObjFromDict(parameters)
            parameters.run_name = "{}_{:03d}".format(args.name, run_id)
            print(parameters)
            train(parameters)
    elif args.independent:
        nb_runs = sum([len(v) - 1 for k, v in
                       PossibleParameters.__dict__.items()
                       if not k.startswith('__')]) + 1
        print("nb total runs = {}".format(nb_runs))
        run_id = 1

        # default run
        parameters = {k: v for k, v in BaseParameters.__dict__.items()
                      if not k.startswith('__')}
        parameters['selectedParameters'] = {}
        parameters = ObjFromDict(parameters)
        parameters.run_name = "{}_{:03d}_ref".format(args.name, run_id)
        print(sorted(parameters.selectedParameters.items()))
        train(parameters)
        run_id += 1

        for key, values in sorted(PossibleParameters.__dict__.items()):
            if key.startswith('__'):
                continue

            for value in values:
                if BaseParameters.__dict__[key] == value:
                    continue

                parameters = {k: v for k, v in BaseParameters.__dict__.items()
                              if not k.startswith('__')}
                parameters['selectedParameters'] = {}
                parameters[key] = value
                parameters['selectedParameters'][key] = value
                parameters = ObjFromDict(parameters)
                parameters.run_name = "{}_{:03d}_{}={}".format(
                    args.name, run_id, key, value)
                print(sorted(parameters.selectedParameters.items()))
                train(parameters)
                run_id += 1
    elif args.sequence:
        run_id = 1
        for seq_dict in sequenceToRun:
            parameters = {k: v for k, v in BaseParameters.__dict__.items()
                          if not k.startswith('__')}
            parameters['selectedParameters'] = {}
            for key, value in seq_dict.items():
                if key not in parameters:
                    raise ValueError("key {} not in base parameters".format(key))
                parameters[key] = value
                parameters['selectedParameters'][key] = value
            parameters = ObjFromDict(parameters)
            parameters.run_name = "{}_{:03d}".format(args.name, run_id)
            print(sorted(parameters.selectedParameters.items()))
            train(parameters)
            run_id += 1

    elif args.model:
        parameters = {k: v for k, v in BaseParameters.__dict__.items()
                      if not k.startswith('__')}
        parameters['selectedParameters'] = {}
        for key, val in [("model_name", args.model),
                         ("pretrained", args.pretrained),
                         ("retrain", args.retrain), ("load", args.load)]:
            parameters[key] = val
            parameters['selectedParameters'][key] = val
        parameters = ObjFromDict(parameters)
        parameters.run_name = "{}".format(args.name)
        train(parameters)

    else:
        parameters = {k: v for k, v in BaseParameters.__dict__.items()
                      if not k.startswith('__')}
        parameters['selectedParameters'] = {}
        parameters = ObjFromDict(parameters)
        parameters.run_name = "{}".format(args.name)
        train(parameters)



if __name__ == "__main__":
    main()
