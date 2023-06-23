import argparse
from copy import deepcopy

import tensorflow as tf
from keras.callbacks import Callback
from keras.layers import *
from tensorflow import keras

from lib.jobs import *
from vitis_model_translator.translator import Translator


def filter_func(layer):
    return isinstance(layer, (Conv2D, Conv2DTranspose, Dense))

def add_l2_w_a_loss(model, lmbda):
    for layer in model.layers:
        if filter_func(layer):
            l2 = tf.keras.regularizers.L2(lmbda)
            layer.add_loss(lambda layer=layer: l2(layer.kernel))
            layer.add_loss(lambda layer=layer: l2(layer.bias))
            layer.add_loss(lambda layer=layer: l2(layer.output))
    return model

def remove_l2_w_a_loss(model, lmbda):
    for layer in model.layers:
        if filter_func(layer):
            l2 = tf.keras.regularizers.L2(lmbda)
            layer.add_loss(lambda layer=layer: -l2(layer.kernel))
            layer.add_loss(lambda layer=layer: -l2(layer.bias))
            layer.add_loss(lambda layer=layer: -l2(layer.output))
    return model

class L2NormPreprocessCallback(Callback):
    def __init__(self, translator: Translator, calib_dataset, validation_dataset, loss, metrics, norm_lmbda=0.001, calib_steps=10):
        super().__init__()
        self.translator = translator
        self.calib_dataset = calib_dataset
        self.validation_dataset = validation_dataset
        self.loss = loss
        self.metrics = metrics
        self.norm_lmbda = norm_lmbda
        self.calib_steps = calib_steps

        self.best_loss = None
    
    def on_train_begin(self, logs=None):
        add_l2_w_a_loss(self.model, lmbda=self.norm_lmbda)
    
    def on_train_end(self, logs=None):
        remove_l2_w_a_loss(self.model, lmbda=self.norm_lmbda)
    
    def on_epoch_end(self, epoch, logs=None):
        model_q = deepcopy(self.model)
        self.translator.translate(model_q, self.calib_dataset, self.calib_steps, True, False)
        loss_test = test()
        # Perform quantization & save the epoch with best loss

def main(args, model, tnd_filename, calib_dataset, train_dataset, validation_dataset, loss, metrics, callbacks, norm_lmbda):
    translator = Translator(tnd_filename)
    prep_callback = L2NormPreprocessCallback(translator, calib_dataset, validation_dataset, loss, metrics, norm_lmbda)
    callbacks.append(prep_callback)
    train(args, model, train_dataset, validation_dataset, loss, metrics, callbacks)