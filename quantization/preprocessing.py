import argparse
from copy import deepcopy

import tensorflow as tf
from keras.callbacks import Callback
from keras.layers import *
from tensorflow import keras

from lib.jobs import *
from vitis_model_translator.translator import Translator


def filter_func(layer):
    return isinstance(layer, (Conv2D, Conv2DTranspose, Dense, Add, Multiply, Concatenate))

def add_l2_w_a_loss(model, lmbda):
    if not hasattr(model, 'layers'):
        return
    for layer in model.layers:
        if filter_func(layer):
            l2 = tf.keras.regularizers.L2(lmbda)
            layer.add_loss(lambda layer=layer: l2(layer.kernel))
            layer.add_loss(lambda layer=layer: l2(layer.bias))
            layer.add_loss(lambda layer=layer: l2(layer.output))
        else:
            add_l2_w_a_loss(layer, lmbda)
    return model

def remove_l2_w_a_loss(model, lmbda):
    if not hasattr(model, 'layers'):
        return
    for layer in model.layers:
        if filter_func(layer):
            l2 = tf.keras.regularizers.L2(lmbda)
            layer.add_loss(lambda layer=layer: -l2(layer.kernel))
            layer.add_loss(lambda layer=layer: -l2(layer.bias))
            layer.add_loss(lambda layer=layer: -l2(layer.output))
        else:
            add_l2_w_a_loss(layer, lmbda)
    return model

class L2NormPreprocessCallback(Callback):
    def __init__(self, args, translator: Translator, calib_dataset, validation_dataset, loss, metrics, norm_lmbda=0.001, calib_steps=10):
        super().__init__()
        self.args = args
        self.translator = translator
        self.calib_dataset = calib_dataset
        self.validation_dataset = validation_dataset
        self.loss = loss
        self.metrics = metrics
        self.norm_lmbda = norm_lmbda
        self.calib_steps = calib_steps
    
    def on_train_begin(self, logs=None):
        for submodel in self.model.layers:
            add_l2_w_a_loss(submodel, lmbda=self.norm_lmbda)
    
    def on_train_end(self, logs=None):
        for submodel in self.model.layers:
            remove_l2_w_a_loss(submodel, lmbda=self.norm_lmbda)
    
    def on_epoch_end(self, epoch, logs=None):
        model_q = deepcopy(self.model)
        self.translator.translate(model_q, self.calib_dataset, self.calib_steps, True, False)
        quant_loss = test(self.args, self.model, self.validation_dataset, self.loss, self.metrics, False)

        current_path = "last_epoch(quantization)"
        best_epoch_path = "best_epoch(quantization)"
        meta = {
            "epoch": epoch,
            "loss": quant_loss,
        }
        save_checkpoint(current_path, self.model, meta)

        # update best_epoch
        if not os.path.isdir(best_epoch_path):
            save_checkpoint(best_epoch_path, self.model, meta)
        else:
            best_epoch_meta = load_pkl(os.path.join(best_epoch_path, "meta.pkl"))
            if "loss" not in best_epoch_meta or quant_loss < best_epoch_meta["loss"]:
                shutil.rmtree(best_epoch_path)
                save_checkpoint(best_epoch_path, self.model, meta)

def main(args, model, tnd_filename, calib_dataset, train_dataset, validation_dataset, loss, metrics, callbacks, norm_lmbda):
    translator = Translator(tnd_filename)
    prep_callback = L2NormPreprocessCallback(args, translator, calib_dataset, validation_dataset, loss, metrics, norm_lmbda)
    callbacks.append(prep_callback)
    train(args, model, train_dataset, validation_dataset, loss, metrics, callbacks)