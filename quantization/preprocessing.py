import tensorflow as tf
from tensorflow import keras
from keras.layers import *
import argparse
from lib.jobs import *

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

def main(args, model, train_dataset, validation_dataset, losses, metrics, callbacks, norm_lmbda):
    add_l2_w_a_loss(model, lmbda=norm_lmbda)
    # TODO: 实现preprocessing；calib dataset在过程中需要被引入；每个train epoch结束之后进行量化性能测试并保存最优的epoch
    # train(args, model, train_dataset, validation_dataset, )