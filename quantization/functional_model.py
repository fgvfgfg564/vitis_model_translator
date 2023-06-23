import logging
from copy import deepcopy

import numpy as np
import tensorflow as tf
from keras.layers import *
from tensorflow import keras
from tensorflow_model_optimization.quantization.keras import vitis_quantize

from .utils import *

# ----- Set weights for layer & activations -----


def min_max_from_log_th(log_th):
    clogth = np.ceil(log_th)
    maxx = np.exp2(clogth)
    minx = - maxx
    return minx, maxx


def set_new_log_th_for_activation(q_act, new_log_th):
    """
    Handles Activation, Add, Multiply, Concatenate
    """
    minx, maxx = min_max_from_log_th(new_log_th)
    optimizer_steps = q_act.get_weights()[1]
    q_act.set_weights([new_log_th, optimizer_steps, minx, maxx])


def set_log_ths_for_convolution(q_conv, new_weight_log_th, new_bias_log_th, new_activation_log_th=None):
    """
    Handles Conv2D, Conv2DTranspose
    """
    minxw, maxxw = min_max_from_log_th(new_weight_log_th)
    minxb, maxxb = min_max_from_log_th(new_bias_log_th)
    old_weights = q_conv.get_weights()
    include_post_activation = (len(old_weights) == 12)

    if include_post_activation and new_activation_log_th is None:
        raise ValueError(
            "Layer includes post activation quant. Please provide activation log_th")
    if not include_post_activation and new_activation_log_th is not None:
        logging.warn("layer doesn't include post activation. The activation log_th is ignored.")

    if include_post_activation:
        minxa, maxxa = min_max_from_log_th(new_activation_log_th)
        k = old_weights[1]
        b = old_weights[3]
        step = old_weights[5]
        new_weights = [new_weight_log_th, k, new_bias_log_th, b,
                       new_activation_log_th, step, minxw, maxxw, minxb, maxxb, minxa, maxxa]
    else:
        k = old_weights[1]
        b = old_weights[3]
        step = old_weights[4]
        new_weights = [new_weight_log_th, k, new_bias_log_th,
                       b, step, minxw, maxxw, minxb, maxxb]
    q_conv.set_weights(new_weights)

# ----- Calculation of tensor ranges -----


def swap_axes(input_tensor, axis1, axis2):
    """
    Swaps two axes of a TensorFlow tensor.

    Args:
        input_tensor (tf.Tensor): The input tensor to swap axes.
        axis1 (int): The index of the first axis to swap.
        axis2 (int): The index of the second axis to swap.

    Returns:
        tf.Tensor: The tensor with the swapped axes.
    """
    perm = list(range(len(input_tensor.shape)))
    perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
    return tf.transpose(input_tensor, perm=perm)

def tensor_range(x: tf.Tensor, channel_axis, scale_type='0.99'):
    if scale_type == '0.99':
        x = tf.abs(x)
        x = swap_axes(x, 0, channel_axis)
        C = x.shape[0]
        x = tf.reshape(x, [C, -1])
        N = tf.shape(x)[1]
        K = N // 100 + 2
        K = tf.minimum(K, N)
        x = tf.math.top_k(x, K, sorted=False).values
        x = tf.reduce_min(x, axis=1)
        x = tf.reduce_max(x)
        return x


def tensor_log_th(x, channel_axis, scale_type='0.99'):
    rng = tensor_range(x, channel_axis, scale_type)
    if tf.math.equal(rng, 0.):
        return tf.constant(-12.)
    log_th = tf.math.log(rng) / tf.math.log(2.)
    return log_th

# ----- Read and reset log_th for weights -----


def read_weight_from_qconv(qconv_layer):
    for weight in qconv_layer.weights:
        if weight.name.endswith("kernel:0"):
            k = weight.numpy()
        if weight.name.endswith("bias:0"):
            b = weight.numpy()
    return (k, b)


def set_new_log_th_for_convolution(qconv_layer, activation_log_th=None):
    k, b = read_weight_from_qconv(qconv_layer)
    transposed = isinstance(qconv_layer.layer, Conv2DTranspose)
    channel_axis = 2 if transposed else 3
    kernel_log_th = tensor_log_th(k, channel_axis).numpy()
    bias_log_th = tensor_log_th(b, 0).numpy()
    set_log_ths_for_convolution(
        qconv_layer, kernel_log_th, bias_log_th, activation_log_th)

# ----- Capture the activations -----


class ActivationRangeCapture(Wrapper):
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
        self.log_th = tf.Variable(0., trainable=False, name="log_th")

    def call(self, inputs):
        outputs = self.layer(inputs)
        self.log_th.assign(tensor_log_th(outputs, channel_axis=3))
        return outputs

    def apply_to_layer(self):
        new_log_th = self.log_th.numpy()
        if isinstance(self.layer.layer, (Conv2D, Conv2DTranspose)):
            set_new_log_th_for_convolution(self.layer, new_log_th)
        else:
            set_new_log_th_for_activation(self.layer, new_log_th)


def add_activation_capture(model):
    """
    Handles all layers with quantized output
    """
    def filter(layer):
        for weight in layer.weights:
            if weight.name.endswith('post_activation_log_th:0') or weight.name.endswith('output_0_log_th:0'):
                return True
        return False

    def factory(layer): return ActivationRangeCapture(layer)
    return insert_layer_nonseq(model, filter, factory, 'replace')


def remove_activation_capture(model):
    def filter(layer): return isinstance(layer, ActivationRangeCapture)
    def factory(layer): return layer.layer
    return insert_layer_nonseq(model, filter, factory, 'replace')

# ----- Model quantization -----


def quantize_model(model, calib_dataset, calib_steps, scale_type='0.99'):
    """
    Returns a quantized model. Calibrated with a single NumPy batch
    """
    model = deepcopy(model)
    quantizer = vitis_quantize.VitisQuantizer(
        model, quantize_strategy="8bit_tqt")

    # Use Vitis Quantizer to apply CLE
    model_cle = quantizer.get_qat_model(init_quant=True, calib_dataset=calib_dataset,
                                        calib_steps=calib_steps)

    model_capture = add_activation_capture(model_cle)
    if isinstance(calib_dataset, list):
        # Multiple inputs
        sample = []
        for d in calib_dataset:
            if isinstance(d, tf.data.Dataset):
                for s in d:
                    sample.append(s)
            else:
                sample.append(d)
    elif isinstance(calib_dataset, tf.data.Dataset):
        sample = next(calib_dataset.iter())
    else:
        sample = calib_dataset
    _ = model_capture(sample)
    for layer in model_capture.layers:
        if isinstance(layer, ActivationRangeCapture):
            layer.apply_to_layer()
    model_optimized = remove_activation_capture(model_capture)

    return model_optimized
