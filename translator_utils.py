"""
This file should not be accessed under RUNNER MODE.
"""

import tensorflow as tf
from unittest import result
from tensorflow_model_optimization.quantization.keras import vitis_quantize
import numpy as np


def get_quantized_model(model, calib_dataset=None, calib_steps=None, init_quant=False):
    quantizer = vitis_quantize.VitisQuantizer(model, quantize_strategy="8bit_tqt")
    if not init_quant:
        qat_model = quantizer.get_qat_model(init_quant=False)
    else:
        qat_model = quantizer.get_qat_model(
            init_quant=True,
            calib_dataset=calib_dataset,
            calib_steps=calib_steps,
        )
    return qat_model


def deploy_qat_model(model, output_file):
    quantized_model = vitis_quantize.VitisQuantizer.get_deploy_model(model)
    quantized_model.save(output_file)


def dataset2np(dataset: tf.data.Dataset, num_items):
    if num_items == 1:
        return np.array(list(dataset.as_numpy_iterator))

    results = []
    for i in range(num_items):
        cur = dataset.map(lambda x: x[i])
        results.append(np.array(list(cur.as_numpy_iterator)))
    return tuple(results)
