# QKeras 0.9.0 requires legacy Keras 2 (tf_keras), not Keras 3 -- running it
# under Keras 3 silently breaks gradient tracking for one of a layer's
# kernel_quantizer/bias_quantizer (root-caused 2026-08-25: a minimal QDense
# with both quantizers gets a real gradient on only one of the two under
# Keras 3, and on BOTH once forced onto tf_keras -- see das's own
# requirements.txt, which pins tensorflow==2.15.1 specifically "required by
# QKeras"). This file is used exclusively by the QConv2D (Part 2.5) path, so
# it's safe to force tf_keras unconditionally here -- no other train_*.py
# imports this module. models/mdmm.py (shared with the ViT/non-quantized
# Conv2D paths, which work fine under Keras 3) makes this conditional
# instead; see that file.
import tf_keras as keras
from tf_keras.layers import *
from tf_keras.models import Sequential, Model
from tf_keras.utils import Sequence
from qkeras import *

import tensorflow as tf
from tf_keras import regularizers

def var_network(var, hidden=10, output=2):
    var = Flatten()(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=regularizers.L1L2(0.01),
        activity_regularizer=regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(8, 0, 1)")(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=regularizers.L1L2(0.01),
        activity_regularizer=regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(8, 0, 1)")(var)
    return QDense(
        output,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=regularizers.L1L2(0.01),
    )(var)

def conv_network(var, n_filters=5, kernel_size=3):
    var = QSeparableConv2D(
        n_filters,kernel_size,
        depthwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        pointwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        depthwise_regularizer=regularizers.L1L2(0.01),
        pointwise_regularizer=regularizers.L1L2(0.01),
        activity_regularizer=regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)
    var = QConv2D(
        n_filters,1,
        kernel_quantizer=quantized_bits(4, 0, alpha=1),
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        kernel_regularizer=regularizers.L1L2(0.01),
        activity_regularizer=regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)    
    return var

def CreateModel(shape, output, n_filters, pool_size):
    x_base = x_in = Input(shape)
    stack = conv_network(x_base)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    stack = var_network(stack, hidden=16, output=output)
    model = Model(inputs=x_in, outputs=stack)
    return model