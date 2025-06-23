from keras.layers import *
from keras.models import Sequential, Model
from keras.utils import Sequence
from qkeras import *

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def var_network(var, hidden=10, output=2):
    var = Flatten(name="flatten")(var)
    tf.debugging.check_numerics(var, "nan after flatten")
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="dense_1"
    )(var)
    tf.debugging.check_numerics(var, "nan after dense_1")
    #var = keras.activations.tanh(var)
    var = QActivation("quantized_tanh(8, 0, 1)", name="activation_tanh_2")(var)
    tf.debugging.check_numerics(var, "nan after activation_tanh_2")
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="dense_2"
    )(var)
    tf.debugging.check_numerics(var, "nan after dense_2")
    #var = keras.activations.tanh(var)
    var = QActivation("quantized_tanh(8, 0, 1)", name="activation_tanh_3")(var)
    tf.debugging.check_numerics(var, "nan after activation_tanh_3")
    return QDense(
        output,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        name="dense_3"
    )(var)

def mlp_encoder_network(var, hidden=16, hidden_dimx=16, hidden_dimy=16):
    proj_x = AveragePooling2D(
        pool_size=(1, hidden_dimx), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(var)
    tf.debugging.check_numerics(proj_x, "nan after proj_x AveragePooling2D")
    proj_x = Flatten(name="flatten_x")(proj_x)
    tf.debugging.check_numerics(proj_x, "nan after proj_x Flatten")

    proj_y = AveragePooling2D(
        pool_size=(hidden_dimy, 1), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(var)
    tf.debugging.check_numerics(proj_y, "nan after proj_y AveragePooling2D")
    proj_y = Flatten(name="flatten_y")(proj_y)
    tf.debugging.check_numerics(proj_y, "nan after proj_y Flatten")


    proj_x = QDense(
        hidden_dimx,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(proj_x)
    tf.debugging.check_numerics(proj_x, "nan after proj_x QDense")
    proj_x = QActivation("quantized_relu(bits=13, integer=5)(x)")(proj_x)
    tf.debugging.check_numerics(proj_x, "nan after proj_x QActivation")


    proj_y = QDense(
        hidden_dimy,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(proj_y)
    tf.debugging.check_numerics(proj_y, "nan after proj_y QDense")
    proj_y = QActivation("quantized_relu(bits=13, integer=5)(x)")(proj_y)
    tf.debugging.check_numerics(proj_y, "nan after proj_y QActivation")


    enc_out = Concatenate(axis=1)([proj_x, proj_y])
    tf.debugging.check_numerics(enc_out, "nan after Concatenate")


    enc_out = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(enc_out)
    tf.debugging.check_numerics(enc_out, "nan after enc_out QDense")

    enc_out = QActivation("quantized_tanh(8, 0, 1)")(enc_out)
    tf.debugging.check_numerics(enc_out, "nan after enc_out QActivation")
    return enc_out

def CreateModel(shape, output=8):
    hidden = 16
    hidden_dimx=shape[0]
    hidden_dimy=shape[1]
    x_base = x_in = Input(shape, name="input_pxls/")
    stack = mlp_encoder_network(x_base, hidden, hidden_dimx, hidden_dimy,)
    stack = var_network(stack, hidden=16, output=output) # this network should only be used with 'slim' (3) or 'diagonal' (8) regression targets
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model
