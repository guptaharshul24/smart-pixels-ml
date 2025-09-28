# -*- coding: utf-8 -*-
# SoftQuantizeLayer.py
# @Author: Arghya Ranjan Das

import tensorflow as tf
import numpy as np
import math
from typing import Optional, List

# L_i = L0 + sum over j of softplus(Delta_L_raw[j])
# T_0 = T_off + softplus(Delta_T_raw[0])
# T_i = T_{i-1} + softplus(Delta_T_raw[i]) for i > 0

class SoftQuantizeLayer(tf.keras.layers.Layer):
    """
    A soft quantization layer with fully trainable, non-uniform levels and bins.
    """
    def __init__(self,
                 n_bits: int=2,
                 initial_levels: Optional[List[float]]=None,
                 threshold_offset: float=0.0,
                 initial_thresholds: Optional[List[float]]=None,
                 trainable_levels=True,
                 trainable_thresholds=True,
                 initial_k=1.0,
                 trainable_k=False,
                 **kwargs):
        super(SoftQuantizeLayer, self).__init__(**kwargs)
        assert isinstance(n_bits, int) and n_bits > 0, "'n_bits' must be a positive integer."
        
        self.n_bits = n_bits
        self.num_levels = 2 ** self.n_bits
        
        self.initial_levels = initial_levels
        self.initial_thresholds = initial_thresholds
        self.threshold_offset = threshold_offset
        
        self.trainable_levels = trainable_levels
        self.trainable_thresholds = trainable_thresholds
        self.initial_k = initial_k
        self.trainable_k = trainable_k
        
        if initial_levels is None or initial_thresholds is None:
            self.initial_range = (-1.0, 1.0)
        
    @staticmethod
    def _softplus(z):
        return tf.nn.softplus(z)
    
    def _inv_softplus_tf(z, clip_min: Optional[float] = None):
        z = tf.convert_to_tensor(z)
        if not z.dtype.is_floating: z = tf.cast(z, tf.float32)
        if clip_min is not None: z = tf.maximum(z, tf.cast(clip_min, z.dtype))

        if z.dtype in (tf.float16, tf.bfloat16):
            cutoff = tf.constant(8.0, dtype=z.dtype)
            z = tf.cast(z, tf.float32)  
        elif z.dtype == tf.float32:
            cutoff = tf.constant(20.0, dtype=z.dtype)
        else:
            cutoff = tf.constant(40.0, dtype=z.dtype)

        small_branch = tf.math.log(tf.math.expm1(z))
        large_branch = z + tf.math.log1p(-tf.math.exp(-z))
        out = tf.where(z > cutoff, large_branch, small_branch)
        return tf.cast(out, tf.float32)
    
    def _init_levels(self) -> np.ndarray:
        L = self.num_levels
        if self.initial_levels is not None:
            arr = np.asarray(self.initial_levels, dtype=np.float32)
            assert arr.shape[0] == L, "initial_levels length must equal 2^n_bits"
        else:
            lo, hi = self.initial_range
            arr = np.linspace(lo, hi, L, dtype=np.float32)
        if not np.all(np.diff(arr) > 0):
            arr = np.sort(arr)
        return arr  # (L,)
    
    def _init_thresholds(self):
        B = self.num_levels - 1
        if self.initial_thresholds is not None:
            arr = np.asarray(self.initial_thresholds, dtype=np.float32).astype(np.float32)
            assert arr.shape[0] == B, "initial_thresholds length must equal 2^n_bits - 1"
        else:
            lo, hi = self.initial_range
            arr = np.linspace(lo, hi, B, dtype=np.float32)
        if not np.all(np.diff(arr) > 0):
            arr = np.sort(arr)
        return arr  # (B,)
    
    def build_levels(self):
        initial_levels = self._init_levels()
        first_level_init = initial_levels[0]
        deltas_levels_init = np.diff(initial_levels)
        
        self.first_level = self.add_weight(
            name='first_level',
            shape=(1,),
            initializer=tf.constant_initializer(first_level_init),
            trainable=self.trainable_levels
        )

        level_deltas_raw_init = self._inv_softplus(deltas_levels_init).numpy().astype(np.float32)
        self.level_deltas_raw = self.add_weight(
            name='level_deltas_raw',
            shape=(self.num_levels - 1,),
            initializer=tf.constant_initializer(level_deltas_raw_init),
            trainable=self.trainable_levels
        )


    def build_thresholds(self):
        B = self.num_levels - 1
        initial_thresholds = self._init_thresholds()
        # Deltas are: T0-T_off, T1-T0, T2-T1, ...
        deltas_thresholds_init = np.diff(initial_thresholds, 
                                         prepend=self.threshold_offset)

        assert np.all(deltas_thresholds_init > 0), f"\nInitial thresholds must be strictly increasing. \nGiven threshold_offset: {self.threshold_offset}, initial_thresholds: {initial_thresholds}\n Check if they satisfy: threshold_offset < T0 < T1 < ... < T{B-1}."
            
        threshold_deltas_raw_init = self._inv_softplus(deltas_thresholds_init).numpy()
        self.threshold_deltas_raw = self.add_weight(
            name='threshold_deltas_raw',
            shape=(B,), 
            initializer=tf.constant_initializer(threshold_deltas_raw_init),
            trainable=self.trainable_thresholds
        )

    def build(self, input_shape):
        self.build_levels()
        self.build_thresholds()
        
        # --- parameter 'k' ---
        self.log_k = self.add_weight(
            name='log_k',
            shape=(1,),
            initializer=tf.constant_initializer(math.log(self.initial_k)),
            trainable=self.trainable_k
        )
        super(SoftQuantizeLayer, self).build(input_shape)

    @property
    def levels(self):
        """Calculates the trainable, non-uniform output levels."""
        deltas = self._softplus(self.level_deltas_raw)
        cumulative_deltas = tf.cumsum(deltas)
        return tf.concat([self.first_level, 
                          self.first_level + cumulative_deltas], 
                         axis=0
                        )
        
    @property
    def thresholds(self):
        deltas = self._softplus(self.threshold_deltas_raw)
        return self.threshold_offset + tf.cumsum(deltas)

    @property
    def k(self):
        return tf.exp(self.log_k)
    
    def call(self, inputs, training=None):
        q_levels = self.levels
        q_thresholds = self.thresholds
        hard_q = self._hard_quantize(inputs, q_levels, q_thresholds)
        
        if training:
            soft_q = self._soft_quantize(inputs, self.k, q_levels, q_thresholds)
            return tf.stop_gradient(hard_q - soft_q) + soft_q
        return tf.stop_gradient(hard_q) 

    @staticmethod
    def _soft_quantize(x, k, levels, thresholds):
        """
        Soft bin weights via smoothed CDF differences:
            weights = CDF_i - CDF_{i-1}, where CDF uses sigmoid(k*(t - x)).
        """
        x_exp = tf.expand_dims(x, axis=-1)                # (..., 1)
        sigs = tf.sigmoid(k * (thresholds - x_exp))       # (..., num_levels-1)
        left = tf.zeros_like(sigs[..., :1])               # (..., 1)
        right = tf.ones_like(sigs[..., :1])               # (..., 1)
        cdf = tf.concat([left, sigs, right], axis=-1)     # (..., B+2)
        weights = cdf[..., 1:] - cdf[..., :-1]            # (..., num_levels)
        return tf.reduce_sum(weights * levels, axis=-1)   # (...,)

    @staticmethod
    def _hard_quantize(x, levels, thresholds):
        x_reshaped = tf.expand_dims(x, axis=-1) 
        is_grt_th = x_reshaped > thresholds 
        indices = tf.reduce_sum(tf.cast(is_grt_th, dtype=tf.int32), axis=-1) 
        return tf.gather(levels, indices)

    def get_config(self):
        config = super(SoftQuantizeLayer, self).get_config()
        config.update({
            'n_bits': self.n_bits,
            'initial_range': self.initial_range,
            'trainable_levels': self.trainable_levels,
            'trainable_thresholds': self.trainable_thresholds,
            'initial_k': self.initial_k,
            'trainable_k': self.trainable_k
        })
        return config
    
   

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    n_bits = 2
    num_levels = 2 ** n_bits
    B = num_levels - 1
    initial_k_val = 50.0
    threshold_offset_init = -1.2

    layer = SoftQuantizeLayer(
        n_bits=n_bits,
        initial_k=initial_k_val,
        threshold_offset=threshold_offset_init,
    )
    x_input = tf.constant(np.linspace(-1.5, 1.5, 1000), dtype=tf.float32)
    layer.build(input_shape=x_input.shape)

    # Current absolute values
    L_abs0 = layer.levels.numpy().astype(np.float32)      # (L,)
    T_abs0 = layer.thresholds.numpy().astype(np.float32)  # (B,)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.60)

    y_hard_initial = layer._hard_quantize(x_input, layer.levels, layer.thresholds)
    y_soft_initial = layer._soft_quantize(x_input, layer.k, layer.levels, layer.thresholds)

    (line_hard,) = ax.plot(x_input, y_hard_initial, 'r-', lw=2.5, label='Hard Quantize (Forward Pass)')
    (line_soft,) = ax.plot(x_input, y_soft_initial, 'b-', alpha=0.8, lw=2.0, label='Soft Quantize (Backprop Approx.)')

    vlines = ax.vlines(layer.thresholds.numpy(), -1.5, 1.5,
                       colors='g', lw=2, alpha=0.7, linestyles='--', label='Thresholds (T)')
    ax.vlines(layer.threshold_offset, -1.5, 1.5,
              colors='m', lw=2, alpha=0.7, linestyles='--', label='Threshold Offset (T_off)')

    ax.set_title(f"Interactive {n_bits}-bit Soft Quantizer", fontsize=16)
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

   

    plt.show()