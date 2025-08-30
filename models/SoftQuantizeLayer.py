# -*- coding: utf-8 -*-
# SoftQuantizeLayer.py
# @Author: Arghya Ranjan Das

import tensorflow as tf
import numpy as np
import math

class SoftQuantizeLayer(tf.keras.layers.Layer):
    """
    A soft quantization layer with fully trainable, non-uniform levels and bins.
    """
    def __init__(self,
                 n_bits=2,
                 initial_range=[-1.0, 1.0],
                 trainable_levels=True,
                 trainable_thresholds=True,
                 initial_k=1.0,
                 trainable_k=False,
                 **kwargs):
        super(SoftQuantizeLayer, self).__init__(**kwargs)
        assert isinstance(n_bits, int) and n_bits > 0, "'n_bits' must be a positive integer."
        
        self.n_bits = n_bits
        self.num_levels = 2 ** self.n_bits
        self.initial_range = initial_range
        self.trainable_levels = trainable_levels
        self.trainable_thresholds = trainable_thresholds
        self.initial_k = initial_k
        self.trainable_k = trainable_k

    def build(self, input_shape):
        # --- (Y-axis) ---
        initial_levels = tf.linspace(self.initial_range[0], 
                                self.initial_range[1], 
                                self.num_levels)
                
        self.first_level = self.add_weight(
            name='first_level',
            shape=(1,),
            initializer=tf.constant_initializer(initial_levels[0].numpy()),
            trainable=self.trainable_levels
        )
        self.log_level_deltas = self.add_weight(
            name='log_level_deltas',
            shape=(self.num_levels - 1,),
            initializer=tf.constant_initializer(np.log(np.diff(initial_levels))),
            trainable=self.trainable_levels
        )

        # --- (X-axis) ---
        initial_thresholds =  tf.linspace(self.initial_range[0], 
                                     self.initial_range[1], 
                                     self.num_levels - 1)
        
        self.first_threshold = self.add_weight(
            name='first_threshold',
            shape=(1,),
            initializer=tf.constant_initializer(initial_thresholds[0].numpy()),
            trainable=self.trainable_thresholds
        )
        
        if self.num_levels > 2:
            self.log_threshold_deltas = self.add_weight(
                name='log_threshold_deltas',
                shape=(self.num_levels - 2,),
                initializer=tf.constant_initializer(np.log(np.diff(initial_thresholds))),
                trainable=self.trainable_thresholds
            )
            
        # --- parameter 'k' ---
        self.log_k = self.add_weight(
            name='log_k',
            shape=(1,),
            initializer=tf.constant_initializer(math.log(self.initial_k)),
            trainable=self.trainable_k
        )
        super(SoftQuantizeLayer, self).build(input_shape)
        
    @staticmethod
    def _delta_transform(x):
        pass
    
    def _inv_delta_transform(self, x):
        pass
        

    @property
    def levels(self):
        """Calculates the trainable, non-uniform output levels."""
        deltas = tf.exp(self.log_level_deltas)
        cumulative_deltas = tf.cumsum(deltas)
        return tf.concat([self.first_level, 
                          self.first_level + cumulative_deltas], 
                         axis=0
                        )
        
    @property
    def thresholds(self):
        if self.num_levels == 2:
            return self.first_threshold
        deltas = tf.exp(self.log_threshold_deltas)
        cum_deltas = tf.cumsum(deltas)
        return tf.concat([self.first_threshold, 
                          self.first_threshold + cum_deltas], 
                         axis=0)

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
        flat_thresholds = tf.reshape(thresholds, [-1])
        idx = tf.searchsorted(flat_thresholds, x, side='right')  # int32
        return tf.gather(levels, idx)

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
    num_levels = 2**n_bits
    initial_k_val = 50.0

    layer = SoftQuantizeLayer(n_bits=n_bits, initial_k=initial_k_val)
    x_input = tf.constant(np.linspace(-1.5, 1.5, 1000), dtype=tf.float32)
    layer.build(input_shape=x_input.shape)

    initial_first_level = layer.first_level.numpy()[0]
    initial_level_deltas = tf.exp(layer.log_level_deltas).numpy()
    
    initial_first_thresh = layer.first_threshold.numpy()[0]
    if num_levels > 2:
        initial_thresh_deltas = tf.exp(layer.log_threshold_deltas).numpy()

    fig, ax = plt.subplots(figsize=(10, 9))
    plt.subplots_adjust(bottom=0.55)

    y_hard_initial = layer._hard_quantize(x_input, layer.levels, layer.thresholds)
    y_soft_initial = layer._soft_quantize(x_input, initial_k_val, layer.levels, layer.thresholds)

    line_hard, = ax.plot(x_input, y_hard_initial, 'r-', lw=2.5, 
                         label='Hard Quantize (Inference)')
    line_soft, = ax.plot(x_input, y_soft_initial, 'b-', alpha=0.8, lw=2.0, 
                         label='Soft Quantize (Training Approx.)')
    
    vlines = ax.vlines(layer.thresholds.numpy(), -1.5, 1.5, 
                       colors='g', lw=2, alpha=0.7, linestyles='--',
                       label='Thresholds (X)')
    
    ax.set_title(f"Interactive {n_bits}-bit Quantizer (Threshold Control)", 
                 fontsize=16)
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)

    ax_k = fig.add_axes([0.15, 0.45, 0.75, 0.02])
    k_slider = Slider(ax=ax_k, label='k (Softness)',
                      valmin=0.1, valmax=200.0, valinit=initial_k_val)
    
    ax.text(0.05, 0.4, 'Level Sliders (Y-axis)', transform=fig.transFigure, fontsize=12)
    level_slider_axes = [fig.add_axes([0.15, 0.35 - i*0.04, 0.75, 0.02]) for i in range(num_levels)]
    first_level_slider = Slider(ax=level_slider_axes[0], label='L0 Position', valmin=-1.5, valmax=0.0, valinit=initial_first_level)
    level_delta_sliders = [Slider(ax=level_slider_axes[i+1], label=f'L Delta {i+1}', valmin=0.01, valmax=1.5, valinit=initial_level_deltas[i]) for i in range(num_levels - 1)]
    
    ax.text(0.05, 0.18, 'Threshold Sliders (X-axis)', transform=fig.transFigure, fontsize=12)
    thresh_slider_axes = [fig.add_axes([0.15, 0.15 - i*0.04, 0.75, 0.02]) for i in range(num_levels-1)]
    first_thresh_slider = Slider(ax=thresh_slider_axes[0], label='T1 Position', valmin=-1.5, valmax=0.0, valinit=initial_first_thresh)
    if num_levels > 2:
      thresh_delta_sliders = [Slider(ax=thresh_slider_axes[i+1], label=f'T Delta {i+2}', valmin=0.01, valmax=1.5, valinit=initial_thresh_deltas[i]) for i in range(num_levels - 2)]

    def update(val):
        layer.first_level.assign([first_level_slider.val])
        level_delta_vals = [s.val for s in level_delta_sliders]
        layer.log_level_deltas.assign(np.log(level_delta_vals))

        layer.first_threshold.assign([first_thresh_slider.val])
        if num_levels > 2:
            thresh_delta_vals = [s.val for s in thresh_delta_sliders]
            layer.log_threshold_deltas.assign(np.log(thresh_delta_vals))
        
        layer.log_k.assign([tf.math.log(k_slider.val)])

        current_levels = layer.levels
        current_thresholds = layer.thresholds
        
        y_hard_new = layer._hard_quantize(x_input, current_levels, current_thresholds)
        y_soft_new = layer._soft_quantize(x_input, k_slider.val, current_levels, current_thresholds)
        
        line_hard.set_ydata(y_hard_new)
        line_soft.set_ydata(y_soft_new)
        
        global vlines
        vlines.remove()
        vlines = ax.vlines(current_thresholds.numpy(), -1.5, 1.5, 
                           colors='g', lw=2, alpha=0.7, linestyles='--',
                           label='Thresholds (X)')

        fig.canvas.draw_idle()

    k_slider.on_changed(update)
    first_level_slider.on_changed(update)
    for s in level_delta_sliders: s.on_changed(update)
    first_thresh_slider.on_changed(update)
    if num_levels > 2:
      for s in thresh_delta_sliders: s.on_changed(update)

    plt.show()

    
    

