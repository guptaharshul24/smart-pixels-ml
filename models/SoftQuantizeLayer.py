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
                 trainable_bins=True,
                 initial_k=1.0,
                 trainable_k=False,
                 **kwargs):
        super(SoftQuantizeLayer, self).__init__(**kwargs)
        assert isinstance(n_bits, int) and n_bits > 0, "'n_bits' must be a positive integer."
        
        self.n_bits = n_bits
        self.num_levels = 2 ** self.n_bits
        self.initial_range = initial_range
        self.trainable_levels = trainable_levels
        self.trainable_bins = trainable_bins
        self.initial_k = initial_k
        self.trainable_k = trainable_k

    def build(self, input_shape):
        initial_points = tf.linspace(self.initial_range[0], 
                                     self.initial_range[1], 
                                     self.num_levels)
        initial_first_point = initial_points[0]
        initial_deltas = np.diff(initial_points.numpy())

        # --- (Y-axis) ---
        self.first_level = self.add_weight(
            name='first_level',
            shape=(1,),
            initializer=tf.constant_initializer(initial_first_point.numpy()),
            trainable=self.trainable_levels
        )
        self.log_level_deltas = self.add_weight(
            name='log_level_deltas',
            shape=(self.num_levels - 1,),
            initializer=tf.constant_initializer(np.log(initial_deltas)),
            trainable=self.trainable_levels
        )

        # --- (X-axis) ---
        self.first_bin_center = self.add_weight(
            name='first_bin_center',
            shape=(1,),
            initializer=tf.constant_initializer(initial_first_point.numpy()),
            trainable=self.trainable_bins
        )
        self.log_bin_deltas = self.add_weight(
            name='log_bin_deltas',
            shape=(self.num_levels - 1,),
            initializer=tf.constant_initializer(np.log(initial_deltas)),
            trainable=self.trainable_bins
        )
        
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
        deltas = tf.exp(self.log_level_deltas)
        cumulative_deltas = tf.cumsum(deltas)
        return tf.concat([self.first_level, 
                          self.first_level + cumulative_deltas], 
                         axis=0
                        )

    @property
    def bin_centers(self):
        """Calculates the trainable, non-uniform bin centers."""
        deltas = tf.exp(self.log_bin_deltas)
        cumulative_deltas = tf.cumsum(deltas)
        return tf.concat([self.first_bin_center, 
                          self.first_bin_center + cumulative_deltas], 
                         axis=0
                         )

    @property
    def k(self):
        return tf.exp(self.log_k)
    
    def call(self, inputs, training=None):
        q_levels = self.levels
        q_bins = self.bin_centers
        hard_q = self._hard_quantize(inputs, q_levels, q_bins)
        
        if training:
            soft_q = self._soft_quantize(inputs, q_levels, q_bins)
            return tf.stop_gradient(hard_q - soft_q) + soft_q
        return tf.stop_gradient(hard_q) 

    def _soft_quantize(self, x, levels, bin_centers):
        x_reshaped = tf.expand_dims(x, axis=-1)
        dist = tf.square(x_reshaped - bin_centers)
        exp_term = tf.exp(-self.k * dist)
        weights = exp_term / tf.reduce_sum(exp_term, axis=-1, keepdims=True)
        return tf.reduce_sum(weights * levels, axis=-1)

    def _hard_quantize(self, x, levels, bin_centers):
        x_reshaped = tf.expand_dims(x, axis=-1)
        abs_diff = tf.abs(x_reshaped - bin_centers)
        indices = tf.argmin(abs_diff, axis=-1)
        return tf.gather(levels, indices)

    def get_config(self):
        config = super(SoftQuantizeLayer, self).get_config()
        config.update({
            'n_bits': self.n_bits,
            'initial_range': self.initial_range,
            'trainable_levels': self.trainable_levels,
            'trainable_bins': self.trainable_bins,
            'initial_k': self.initial_k,
            'trainable_k': self.trainable_k
        })
        return config
    
   
   
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    n_bits = 2 
    num_levels = 2**n_bits
    initial_k_val = 20.0

    layer = SoftQuantizeLayer(n_bits=n_bits, initial_k=initial_k_val)
    x_input = tf.constant(np.linspace(-1.5, 1.5, 500), dtype=tf.float32)
    layer.build(input_shape=x_input.shape)

    initial_first_level = layer.first_level.numpy()[0]
    initial_level_deltas = tf.exp(layer.log_level_deltas).numpy()
    initial_first_bin = layer.first_bin_center.numpy()[0]
    initial_bin_deltas = tf.exp(layer.log_bin_deltas).numpy()

    fig, ax = plt.subplots(figsize=(10, 9))
    plt.subplots_adjust(bottom=0.55)

    y_hard_initial = layer._hard_quantize(x_input, layer.levels, layer.bin_centers)
    y_soft_initial = layer._soft_quantize(x_input, layer.levels, layer.bin_centers)

    line_hard, = ax.plot(x_input, y_hard_initial, 'r-', lw=2.5, 
                         label='Hard Quantize (Inference)')
    line_soft, = ax.plot(x_input, y_soft_initial, 'b-', alpha=0.8, 
                         lw=2.0, label='Soft Quantize (Training Approx.)')
    bin_markers, = ax.plot(layer.bin_centers.numpy(), layer.levels.numpy(), 'x', 
                           color='green', mew=3, ms=10, label='Bin Centers (X)')

    ax.set_title(f"Interactive {n_bits}-bit Quantizer (Delta Control)", fontsize=16)
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)

    ax_k = fig.add_axes([0.15, 0.45, 0.75, 0.02])
    k_slider = Slider(ax=ax_k, label='k (Softness)', 
                      valmin=0.1, valmax=200.0, 
                      valinit=initial_k_val
                      )
    
    ax.text(0.05, 0.4, 'Level Sliders (Y-axis)', transform=fig.transFigure, fontsize=12)
    level_slider_axes = [fig.add_axes([0.15, 0.35 - i*0.04, 0.75, 0.02]) for i in range(num_levels)]
    first_level_slider = Slider(ax=level_slider_axes[0], label='L0 Position', valmin=-1.5, valmax=0.0, valinit=initial_first_level)
    level_delta_sliders = [Slider(ax=level_slider_axes[i+1], 
                                  label=f'L Delta {i+1}', 
                                  valmin=0.01, valmax=1.5, 
                                  valinit=initial_level_deltas[i]
                                  ) 
                           for i in range(num_levels - 1)]
    
    ax.text(0.05, 0.18, 'Bin Sliders (X-axis)', transform=fig.transFigure, fontsize=12)
    bin_slider_axes = [fig.add_axes([0.15, 0.15 - i*0.04, 0.75, 0.02]) for i in range(num_levels)]
    first_bin_slider = Slider(ax=bin_slider_axes[0], 
                              label='B0 Position', 
                              valmin=-1.5, valmax=0.0, 
                              valinit=initial_first_bin
                              )
    bin_delta_sliders = [Slider(ax=bin_slider_axes[i+1], 
                                label=f'B Delta {i+1}', 
                                valmin=0.01, valmax=1.5, 
                                valinit=initial_bin_deltas[i]
                                ) 
                         for i in range(num_levels - 1)]

    def update(val):
        layer.first_level.assign([first_level_slider.val])
        level_delta_vals = [s.val for s in level_delta_sliders]
        layer.log_level_deltas.assign(np.log(level_delta_vals))

        layer.first_bin_center.assign([first_bin_slider.val])
        bin_delta_vals = [s.val for s in bin_delta_sliders]
        layer.log_bin_deltas.assign(np.log(bin_delta_vals))

        layer.log_k.assign([tf.math.log(k_slider.val)])

        current_levels = layer.levels
        current_bins = layer.bin_centers
        
        y_hard_new = layer._hard_quantize(x_input, current_levels, current_bins)
        y_soft_new = layer._soft_quantize(x_input, current_levels, current_bins)
        
        line_hard.set_ydata(y_hard_new)
        line_soft.set_ydata(y_soft_new)
        bin_markers.set_data(current_bins.numpy(), current_levels.numpy())

        fig.canvas.draw_idle()

    k_slider.on_changed(update)
    first_level_slider.on_changed(update)
    for s in level_delta_sliders: s.on_changed(update)
    first_bin_slider.on_changed(update)
    for s in bin_delta_sliders: s.on_changed(update)

    plt.show()
    
    

