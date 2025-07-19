# -*- coding: utf-8 -*-
# SoftQuantizeLayer.py
# @Author: Arghya Ranjan Das

import tensorflow as tf
import math

class SoftQuantizeLayer(tf.keras.layers.Layer):
    def __init__(self,
                 initial_levels=(-1.0, -0.5, 0.0, 0.5),
                 max_delta = 2.0,
                 trainable_levels=True,
                 initial_k=1.0,
                 trainable_k=False,
                 **kwargs):
        super(SoftQuantizeLayer, self).__init__(**kwargs)
        self.initial_levels = initial_levels
        self.num_levels = len(self.initial_levels)
        self.initial_k = initial_k
        self.trainable_levels = trainable_levels
        self.trainable_k = trainable_k
        self.max_delta = max_delta

    def build(self, input_shape):
        self.level_0 = self.add_weight(
            name='level_0',
            shape=(1,),
            initializer=tf.constant_initializer(self.initial_levels[0]), 
            trainable=True
        )
        
        initial_deltas = [self.initial_levels[i] - self.initial_levels[i-1] for i in range(1, self.num_levels)]
        initial_log_deltas = [math.log(d / (self.max_delta - d)) for d in initial_deltas]

        
        self.log_deltas= self.add_weight(
            name='log_deltas',
            shape=(self.num_levels - 1,),
            initializer=tf.constant_initializer(initial_log_deltas),
            trainable=self.trainable_levels
        )
        
        # if k is trainable log_k is good variable to use
        self.log_k = self.add_weight(
            name='log_k',
            initializer=tf.constant_initializer(math.log(self.initial_k)),
            trainable=self.trainable_k,
            dtype=tf.float32
        )
        super(SoftQuantizeLayer, self).build(input_shape)

    @property
    def levels(self):
        deltas = self.max_delta * tf.sigmoid(self.log_deltas)
        cumulative_deltas = tf.cumsum(deltas)
        all_levels = tf.concat(
            [
                self.level_0, 
                self.level_0 + cumulative_deltas
                ], 
            axis=0
            )
        return all_levels
    
    @property
    def k(self):
        return tf.exp(self.log_k)
    
    def call(self, inputs, training=None):
        current_levels = self.levels
        hard_q = self._hard_quantize(inputs, current_levels)
        if training:
            soft_q = self._soft_quantize(inputs, current_levels)
            # Straight-Through Estimator
            return tf.stop_gradient(hard_q - soft_q) + soft_q 
        return tf.stop_gradient(hard_q) 

    def _soft_quantize(self, x, levels):
        """Applies the soft quantization formula."""
        x_reshaped = tf.expand_dims(x, axis=-1)
        levels_reshaped = tf.reshape(levels, (1,) * len(x.shape) + (-1,))

        dist = tf.square(x_reshaped - levels_reshaped)
        exp_term = tf.exp(-self.k * dist)
        weights = exp_term / tf.reduce_sum(exp_term, axis=-1, keepdims=True)

        return tf.reduce_sum(weights * levels, axis=-1)

    def _hard_quantize(self, x, levels):
        """Finds the closest quantization level for each input value."""
        x_reshaped = tf.expand_dims(x, axis=-1)
        abs_diff = tf.abs(x_reshaped - levels)
        indices = tf.argmin(abs_diff, axis=-1)

        return tf.gather(levels, indices)

    def get_config(self):
        config = super(SoftQuantizeLayer, self).get_config()
        config.update({
            'initial_levels': self.initial_levels,
            'initial_k': self.initial_k,
            'trainable_levels': self.trainable_levels,
            'trainable_k': self.trainable_k
        })
        return config
    
    
class AnnealingScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule, target_layer_name, verbose=0, **kwargs):
        super(AnnealingScheduler, self).__init__()
        self.schedule_params = kwargs
        self.target_layer_name = target_layer_name
        self.verbose = verbose
        self._pi = tf.constant(3.14159265358979, dtype=tf.float32)
        self._set_schedule_function(schedule)

    def _set_schedule_function(self, schedule):
        """Selects the schedule function based on the input string."""
        schedule_map = {
            'linear': self._linear_schedule,
            'cosine': self._cosine_schedule,
            'exponential': self._exponential_schedule,
            'step': self._step_schedule,
        }
        if callable(schedule):
            self.schedule_fn = schedule
        elif schedule in schedule_map:
            self.schedule_fn = schedule_map[schedule]
        else:
            raise ValueError(f"Unknown schedule: '{schedule}'. Supported: {list(schedule_map.keys())}")

    def on_train_begin(self, logs=None):
        try:
            self.layer = self.model.get_layer(self.target_layer_name)
            if not isinstance(self.layer, SoftQuantizeLayer):
                raise TypeError("Target layer must be a SoftQuantizeLayer.")
        except ValueError:
            raise ValueError(f"Layer '{self.target_layer_name}' not found in model.")
        
        self.schedule_params['total_epochs'] = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of an epoch to update k."""
        new_k = self.schedule_fn(epoch, **self.schedule_params)
        self.layer.log_k.assign(tf.math.log(tf.cast(new_k, tf.float32)))

        current_levels = self.layer.levels.numpy()
        levels_str = ", ".join([f"{level:.4f}" for level in current_levels])

        if self.verbose > 0:
            print(f"\nEpoch {epoch + 1}: Annealing 'k' set to {new_k:.4f}")
            print(f"\tLevels: {levels_str}")

    def _linear_schedule(self, epoch, total_epochs, initial_k=1.0, final_k=50.0):
        rate = tf.cast(epoch, tf.float32) / tf.cast(total_epochs, tf.float32)
        return initial_k + (final_k - initial_k) * rate

    def _cosine_schedule(self, epoch, total_epochs, initial_k=1.0, final_k=50.0):
        pi = self._pi
        rate = 0.5 * (1.0 - tf.cos(pi * tf.cast(epoch, tf.float32) / tf.cast(total_epochs, tf.float32)))
        return initial_k + (final_k - initial_k) * rate

    def _exponential_schedule(self, epoch, total_epochs, initial_k=1.0, final_k=50.0):
        rate = tf.cast(epoch, tf.float32) / tf.cast(total_epochs, tf.float32)
        return initial_k * (final_k / initial_k) ** rate

    def _step_schedule(self, epoch, total_epochs, initial_k=1.0, step_size=5, gamma=2.0):
        return initial_k * (gamma ** tf.floor(tf.cast(epoch, tf.float32) / step_size))


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    layer = SoftQuantizeLayer(initial_levels=(-1.0, -0.5, 0.0, 0.5))
    x_input = tf.constant(np.linspace(-1.5, 1.5, 500), dtype=tf.float32)
    layer.build(input_shape=x_input.shape)

    k_low = 2.0
    k_high = 25.0
    k_very_high = 500.0

    layer.log_k.assign(tf.math.log(k_low))
    y_soft_low_k = layer._soft_quantize(x_input, layer.levels)

    layer.log_k.assign(tf.math.log(k_high))
    y_soft_high_k = layer._soft_quantize(x_input, layer.levels)

    layer.log_k.assign(tf.math.log(k_very_high))
    y_soft_very_high_k = layer._soft_quantize(x_input, layer.levels)
    
    y_hard = layer(x_input, training=False)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(x_input, x_input, 'k--', alpha=0.4, label='Identity ($y=x$)')
    ax.plot(x_input, y_hard, color='red', linewidth=3.5, label=f'Hard Quantize (Inference)')
    ax.plot(x_input, y_soft_low_k, 'b-', linewidth=2.5, label=f'Soft Quantize (Early Training, $k={k_low}$)')
    ax.plot(x_input, y_soft_high_k, 'c-', linewidth=2.5, label=f'Soft Quantize (Mid Training, $k={k_high}$)')
    ax.plot(x_input, y_soft_very_high_k, 'black', linestyle='-.', linewidth=2.5, label=f'Soft Quantize (Late Training, $k={k_very_high}$)')

    ax.set_title("SoftQuantizeLayer Behavior Verification", fontsize=16)
    ax.set_xlabel("Input Value", fontsize=12)
    ax.set_ylabel("Output Value", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    plt.show()