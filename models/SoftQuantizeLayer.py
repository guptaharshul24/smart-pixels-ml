# -*- coding: utf-8 -*-
# SoftQuantizeLayer.py
# @Author: Arghya Ranjan Das

import tensorflow as tf
import math

class SoftQuantizeLayer(tf.keras.layers.Layer):
    def __init__(self,
                 levels=(-1.0, -0.5, 0.0, 0.5),
                 initial_k=1.0,
                 trainable_k=False,
                 **kwargs):
        super(SoftQuantizeLayer, self).__init__(**kwargs)
        self.levels_tuple = levels
        self.initial_k = initial_k
        self.trainable_k = trainable_k

    def build(self, input_shape):
        self.levels = tf.constant(self.levels_tuple, dtype=tf.float32)

        # if k is trainable log_k is good variable to use
        self.log_k = self.add_weight(
            name='log_k',
            initializer=tf.constant_initializer(math.log(self.initial_k)),
            trainable=self.trainable_k,
            dtype=tf.float32
        )
        self.k = tf.exp(self.log_k)
        super(SoftQuantizeLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        hard_q = self._hard_quantize(inputs)
        if training:
            soft_q = self._soft_quantize(inputs)
            return tf.stop_gradient(hard_q - soft_q) + soft_q # Straight-Through Estimator
        return tf.stop_gradient(hard_q) 

    def _soft_quantize(self, x):
        """Applies the soft quantization formula."""
        x_reshaped = tf.expand_dims(x, axis=-1)
        levels_reshaped = tf.reshape(self.levels, (1,) * len(x.shape) + (-1,))

        dist = tf.square(x_reshaped - levels_reshaped)
        exp_term = tf.exp(-self.k * dist)
        weights = exp_term / tf.reduce_sum(exp_term, axis=-1, keepdims=True)

        return tf.reduce_sum(weights * self.levels, axis=-1)

    def _hard_quantize(self, x):
        """Finds the closest quantization level for each input value."""
        x_reshaped = tf.expand_dims(x, axis=-1)
        abs_diff = tf.abs(x_reshaped - self.levels)
        indices = tf.argmin(abs_diff, axis=-1)

        return tf.gather(self.levels, indices)

    def get_config(self):
        config = super(SoftQuantizeLayer, self).get_config()
        config.update({
            'levels': self.levels_tuple,
            'initial_k': self.initial_k,
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

        if self.verbose > 0:
            print(f"\nEpoch {epoch + 1}: Annealing 'k' set to {new_k:.4f}")

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

    x_input = tf.constant(np.linspace(-1.5, 1.5, 500), dtype=tf.float32)


    k_low = 2.0
    k_high = 50.0
    k_very_high = 500.0
    layer_low_k = SoftQuantizeLayer(
        levels=(-1.0, -0.5, 0.0, 0.5),
        initial_k=k_low
    )
    layer_high_k = SoftQuantizeLayer(
        levels=(-1.0, -0.5, 0.0, 0.5),
        initial_k=k_high
    )
    layer_very_high_k = SoftQuantizeLayer(
        levels=(-1.0, -0.5, 0.0, 0.5),
        initial_k=k_very_high
    )
    

    y_soft_low_k = layer_low_k(x_input, training=True)    # Low 'k' (smooth quantization for early training)
    y_soft_high_k = layer_high_k(x_input, training=True)  # High 'k' (sharp quantization for mid training)
    y_soft_very_high_k = layer_very_high_k(x_input, training=True) # Very high 'k' (sharp quantization for late training)
    y_hard = layer_high_k(x_input, training=False)        # Hard quantization (inference mode)

    
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