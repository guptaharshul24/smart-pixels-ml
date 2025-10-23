
# -*- coding: utf-8 -*-
# AnnealingScheduler.py
# @Author: Arghya Ranjan Das

from models.SoftQuantizeLayer import SoftQuantizeLayer
import tensorflow as tf
    
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
        self.layer.log_k.assign([tf.math.log(tf.cast(new_k, tf.float32))])

        current_levels = self.layer.levels.numpy()
        current_thresholds = self.layer.thresholds.numpy()
        current_tau = self.layer.tau.numpy()

        levels_str = ", ".join([f"{level:.4f}" for level in current_levels])
        thresholds_str = ", ".join([f"{threshold:.4f}" for threshold in current_thresholds])
        if self.verbose > 0:
            print(f"\nEpoch {epoch + 1}: Annealing 'k' set to {new_k:.4f}")
            print(f"\tLevels: {levels_str}")
            print(f"\tThresholds: {thresholds_str}")
            print(f"\tTau: {current_tau}")

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


