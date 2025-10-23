# -*- coding: utf-8 -*-
# softQuantPlot.py
# @Author: Arghya Ranjan Das

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Assuming the corrected SoftQuantizeLayer is in this path
from models.SoftQuantizeLayer import SoftQuantizeLayer

# --- Configuration ---
n_bits = 2
num_levels = 2 ** n_bits
n_thresh = num_levels - 1
x_min, x_max = 0, 2000
y_min, y_max = -0.5, 4.0 

initial_k_val = 1.0
initial_levels = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
threshold_offset = 80.0
initial_thresholds = np.array([400.0, 800.0, 1500.0], dtype=np.float32)

deriv_scale = 50.0  # Scaling factor for numerical derivative

# --- Layer Initialization ---
layer = SoftQuantizeLayer(
    n_bits=n_bits,
    initial_k=initial_k_val,
    initial_levels=initial_levels,
    threshold_offset=threshold_offset,
    initial_thresholds=initial_thresholds,
)
x_input = tf.constant(np.linspace(x_min, x_max, 22000), dtype=tf.float32)
layer.build(input_shape=x_input.shape)

def numerical_derivative(x, y, scale=deriv_scale):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.gradient(y, x, edge_order=2) * scale

# --- Plotting Setup ---
fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(bottom=0.5)
ax.set_title(f"Interactive {n_bits}-bit Soft Quantizer", fontsize=16)
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

y_hard_initial = layer._hard_quantize(x_input, layer.levels, layer.thresholds)
y_soft_initial = layer._soft_quantize(x_input, layer.k, layer.levels, layer.thresholds, layer.tau)
dy_soft_init = numerical_derivative(x_input.numpy(), y_soft_initial.numpy())
print(dy_soft_init)

line_hard, = ax.plot(x_input, y_hard_initial, 'r-', lw=2.5, label='Hard Quantize (Inference)')
line_soft, = ax.plot(x_input, y_soft_initial, 'b-', alpha=0.8, lw=2, label='Soft Quantize (Training)')
line_dsoft, = ax.plot(x_input, dy_soft_init, 'k--', lw=1.5, alpha=0.8, label=f'dy/dx (soft approx) * {deriv_scale}')

vlines_thresh = ax.vlines(layer.thresholds.numpy(), y_min, y_max, colors='g', lw=2, alpha=0.7, linestyles='--', label='Thresholds (T)')
vline_offset = ax.axvline(layer.threshold_offset, color='purple', lw=2, alpha=0.7, linestyle=':', label='Offset (T_off)')
ax.legend(loc='upper left')

# --- Sliders Setup ---
initial_L0 = layer.first_level.numpy()[0]
initial_level_deltas = np.diff(layer.levels.numpy())
initial_abs_thresholds = layer.thresholds.numpy()

ax_k = fig.add_axes([0.15, 0.40, 0.75, 0.02])
level_slider_axes = [fig.add_axes([0.15, 0.32 - i*0.03, 0.75, 0.02]) for i in range(num_levels)]
thresh_slider_axes = [fig.add_axes([0.15, 0.15 - i*0.03, 0.75, 0.02]) for i in range(n_thresh + 1)]

k_slider = Slider(ax=ax_k, label='k (Softness)', valmin=0.1, valmax=200.0, valinit=initial_k_val)
L0_slider = Slider(ax=level_slider_axes[0], label='L0', valmin=-0.5, valmax=0.5, valinit=initial_L0)
level_delta_sliders = [Slider(ax=level_slider_axes[i+1], label=f'ΔL{i}', valmin=0.01, valmax=10, valinit=initial_level_deltas[i]) for i in range(n_thresh)]
T_offset_slider = Slider(ax=thresh_slider_axes[0], label='T_off', valmin=-1.5, valmax=300.0, valinit=threshold_offset)
thresh_abs_sliders = [Slider(ax=thresh_slider_axes[i+1], label=f'T{i}', valmin=-1.5, valmax=2000, valinit=initial_abs_thresholds[i]) for i in range(n_thresh)]

_slider_update_guard = False



def update(val):
    global _slider_update_guard, vlines_thresh, vline_offset
    if _slider_update_guard: return

    # --- Read & Process Thresholds ---
    t_off_val = T_offset_slider.val
    thresh_abs_vals = np.array([s.val for s in thresh_abs_sliders])

    sorted_thresh = np.sort(thresh_abs_vals)
    if not np.array_equal(sorted_thresh, thresh_abs_vals):
        _slider_update_guard = True
        for i, v in enumerate(sorted_thresh): thresh_abs_sliders[i].set_val(v)
        _slider_update_guard = False
        thresh_abs_vals = sorted_thresh
    
    if t_off_val >= thresh_abs_vals[0]:
        t_off_val = thresh_abs_vals[0] - 0.01
        _slider_update_guard = True
        T_offset_slider.set_val(t_off_val)
        _slider_update_guard = False

    # --- Update Layer's Internal Parameters ---
    layer.first_level.assign([L0_slider.val])
    
    level_delta_vals = np.array([s.val for s in level_delta_sliders], dtype=np.float32)
    # layer.level_deltas_raw.assign(layer._inv_softplus(level_delta_vals))
    layer.level_deltas_raw.assign(layer._log1p(level_delta_vals))
    
    threshold_deltas = np.diff(thresh_abs_vals, prepend=t_off_val).astype(np.float32)
    layer.threshold_offset = t_off_val
    # layer.threshold_deltas_raw.assign(layer._inv_softplus(threshold_deltas))
    layer.threshold_deltas_raw.assign(layer._log1p(threshold_deltas))
    
    layer.log_k.assign([tf.math.log(k_slider.val)])

    # --- Recalculate and Redraw ---
    current_levels = layer.levels
    current_thresholds = layer.thresholds
    current_tau = layer.tau
    
    line_hard.set_ydata(layer._hard_quantize(x_input, current_levels, current_thresholds))
    line_soft.set_ydata(layer._soft_quantize(x_input, layer.k, current_levels, current_thresholds, current_tau))
    
    # Numerical derivative update
    y_soft_updated = layer._soft_quantize(x_input, layer.k, current_levels, current_thresholds, current_tau)
    dy_soft_updated = numerical_derivative(x_input.numpy(), y_soft_updated.numpy())
    line_dsoft.set_ydata(dy_soft_updated)

    vlines_thresh.remove()
    vline_offset.remove()
    vlines_thresh = ax.vlines(current_thresholds.numpy(), x_min, x_max, colors='g', lw=2, alpha=0.7, linestyles='--')
    vline_offset = ax.axvline(layer.threshold_offset, color='purple', lw=2, alpha=0.7, linestyle=':')
    
    fig.canvas.draw_idle()

# Attach callbacks
k_slider.on_changed(update)
L0_slider.on_changed(update)
T_offset_slider.on_changed(update)
for s in level_delta_sliders: s.on_changed(update)
for s in thresh_abs_sliders: s.on_changed(update)

plt.show()