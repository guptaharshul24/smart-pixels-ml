# anim_softquantizer.py
import tensorflow as tf
import numpy as np
import math, sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from models.SoftQuantizeLayer import SoftQuantizeLayer   # your module

# ---------- core generator ---------------------------------------------------
def make_animation(sweep: str, out_path: Path,
                   n_bits=2, n_frames=60, fps=10,
                   k_range=(1.0, 100.0),
                   d_level_range=(0.05, 0.8),
                   d_bin_range=(0.05, 0.8)):

    assert sweep in {"k", "levels", "bins"}

    layer = SoftQuantizeLayer(n_bits=n_bits, initial_k=k_range[0])
    x = tf.constant(np.linspace(-1.5, 1.5, 500), tf.float32)
    layer.build(input_shape=x.shape)

    # ---------- sweep preparation ----------
    if sweep == "k":
        param_vals = np.linspace(*k_range, n_frames, dtype=np.float32)
    elif sweep == "levels":
        base = tf.exp(layer.log_level_deltas).numpy()
        param_vals = np.linspace(*d_level_range, n_frames).astype(np.float32)
    else:  # bins
        base = tf.exp(layer.log_bin_deltas).numpy()
        param_vals = np.linspace(*d_bin_range, n_frames).astype(np.float32)

    # --------- pre‑compute frames ----------
    y_hard, y_soft, bins, levs = [], [], [], []
    for v in param_vals:
        if sweep == "k":
            layer.log_k.assign([math.log(float(v))]) 
        elif sweep == "levels":
            layer.log_k.assign([math.log(float(50))])
            layer.log_level_deltas.assign(np.log(base * v))
        else:  # bins
            layer.log_k.assign([math.log(float(50))])
            layer.log_bin_deltas.assign(np.log(base * v))

        bins.append(layer.bin_centers.numpy())
        levs.append(layer.levels.numpy())
        y_hard.append(layer._hard_quantize(x, layer.levels, layer.bin_centers).numpy())
        y_soft.append(layer._soft_quantize(x, layer.levels, layer.bin_centers).numpy())

    # --------- plot scaffold ---------------
    fig, ax = plt.subplots(figsize=(8, 6))
    lh, = ax.plot([], [], 'r-', lw=2.5, label='Hard')
    ls, = ax.plot([], [], 'b-', lw=2.0, label='Soft')
    sc  = ax.scatter([], [], c='green', s=60, marker='o', label='Bin centres')
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
           xlabel="Input", ylabel="Output",
           title=f"{n_bits}-bit SoftQuantizer ({sweep}-sweep)")
    ax.grid(True); ax.legend(loc='upper left')

    def init():
        lh.set_data([], []); ls.set_data([], [])
        sc.set_offsets(np.empty((0, 2))); return lh, ls, sc

    def draw(i):
        lh.set_data(x, y_hard[i]); ls.set_data(x, y_soft[i])
        sc.set_offsets(np.c_[bins[i], levs[i]])
        if sweep == "k":
            ax.set_title(f"k = {param_vals[i]:.2f}")
        else:
            ax.set_title(f"{sweep} scale = {param_vals[i]:.2f}")
        return lh, ls, sc

    ani = animation.FuncAnimation(fig, draw, init_func=init,
                                  frames=n_frames, blit=True)

    # --------- write -----------------------
    if out_path.suffix == ".gif":
        ani.save(out_path, writer="pillow", fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, codec='libx264',
                              extra_args=['-pix_fmt', 'yuv420p'])
        ani.save(out_path, writer=writer)
    plt.close(fig)
    print(f"[OK] {sweep} sweep saved → {out_path}")

# ---------- cli helper -------------------------------------------------------
if __name__ == "__main__":
    # run all three sweeps
    make_animation("k",      Path("softquantizer_k.gif"))
    make_animation("levels", Path("softquantizer_levels.gif"))
    make_animation("bins",   Path("softquantizer_bins.gif"))
