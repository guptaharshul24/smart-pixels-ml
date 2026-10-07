"""Two trial layouts for showing the comparator-delay row alongside everything else.

The delay condition's residuals are 3-30x wider than any other condition's (worst
in the angles: +-5 deg vs -100..+150 deg), so on one shared axis it rescales every
panel and compresses the other eight conditions into dots on the zero line. These
are the two candidate fixes, for comparison against the plain single-axis version:

  8panel -- per quantity, TWO continuous axes side by side: a zoomed one scaled to
            the non-delay conditions, and a full-range one. Every row appears in
            both. Nothing is cut out; the same data is shown twice at two scales.

  broken -- per quantity, ONE axis split into three segments with the middle
            covering the non-delay range and the outer two the delay tails, with
            break marks between. More compact, but the delay bars SPAN zero, so
            they cross the breaks rather than sitting beyond them -- which is
            exactly the case broken axes handle badly. Included so that can be
            judged by eye rather than argued about.

Reuses row_geometry() and draw_panel() from plot_residual_comparison.py so the
row order, spacing, grouping boxes and marker styling stay identical across all
three figures.

Usage: python plot_residual_comparison_trials.py [8panel|broken|both]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_residual_comparison import (ARCHS, ARCH_LABELS, DATASET_LABELS, GROUP_COLORS,
                                      GROUP_ORDER, QUANTITIES, VARIANTS, VARIANT_LABELS,
                                      VARIANT_MARKERS, draw_panel, load, row_geometry)

here = os.path.dirname(os.path.abspath(__file__))
# Conditions whose residuals are far wider than the rest, and which the zoomed
# panels therefore scale themselves to exclude.
DELAY_CONDITIONS = {"no_noise_delay", "no_noise_delay_retimed"}


def get_data():
    d = {"pixelav": load("pixelav_3sr"), "pixelav_matched": load("pixelav_matched"),
         "frontend": load("frontend"), "no_noise": load("no_noise")}
    # pick up every delay condition without having to list them twice
    for c in DELAY_CONDITIONS:
        d[c] = load(c)
    return d


def get_groups(data):
    return [g for g in GROUP_ORDER if g[0] in ARCHS and
            any(v in data[g[1]].get(g[0], {}) for v in VARIANTS)]


def x_extent(quantity, groups_subset, data, pad=0.08):
    """Widest span of markers, 68% bars and sigma bands over a subset of rows."""
    lo, hi = float("inf"), float("-inf")
    for arch, dataset in groups_subset:
        for stats in data[dataset].get(arch, {}).values():
            m = stats[f"mean_{quantity}"]
            lo = min(lo, m - stats[f"down68_{quantity}"])
            hi = max(hi, m + stats[f"up68_{quantity}"])
            us, ds = stats.get(f"mean_upsigma{quantity}"), stats.get(f"mean_downsigma{quantity}")
            if us is not None and ds is not None:
                lo, hi = min(lo, m - ds), max(hi, m + us)
    span = hi - lo
    return lo - pad * span, hi + pad * span


def add_legends(fig, groups, title, y_anchor=1.01, row_h=0.24):
    """Legends sit just above y_anchor (figure coords) and grow upward; the title
    clears whichever is taller. y_anchor < 1 keeps everything inside the canvas,
    which matters when the axes do not run to the figure top."""
    gh = [plt.Line2D([0], [0], color=GROUP_COLORS[g], lw=2,
                     label=f"{ARCH_LABELS[g[0]]}, {DATASET_LABELS[g[1]]}") for g in groups]
    vh = [plt.Line2D([0], [0], marker=VARIANT_MARKERS[v], color="black",
                     linestyle="", label=VARIANT_LABELS[v]) for v in VARIANTS]
    leg = fig.legend(handles=gh, loc="lower left", ncol=2,
                     bbox_to_anchor=(0.02, y_anchor), frameon=False)
    fig.add_artist(leg)
    fig.legend(handles=vh, loc="lower right", ncol=1,
               bbox_to_anchor=(0.98, y_anchor), frameon=False)
    fig_h = 0.7 * len(groups) + 2
    rows = max((len(gh) + 1) // 2, len(vh))
    fig.suptitle(title, y=y_anchor + (rows * row_h + 0.12) / fig_h)


def make_8panel(data, groups, geom):
    """Per quantity, a zoomed and a full-range axis sitting as one visual unit.

    The pair members are butted together (no wspace) and share top/bottom spines,
    so each quantity reads as one box split by a scale change rather than as two
    unrelated panels; the gaps between quantities come from narrow empty spacer
    columns in the gridspec. A dotted rectangle around each pair reinforces it,
    matching the grouping boxes already used on the rows.
    """
    y_positions, boxes, dividers = geom
    non_delay = [g for g in groups if g[1] not in DELAY_CONDITIONS]

    fig = plt.figure(figsize=(24, 0.7 * len(groups) + 2))
    # 4 pairs of panels separated by 3 narrow spacer columns
    gs = fig.add_gridspec(1, 11, width_ratios=[1, 1, 0.35] * 3 + [1, 1], wspace=0.0,
                          left=0.03, right=0.995, bottom=0.11, top=0.855)

    pairs = []
    first = None
    for qi, (quantity, xlabel) in enumerate(QUANTITIES):
        c0 = 3 * qi
        ax_zoom = fig.add_subplot(gs[0, c0], sharey=first)
        ax_full = fig.add_subplot(gs[0, c0 + 1], sharey=first or ax_zoom)
        first = first or ax_zoom
        for ax, scope in ((ax_zoom, "zoom"), (ax_full, "full")):
            draw_panel(ax, quantity, groups, y_positions, data, boxes, dividers)
            if scope == "zoom":
                ax.set_xlim(*x_extent(quantity, non_delay, data))
            else:
                ax.margins(x=0.08)
            ax.set_xlabel(f"{xlabel}\n({scope})", fontsize=9)
        ax_full.tick_params(labelleft=False)
        # the zoom panel's last tick and the full panel's first would otherwise
        # nearly touch across the shared seam
        ax_zoom.set_xticks(ax_zoom.get_xticks()[:-1])
        ax_zoom.set_xlim(*x_extent(quantity, non_delay, data))
        pairs.append((ax_zoom, ax_full))

    first.set_ylim(min(y_positions) - 0.8, max(y_positions) + 0.8)
    add_legends(fig, groups, "Residual comparison -- trial A: zoomed + full axis per quantity",
                y_anchor=0.875, row_h=0.20)

    # dotted box round each pair, drawn in figure coords once positions are final
    fig.canvas.draw()
    for ax_zoom, ax_full in pairs:
        b0, b1 = ax_zoom.get_position(), ax_full.get_position()
        pad = 0.004
        fig.add_artist(plt.Rectangle(
            (b0.x0 - pad, b0.y0 - pad), (b1.x1 - b0.x0) + 2 * pad, b0.height + 2 * pad,
            transform=fig.transFigure, fill=False, edgecolor="dimgray",
            linestyle=":", linewidth=1.3, alpha=0.6, zorder=5))

    out = os.path.join(here, "residual_comparison_8panel.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved to {out}")


def make_broken(data, groups, geom):
    y_positions, boxes, dividers = geom
    non_delay = [g for g in groups if g[1] not in DELAY_CONDITIONS]
    fig, axes = plt.subplots(1, 12, figsize=(26, 0.7 * len(groups) + 2), sharey=True,
                             gridspec_kw={"width_ratios": [1, 3, 1] * 4, "wspace": 0.06})
    for qi, (quantity, xlabel) in enumerate(QUANTITIES):
        c_lo, c_hi = x_extent(quantity, non_delay, data)
        f_lo, f_hi = x_extent(quantity, groups, data)
        segs = [(min(f_lo, c_lo), c_lo), (c_lo, c_hi), (c_hi, max(f_hi, c_hi))]
        trio = axes[3 * qi:3 * qi + 3]
        for ax, (lo, hi) in zip(trio, segs):
            draw_panel(ax, quantity, groups, y_positions, data, boxes, dividers)
            ax.set_xlim(lo, hi)
        # hide the spines either side of each break and mark it with diagonals
        for left, right in ((trio[0], trio[1]), (trio[1], trio[2])):
            left.spines["right"].set_visible(False)
            right.spines["left"].set_visible(False)
            for ax, xpos in ((left, 1.0), (right, 0.0)):
                ax.plot([xpos, xpos], [0, 1], transform=ax.transAxes, color="none")
                kw = dict(transform=ax.transAxes, color="k", clip_on=False, lw=1)
                ax.plot((xpos - 0.03, xpos + 0.03), (-0.012, 0.012), **kw)
                ax.plot((xpos - 0.03, xpos + 0.03), (0.988, 1.012), **kw)
        trio[1].set_xlabel(xlabel)
        for ax in (trio[0], trio[2]):
            ax.tick_params(labelsize=8)
    axes[0].set_ylim(min(y_positions) - 0.8, max(y_positions) + 0.8)
    add_legends(fig, groups, "Residual comparison -- trial B: broken axis (3 segments per quantity)")
    plt.tight_layout()
    out = os.path.join(here, "residual_comparison_broken.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved to {out}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    data = get_data()
    groups = get_groups(data)
    geom = row_geometry(groups, data)
    if which in ("8panel", "both"):
        make_8panel(data, groups, geom)
    if which in ("broken", "both"):
        make_broken(data, groups, geom)


if __name__ == "__main__":
    main()
