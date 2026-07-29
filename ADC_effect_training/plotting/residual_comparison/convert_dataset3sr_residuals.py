"""
Copies dataset3sr_residuals_raw.json (fetched via scp from
hgupta@cmslpc-el9.fnal.gov:/uscms/home/jennetd/nobackup/smart-pixels/regression-paper/dataset3sr/residuals.json
-- the real source file behind the prior team's compare-res-3sr-ONEBIG.ipynb)
to residuals_pixelav_3sr.json, content unmodified but pretty-printed (indent=2,
matching residuals_frontend.json's formatting) -- the source file itself is
minified/single-line.

No remapping/renaming: residuals_frontend.json (see aggregate_frontend_results.py)
is written in this exact same structure/key-naming, so plot_residual_comparison.py
reads both sides through one code path with no format translation. All 7
architectures and all 4 variants are kept, not just the ones we have a match
for on our own side -- plot_residual_comparison.py decides what to draw.
"""
import os
import json

here = os.path.dirname(os.path.abspath(__file__))


def main():
    src = os.path.join(here, "dataset3sr_residuals_raw.json")
    dst = os.path.join(here, "residuals_pixelav_3sr.json")
    with open(src) as f:
        data = json.load(f)
    with open(dst, "w") as f:
        json.dump(data, f, indent=2)
    print(f"copied {src} -> {dst} (pretty-printed)")


if __name__ == "__main__":
    main()
