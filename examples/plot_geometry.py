"""Visualise the imported geometry and the iterative max-visibility placement.

    python examples/plot_geometry.py --out results [--buildings data/dtla_buildings.geojson]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from oran_asm_sim.config import SimConfig  # noqa: E402
from oran_asm_sim.geometry import build_deployment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--buildings", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = SimConfig.from_dict({"geometry": {"buildings_geojson": a.buildings}} if a.buildings else {})
    dep = build_deployment(cfg.geometry, cfg.radio)
    b, pl = dep.buildings, dep.placement

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [b.origin[0], b.origin[0] + b.shape[1] * b.res, b.origin[1], b.origin[1] + b.shape[0] * b.res]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1.25, 0.8]))
    im = ax[0].imshow(np.ma.masked_equal(b.h, 0), origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=150,
                      interpolation="nearest")
    fig.colorbar(im, ax=ax[0], fraction=0.035, label="building height [m]")
    sc = ax[0].scatter(*dep.site_xy.T, c="red", s=12 + dep.site_h / 2, marker="^", edgecolor="k", lw=0.4)
    for (x, y), az in zip(dep.sector_xy, dep.sector_az_deg):
        ax[0].plot([x, x + 60 * np.cos(np.radians(az))], [y, y + 60 * np.sin(np.radians(az))], c="red", lw=0.6)
    for (x, y), n in zip(dep.poi_xy, dep.poi_names):
        ax[0].plot(x, y, "*", c="w", ms=6, mec="k", mew=0.4)
    ax[0].set_title(f"{b.n_buildings} buildings ({dep.buildings_source}), {dep.n_sites} rooftop sites "
                    f"from {pl.n_candidates} candidates", fontsize=9)
    union = dep.site_vis.reshape(dep.n_sites, *b.shape).sum(0)
    img = np.ma.masked_where(~b.outdoor, np.minimum(union, 4))
    ax[1].imshow(np.ma.masked_equal(b.h, 0) * 0, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1,
                 interpolation="nearest")
    im2 = ax[1].imshow(img, origin="lower", extent=ext, cmap="magma", vmin=0, vmax=4, interpolation="nearest")
    fig.colorbar(im2, ax=ax[1], fraction=0.035, label="# sites with LOS (capped at 4)")
    ax[1].plot(*dep.site_xy.T, "c^", ms=4)
    ax[1].set_title("LOS visibility of the service area (voxel viewsheds, UE at 1.5 m)", fontsize=9)
    for x in ax[:2]:
        x.set_aspect("equal")
        x.set_xlabel("x east [m]")
        x.set_ylabel("y north [m]")
    ax[2].plot(np.arange(1, len(pl.coverage_curve) + 1), 100 * pl.coverage_curve, "o-", ms=3)
    ax[2].set_xlabel("greedy iteration (sites added)")
    ax[2].set_ylabel("SA cells with LOS to >= 1 site [%]")
    ax[2].grid(alpha=0.3)
    ax[2].set_title("Iterative max-visibility placement", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "placement.png"), dpi=140)
    print("saved", os.path.join(a.out, "placement.png"))


if __name__ == "__main__":
    main()
