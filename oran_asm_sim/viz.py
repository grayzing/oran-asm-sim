"""Matplotlib helpers (optional dependency)."""
from __future__ import annotations

import numpy as np

STATE_COLORS = {0: "#2a9d8f", 1: "#e9c46a", 2: "#264653", 3: "#f4a261"}
LEVEL_COLORS = ["#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#264653"]


def plot_network(sim, ax, show_ues=True, show_pois=True, title=None, cmd=None):
    import matplotlib.patches as mpatches
    dep = sim.dep
    lo, hi = dep.area_min, dep.area_max
    ax.add_patch(mpatches.Rectangle(lo, *(hi - lo), fill=False, ls="--", lw=1, ec="0.4"))
    if dep.buildings is not None:
        b = dep.buildings
        ext = [b.origin[0], b.origin[0] + b.shape[1] * b.res, b.origin[1], b.origin[1] + b.shape[0] * b.res]
        ax.imshow(np.ma.masked_equal(b.h, 0), origin="lower", extent=ext, cmap="Greys", alpha=0.45,
                  vmin=0, vmax=150, interpolation="nearest")
    r0 = min(sim.cfg.geometry.isd_m / 3.0, 110.0)
    macro = ~getattr(sim, "controllable", np.ones(dep.n_sectors, bool))   # always-on (HetNet macro) sectors
    for s in range(dep.n_sectors):
        x, y = dep.sector_xy[s]
        r = 2.5 * r0 if macro[s] else r0
        az = np.radians(dep.sector_az_deg[s])
        if cmd is not None:
            c = LEVEL_COLORS[int(cmd[s])]
        else:
            c = STATE_COLORS[int(sim.state[s])] if sim.state[s] != 0 else LEVEL_COLORS[int(sim.asm_level[s])]
        w = mpatches.Wedge((x, y), r, np.degrees(az) - 55, np.degrees(az) + 55, fc=c, ec="white", lw=0.5,
                           alpha=0.35 if macro[s] else 0.85, zorder=1 if macro[s] else 2)
        ax.add_patch(w)
    ax.plot(*dep.site_xy.T, "k^", ms=4)
    if macro.any():
        ax.plot(*dep.site_xy[np.unique(dep.sector_site[macro])].T, "k*", ms=14, mfc="#ffd166", label="macro")
    if show_ues:
        ue = sim.mob.xy
        cov = sim.serving >= 0
        ax.scatter(*ue[cov].T, s=2, c="#1d3557", alpha=0.5, lw=0)
        ax.scatter(*ue[~cov].T, s=6, c="#e63946", marker="x", lw=0.8)
    if show_pois:
        for (x, y), n in zip(dep.poi_xy, dep.poi_names):
            ax.plot(x, y, "*", c="#d62828", ms=7)
            ax.annotate(n, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(lo[0] - 100, hi[0] + 100)
    ax.set_ylim(lo[1] - 100, hi[1] + 100)
    ax.set_aspect("equal")
    ax.set_xlabel("x east [m]")
    ax.set_ylabel("y north [m]")
    if title:
        ax.set_title(title, fontsize=9)
    handles = [mpatches.Patch(color=c, label=l) for c, l in
               zip(LEVEL_COLORS, ["no ASM", "SM1", "SM2", "SM3", "SM4 (off)"])]
    ax.legend(handles=handles, loc="lower left", fontsize=6, framealpha=0.8)
    return ax
