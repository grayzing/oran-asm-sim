"""Iterative maximum-visibility base-station placement, after

  [MASRUR25] S. Masrur, I. Guvenc, D. Lopez-Perez, "Energy-Efficient Sleep Mode
             Optimization in 5G mmWave Networks via Multi-Agent Deep
             Reinforcement Learning", arXiv:2511.22105, Sec. III-A / VI.

Procedure (as in the paper):
  1. The service area (SA) is the outdoor part of the gridded 3-D map.
  2. Candidate BS locations P_c are rooftop cells on building boundaries that
     face the SA (excluding those near the area boundary), at roof height
     (+ mast).
  3. For every candidate a voxel viewshed gives the set of SA grid points it
     sees (LOS from the antenna to a UE at h_ut).
  4. Greedy: repeatedly add the candidate that increases the covered SA the
     most; stop when no further improvement is possible -> reduced set P_r.

Practical knobs for a large real city (the paper's map is 129 m x 206 m with
143 candidates): candidates are thinned to one per ``candidate_spacing_m`` cell
(tallest boundary roof wins) and capped at ``max_candidates``; the viewshed is
limited to ``viewshed_radius_m``; and greedy also stops at ``coverage_target``,
``max_sites`` or when the marginal gain drops below ``min_gain_frac`` of the SA
(set it to 0 for the paper's strict "no improvement" rule).

Results (sites, heights, azimuths and the per-site viewsheds that the channel
model then uses for LOS) are cached on disk, keyed by a hash of the raster and
all parameters, so the geometry is processed once.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import GeometryConfig, RadioConfig

CACHE_VERSION = 2


def default_cache_dir() -> str:
    return os.environ.get("ORAN_ASM_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "oran_asm_sim"))


def _popcount_rows(a: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(a).sum(axis=-1, dtype=np.int64)
    lut = np.array([bin(i).count("1") for i in range(256)], np.int64)
    return lut[a].sum(axis=-1)


# --------------------------------------------------------------------------- #
def viewshed(raster, obs_xy, h_obs: float, h_tgt: float, radius: float) -> np.ndarray:
    """Radial-sweep voxel viewshed. Returns the flat indices of outdoor cells that
    have LOS from an antenna at (obs_xy, h_obs) to a receiver at height h_tgt."""
    res = raster.res
    H = raster.h
    nr, nc = H.shape
    step = 0.75 * res
    n_steps = int(radius / step)
    n_rays = int(np.ceil(2 * np.pi * radius / res))
    ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    d = (np.arange(1, n_steps + 1) * step)[None, :]
    px = obs_xy[0] + np.cos(ang)[:, None] * d
    py = obs_xy[1] + np.sin(ang)[:, None] * d
    c = np.floor((px - raster.origin[0]) / res).astype(np.int64)
    r = np.floor((py - raster.origin[1]) / res).astype(np.int64)
    inside = (c >= 0) & (c < nc) & (r >= 0) & (r < nr)
    rc, cc = np.where(inside, r, 0), np.where(inside, c, 0)
    hs = np.where(inside, H[rc, cc], 0.0)
    a = (hs - h_obs) / d                                     # elevation tangent of obstacles
    cm = np.maximum.accumulate(a, axis=1)
    excl = np.concatenate([np.full((n_rays, 1), -np.inf), cm[:, :-1]], axis=1)
    t = (h_tgt - h_obs) / d
    vis = inside & (hs <= 0) & (t >= excl)
    return np.unique(rc[vis] * nc + cc[vis])


def find_candidates(raster, geo: GeometryConfig, rng: np.random.Generator):
    """Rooftop boundary cells facing the service area, thinned and capped."""
    H, res = raster.h, raster.res
    bld = H > 0
    out = ~bld
    nb = np.zeros_like(bld)
    nb[1:, :] |= out[:-1, :]
    nb[:-1, :] |= out[1:, :]
    nb[:, 1:] |= out[:, :-1]
    nb[:, :-1] |= out[:, 1:]
    edge = bld & nb & (H >= geo.cand_min_height_m) & (H <= geo.cand_max_height_m)
    m = max(int(round(geo.fill_margin_m / res)), 1)
    edge[:m, :] = edge[-m:, :] = False
    edge[:, :m] = edge[:, -m:] = False
    r, c = np.nonzero(edge)
    if r.size == 0:
        return np.zeros((0, 2)), np.zeros(0)
    h = H[r, c]
    # thin: keep the tallest boundary roof per spacing cell
    g = max(int(round(geo.candidate_spacing_m / res)), 1)
    key = (r // g) * (H.shape[1] // g + 1) + (c // g)
    order = np.lexsort((-h, key))
    first = np.ones(order.size, bool)
    first[1:] = key[order][1:] != key[order][:-1]
    sel = order[first]
    if sel.size > geo.max_candidates:
        sel = rng.choice(sel, geo.max_candidates, replace=False)
    xy = raster.origin + (np.stack([c[sel], r[sel]], 1) + 0.5) * res
    return xy, H[r[sel], c[sel]].astype(float)


def best_azimuth_offset(site_xy, vis_flat, raster, radio: RadioConfig, base_az=(30.0, 150.0, 270.0)):
    """Rotate the tri-sector so that the sectors' horizontal patterns best cover
    the cells this site sees (1/d-weighted). Returns an offset in [0, 120)."""
    if vis_flat.size == 0:
        return 0.0
    nc = raster.shape[1]
    ctr = raster.origin + (np.stack([vis_flat % nc, vis_flat // nc], 1) + 0.5) * raster.res
    v = ctr - site_xy
    bearing = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    w = 1.0 / np.maximum(np.hypot(v[:, 0], v[:, 1]), 30.0)
    best, best_s = 0.0, -np.inf
    for off in np.arange(0.0, 120.0, 5.0):
        g = 0.0
        phis = [((bearing - (b + off) + 180) % 360) - 180 for b in base_az]
        lin = np.max([10 ** (-np.minimum(12 * (p / radio.hpbw_h_deg) ** 2, radio.a_max_db) / 10) for p in phis], 0)
        g = float((w * lin).sum())
        if g > best_s:
            best, best_s = float(off), g
    return best


@dataclass
class Placement:
    site_xy: np.ndarray        # [N, 2]
    site_h: np.ndarray         # [N] antenna height (roof + mast)
    az_offset: np.ndarray      # [N] deg
    vis: np.ndarray            # [N, n_cells] bool, per-site viewshed (LOS to h_ut)
    coverage_curve: np.ndarray # cumulative SA coverage after each greedy pick
    n_candidates: int
    n_sa_cells: int
    cache_file: Optional[str] = None


def _cache_key(raster, geo: GeometryConfig, radio: RadioConfig) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(raster.h).tobytes())
    params = dict(v=CACHE_VERSION, origin=raster.origin.tolist(), res=raster.res, shape=list(raster.shape),
                  h_ut=radio.h_ut_m, mast=geo.mast_height_m, hmin=geo.cand_min_height_m,
                  hmax=geo.cand_max_height_m, spacing=geo.candidate_spacing_m, maxc=geo.max_candidates,
                  radius=geo.viewshed_radius_m, gain=geo.min_gain_frac, cov=geo.coverage_target,
                  maxs=geo.max_sites, margin=geo.fill_margin_m, az=geo.optimize_azimuth,
                  hpbw=radio.hpbw_h_deg, seed=geo.placement_seed)
    h.update(json.dumps(params, sort_keys=True).encode())
    return h.hexdigest()[:20]


def place_sites(raster, geo: GeometryConfig, radio: RadioConfig, verbose: bool = True) -> Placement:
    cache_dir = geo.cache_dir or default_cache_dir()
    key = _cache_key(raster, geo, radio)
    path = os.path.join(cache_dir, f"placement_{key}.npz")
    n_cells = raster.h.size
    if os.path.exists(path):
        z = np.load(path)
        vis = np.unpackbits(z["vis_packed"], axis=1, count=n_cells).astype(bool)
        return Placement(z["site_xy"], z["site_h"], z["az_offset"], vis, z["coverage_curve"],
                         int(z["n_candidates"]), int(z["n_sa_cells"]), path)

    t0 = time.time()
    rng = np.random.default_rng(geo.placement_seed)
    cand_xy, cand_roof = find_candidates(raster, geo, rng)
    if len(cand_xy) == 0:
        raise ValueError("no rooftop candidates found; check building heights / cand_min_height_m")
    cand_h = cand_roof + geo.mast_height_m
    sa = raster.outdoor.reshape(-1)
    sa_idx = np.full(n_cells, -1, np.int64)
    sa_idx[sa] = np.arange(sa.sum())
    n_sa = int(sa.sum())
    nbytes = (n_sa + 7) // 8
    packed = np.zeros((len(cand_xy), nbytes), np.uint8)
    flats = []
    for i, (xy, h) in enumerate(zip(cand_xy, cand_h)):
        f = viewshed(raster, xy, h, radio.h_ut_m, geo.viewshed_radius_m)
        flats.append(f)
        bits = np.zeros(n_sa, bool)
        bits[sa_idx[f]] = True
        packed[i] = np.packbits(bits)
    t1 = time.time()

    # greedy max-coverage ("adds the BS that improves visibility the most")
    covered = np.zeros(nbytes, np.uint8)
    chosen, curve = [], []
    avail = np.ones(len(cand_xy), bool)
    while True:
        gain = _popcount_rows(packed & ~covered[None, :])
        gain[~avail] = -1
        j = int(np.argmax(gain))
        cov_frac = (_popcount_rows(covered[None, :])[0]) / n_sa
        if (gain[j] <= max(geo.min_gain_frac * n_sa, 0) or cov_frac >= geo.coverage_target
                or len(chosen) >= geo.max_sites):
            break
        chosen.append(j)
        avail[j] = False
        covered |= packed[j]
        curve.append(_popcount_rows(covered[None, :])[0] / n_sa)
    chosen = np.array(chosen, int)
    site_xy, site_h = cand_xy[chosen], cand_h[chosen]
    vis = np.zeros((len(chosen), n_cells), bool)
    az = np.zeros(len(chosen))
    for k, j in enumerate(chosen):
        vis[k, flats[j]] = True
        if geo.optimize_azimuth:
            az[k] = best_azimuth_offset(site_xy[k], flats[j], raster, radio, geo.sector_azimuths_deg)
    if verbose:
        print(f"[placement] {len(cand_xy)} candidates, viewsheds {t1 - t0:.1f}s, greedy picked "
              f"{len(chosen)} sites, SA visibility {curve[-1] if curve else 0:.3f} "
              f"({time.time() - t0:.1f}s total) -> cached {path}")
    pl = Placement(site_xy, site_h, az, vis, np.array(curve), len(cand_xy), n_sa, path)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, site_xy=site_xy, site_h=site_h, az_offset=az,
                            vis_packed=np.packbits(vis, axis=1), coverage_curve=pl.coverage_curve,
                            n_candidates=len(cand_xy), n_sa_cells=n_sa)
        os.replace(tmp, path)
    except OSError as e:  # pragma: no cover
        warnings.warn(f"could not write placement cache: {e}")
    return pl


def site_viewsheds(raster, site_xy, site_h, h_ut, radius, cache_dir=None) -> np.ndarray:
    """Viewsheds for arbitrary (e.g. hexagonal) sites -> bool [N, n_cells], disk-cached."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(raster.h).tobytes())
    h.update(np.ascontiguousarray(np.asarray(site_xy, float)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(site_h, float)).tobytes())
    h.update(json.dumps([CACHE_VERSION, raster.origin.tolist(), raster.res, h_ut, radius]).encode())
    cache_dir = cache_dir or default_cache_dir()
    path = os.path.join(cache_dir, f"viewsheds_{h.hexdigest()[:20]}.npz")
    if os.path.exists(path):
        return np.unpackbits(np.load(path)["vis"], axis=1, count=raster.h.size).astype(bool)
    vis = np.zeros((len(site_xy), raster.h.size), bool)
    for k, (xy, hh) in enumerate(zip(site_xy, site_h)):
        vis[k, viewshed(raster, xy, float(hh), h_ut, radius)] = True
    try:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(path, vis=np.packbits(vis, axis=1))
    except OSError:  # pragma: no cover
        pass
    return vis
