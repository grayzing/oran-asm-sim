"""Vectorised proportional-fair PRB (RBG) scheduler with buffer draining.

Each scheduling mini-period stands for many slots that share one sampled
fast-fading realisation. Within it the PF rule is emulated over a few virtual
slots: each RBG goes to the backlogged UE maximising r_{u,b} / T_u and the PF
averages T_u are updated, which yields PF time-sharing with multi-user
(frequency) diversity rather than a single winner. When a UE drains its
buffer part-way through the mini-period its RBGs are re-assigned by PF among the
remaining backlogged UEs (multi-pass "drain" loop). This yields both served bits
and the exact busy time of every sector, which the ASM energy accounting needs.
"""
from __future__ import annotations

import numpy as np


def _pf_alloc(serv, rates, avg, active, n_sec, n_virtual=8):
    """PF allocation over one mini-period, emulated with ``n_virtual`` virtual
    slots: in each, every RBG goes to argmax_u r_ub / T_u among the sector's
    active UEs and T_u (PF average, same units as ``rates``) is updated with the
    rate obtained. Returns per-UE rate [U] and RBG-time share x [U,B]."""
    U, B = rates.shape
    ok = active[:, None] & (rates > 0)
    s_idx = np.broadcast_to(serv[:, None], (U, B))[ok]
    b_idx = np.broadcast_to(np.arange(B)[None, :], (U, B))[ok]
    x = np.zeros((U, B))
    T = np.maximum(avg.astype(float), 1e-9)
    beta = 1.0 / max(n_virtual, 1)
    for _ in range(n_virtual):
        m = np.where(ok, rates / T[:, None], -np.inf)
        mx = np.full((n_sec, B), -np.inf)
        np.maximum.at(mx, (s_idx, b_idx), m[ok])
        win = ok & (m >= mx[serv])
        # split exact ties evenly
        cnt = np.zeros((n_sec, B))
        np.add.at(cnt, (s_idx, b_idx), win[ok].astype(float))
        w = win / np.maximum(cnt[serv], 1.0)
        x += w * beta
        T = (1 - beta) * T + beta * (w * rates).sum(1)
    return (x * rates).sum(1), x


def pf_schedule(serv, rates, backlog, pf_avg, t_avail, n_sec, rng=None, max_passes=4,
                mini_s=0.1, n_virtual=8):
    """
    serv     [U] serving sector (>=0)        rates  [U,B] bits per full mini-period per RBG
    backlog  [U] bits                        pf_avg [U] bps (PF averaged throughput)
    t_avail  [S] fraction of the mini-period available for data (wake-up overheads removed)
    Returns served [U] bits, busy [S] fraction, prb_share [U] (fraction of RBG-time).
    """
    U, B = rates.shape
    served = np.zeros(U)
    share = np.zeros(U)
    busy = np.zeros(n_sec)
    if U == 0:
        return served, busy, share
    bl = backlog.astype(float).copy()
    rem = t_avail.astype(float).copy()
    # PF average expressed in bits per mini-period (rates' unit); pf_avg is bps
    avg = np.maximum(pf_avg, 1.0) * mini_s
    for p in range(max_passes + 1):
        active = bl > 1e-6
        R, win = _pf_alloc(serv, rates, avg, active, n_sec, n_virtual)
        has = np.zeros(n_sec, bool)
        np.logical_or.at(has, serv, active & (R > 0))
        if not np.any(has & (rem > 1e-12)):
            break
        if p < max_passes:
            tdrain = np.where(R > 0, bl / np.maximum(R, 1e-12), np.inf)
            tmin = np.full(n_sec, np.inf)
            np.minimum.at(tmin, serv, np.where(active, tdrain, np.inf))
            dt = np.where(has, np.minimum(tmin, rem), 0.0)
            got = np.minimum(R * dt[serv], bl)
        else:
            # last pass: work-conserving closure - the sector keeps serving its remaining
            # backlog at its current aggregate rate (re-allocating as UEs drain)
            ok = active & (R > 0)
            tot_bl = np.bincount(serv, weights=np.where(ok, bl, 0.0), minlength=n_sec)
            tot_R = np.bincount(serv, weights=np.where(ok, R, 0.0), minlength=n_sec)
            need = np.where(tot_R > 0, tot_bl / np.maximum(tot_R, 1e-12), 0.0)
            dt = np.where(has, np.minimum(need, rem), 0.0)
            frac = np.where(need > 0, np.minimum(rem / np.maximum(need, 1e-12), 1.0), 0.0)
            got = np.where(ok, bl * frac[serv], 0.0)
        served += got
        bl -= got
        share += win.sum(1) / B * dt[serv]
        avg = 0.5 * avg + 0.5 * np.where(dt[serv] > 0, R, avg)
        busy += dt
        rem -= dt
    return served, busy, share
