"""Vectorised Time-Variant Community (TVC) mobility model [HSU07].

Faithful to Sec. III-B of the paper:
  * time is split into period types (NMP / CMP); each UE has a community per
    period (random square, or a fixed point-of-interest as in the Sec. V
    extension), giving skewed location preference and periodic re-appearance;
  * within a period a UE alternates local / roaming epochs following the
    2-state Markov chain of Fig. 3 (p_r: local->roaming, p_l: roaming->local);
  * each epoch: speed ~ U[vmin, vmax], direction ~ U[0, 2pi), random-direction
    movement of length ~ Exp(L), then pause ~ U[0, Tmax];
  * torus boundaries (re-insert from the other side) for the whole area in
    roaming epochs and for the community square in local epochs.

One deviation, needed for a cellular simulator (handover continuity): when a
local epoch starts while the UE is outside its community (e.g. after the period
changed) the UE first walks/drives straight to a random point of the community
instead of being teleported; the local epoch starts on arrival.
"""
from __future__ import annotations

import numpy as np

from .config import MobilityConfig, RadioConfig

LOCAL, ROAM = 0, 1


class TVCMobility:
    def __init__(self, cfg: MobilityConfig, dep, n_ue: int, rng: np.random.Generator,
                 radio: RadioConfig):
        self.cfg, self.dep, self.n, self.rng, self.radio = cfg, dep, n_ue, rng, radio
        self.lo, self.hi = dep.area_min.astype(float), dep.area_max.astype(float)
        self.size = self.hi - self.lo
        self.period_names = list(cfg.periods.keys())
        self._sched = sorted(cfg.schedule)
        self.is_vehicle = rng.random(n_ue) < cfg.vehicle_fraction
        self.high_loss_bld = rng.random(n_ue) < radio.high_loss_building_prob
        self.car_loss_db = np.maximum(
            rng.normal(radio.car_penetration_mu_db, radio.car_penetration_sigma_db, n_ue), 0)

        # community centre per (period, UE): fixed for the whole run -> periodic re-appearance
        self.centers = {}
        for name, p in cfg.periods.items():
            half = p.community_size_m / 2
            if p.community_mode == "poi" and len(dep.poi_xy):
                w = np.array([dep.poi_weights[i] if set(t) & set(p.poi_tags) else 0.0
                              for i, t in enumerate(dep.poi_tags)])
                if w.sum() == 0:
                    w = dep.poi_weights.copy()
                idx = rng.choice(len(w), n_ue, p=w / w.sum())
                c = dep.poi_xy[idx]
            else:
                c = rng.uniform(self.lo + half, self.hi - half, (n_ue, 2))
            self.centers[name] = np.clip(c, self.lo + half, self.hi - half)
        self.reset(0.0)

    # ------------------------------------------------------------------ #
    def period_at(self, t_s: float) -> str:
        h = (t_s / 3600.0) % 24.0
        for a, b, name in self._sched:
            if a <= h < b:
                return name
        return self._sched[-1][2]

    def reset(self, t_s: float):
        n, rng = self.n, self.rng
        self.period = self.period_at(t_s)
        p = self.cfg.periods[self.period]
        self.comm = self.centers[self.period].copy()
        self.csize = p.community_size_m
        pi_r = p.p_r / (p.p_r + p.p_l)                  # [HSU07] Lemma 3.1
        self.mode = np.where(rng.random(n) < pi_r, ROAM, LOCAL)
        local_xy = self.comm + rng.uniform(-0.5, 0.5, (n, 2)) * self.csize
        roam_xy = rng.uniform(self.lo, self.hi, (n, 2))
        self.xy = np.where((self.mode == LOCAL)[:, None], local_xy, roam_xy)
        self.moving = np.zeros(n, bool)
        self.transit = np.zeros(n, bool)
        self.target = self.xy.copy()
        self.heading = np.zeros((n, 2))
        self.speed = np.zeros(n)
        self.remaining = np.zeros(n)
        tmax = np.where(self.mode == LOCAL, p.t_max_pause_local_s, p.t_max_pause_roam_s)
        self.pause_left = rng.uniform(0, tmax)
        self.indoor = rng.random(n) < self.radio.paused_indoor_prob

    # ------------------------------------------------------------------ #
    def _speeds(self, k, roam_mask):
        lo_p, hi_p = self.cfg.pedestrian_speed_mps
        lo_v, hi_v = self.cfg.vehicle_speed_mps
        veh = self.is_vehicle[k] & roam_mask
        return np.where(veh, self.rng.uniform(lo_v, hi_v, len(k)), self.rng.uniform(lo_p, hi_p, len(k)))

    def _in_community(self, k):
        return np.all(np.abs(self.xy[k] - self.comm[k]) <= self.csize / 2 + 1e-6, axis=1)

    def step(self, t_s: float, dt: float):
        rng, p_name = self.rng, self.period_at(t_s)
        if p_name != self.period:                       # period boundary: new communities
            self.period = p_name
            self.comm = self.centers[p_name].copy()
            self.csize = self.cfg.periods[p_name].community_size_m
        p = self.cfg.periods[self.period]

        # ---------------- moving UEs
        mv = np.where(self.moving)[0]
        if mv.size:
            adv = np.minimum(self.speed[mv] * dt, self.remaining[mv])
            tr = self.transit[mv]
            # transit legs aim at a fixed target
            if tr.any():
                k = mv[tr]
                vec = self.target[k] - self.xy[k]
                nrm = np.maximum(np.hypot(*vec.T), 1e-9)
                self.heading[k] = vec / nrm[:, None]
            self.xy[mv] += self.heading[mv] * adv[:, None]
            self.remaining[mv] -= adv
            # torus boundaries
            roam = mv[(self.mode[mv] == ROAM) & ~tr]
            self.xy[roam] = self.lo + np.mod(self.xy[roam] - self.lo, self.size)
            loc = mv[(self.mode[mv] == LOCAL) & ~tr]
            c0 = self.comm[loc] - self.csize / 2
            self.xy[loc] = c0 + np.mod(self.xy[loc] - c0, self.csize)

            done = mv[self.remaining[mv] <= 1e-9]
            # transit finished -> start the actual local epoch
            td = done[self.transit[done]]
            if td.size:
                self.transit[td] = False
                ang = rng.uniform(0, 2 * np.pi, td.size)
                self.heading[td] = np.stack([np.cos(ang), np.sin(ang)], 1)
                self.speed[td] = self._speeds(td, np.zeros(td.size, bool))
                self.remaining[td] = rng.exponential(p.mean_epoch_local_m, td.size)
            ed = done[~self.transit[done] & ~np.isin(done, td)]
            if ed.size:                                 # epoch over -> pause
                self.moving[ed] = False
                tmax = np.where(self.mode[ed] == LOCAL, p.t_max_pause_local_s, p.t_max_pause_roam_s)
                self.pause_left[ed] = rng.uniform(0, tmax)
                self.indoor[ed] = rng.random(ed.size) < self.radio.paused_indoor_prob

        # ---------------- paused UEs
        ps = np.where(~self.moving)[0]
        if ps.size:
            self.pause_left[ps] -= dt
            st = ps[self.pause_left[ps] <= 0]
            if st.size:
                # 2-state Markov chain of [HSU07] Fig. 3
                u = rng.random(st.size)
                was_local = self.mode[st] == LOCAL
                new_mode = np.where(was_local, np.where(u < p.p_r, ROAM, LOCAL),
                                    np.where(u < p.p_l, LOCAL, ROAM))
                self.mode[st] = new_mode
                ang = rng.uniform(0, 2 * np.pi, st.size)
                self.heading[st] = np.stack([np.cos(ang), np.sin(ang)], 1)
                roam_m = new_mode == ROAM
                self.speed[st] = self._speeds(st, roam_m)
                L = np.where(roam_m, p.mean_epoch_roam_m, p.mean_epoch_local_m)
                self.remaining[st] = rng.exponential(L)
                self.moving[st] = True
                self.indoor[st] = False
                # local epoch but outside the community -> transit first
                lk = st[~roam_m]
                out = lk[~self._in_community(lk)]
                if out.size:
                    self.transit[out] = True
                    self.target[out] = self.comm[out] + rng.uniform(-0.5, 0.5, (out.size, 2)) * self.csize
                    self.remaining[out] = np.hypot(*(self.target[out] - self.xy[out]).T)
                    self.speed[out] = self._speeds(out, np.ones(out.size, bool))

        self.in_car = self.moving & self.is_vehicle & ((self.mode == ROAM) | self.transit)
        return self.xy
