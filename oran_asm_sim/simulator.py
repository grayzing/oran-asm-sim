"""Core system-level simulation engine (no RL dependencies).

Time structure
--------------
* one *decision step* (``env.decision_interval_s``, default 1 s) = the period
  during which the per-sector ASM command is held (Near-RT RIC loop);
* each step is resolved into ``env.n_mini`` scheduling mini-periods (default
  10 x 100 ms). In each mini-period: TVC mobility moves UEs, the large-scale
  channel / association are updated, FTP-3 traffic arrives, one fast-fading
  realisation per (UE, RBG) is drawn, the PF scheduler drains buffers and the
  per-sector energy is integrated with the ASM model.

Per-sector sleep command (action)
---------------------------------
0      : no ASM - sector stays active (idle at signalling load between bursts)
1,2,3  : opportunistic SMk - sector stays on air, but enters SMk in every idle
         gap long enough (>= Table I minimum sleep). Gaps are delimited by
         traffic bursts and by SSB transmissions (20 ms). Waking up for a
         traffic burst costs the Table I activation time (service delay).
4      : SM4 - cell switch-off: deactivation 0.5 s -> deep sleep -> activation
         0.5 s when any other command is given. UEs are handed over at once;
         no SSB is transmitted while in SM4.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .channel import LargeScaleChannel, link_rates, mimo_eigs, o2i_loss_db, sample_mimo_channel, db2lin
from .config import SimConfig
from .geometry import build_deployment
from .mobility import TVCMobility
from .power import ASM_TIMING, ORUPowerModel
from .scheduler import pf_schedule
from .traffic import FTP3Traffic

ON, DEACT, OFF, ACT = 0, 1, 2, 3
STATE_NAMES = ("on", "deactivating", "sm4", "activating")


class ORANSystemSimulator:
    def __init__(self, cfg: Optional[SimConfig] = None, seed: Optional[int] = None):
        self.cfg = cfg or SimConfig()
        self.dep = build_deployment(self.cfg.geometry, self.cfg.radio)
        r = self.cfg.radio
        self.S = self.dep.n_sectors
        e = self.cfg.env
        self.U = int(e.n_ues) if e.n_ues else int(round(e.ues_per_sector * self.S))
        self.B = r.n_rbg
        self.prb_per_rbg = np.full(self.B, r.rbg_size, float)
        self.prb_per_rbg[-1] = r.n_prb - r.rbg_size * (self.B - 1)
        self.power = ORUPowerModel(self.cfg.power, r)
        self.dt_mini = self.cfg.env.decision_interval_s / self.cfg.env.n_mini
        self.slots_per_mini = self.dt_mini / r.slot_s * r.tdd_dl_fraction
        self.p_ref_total = self.S * self.power.p_full + self.dep.n_sites * self.cfg.power.site_overhead_w
        self.reset(seed)

    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None, start_hour: Optional[float] = None):
        self.rng = np.random.default_rng(seed)
        e, r = self.cfg.env, self.cfg.radio
        if start_hour is None:
            start_hour = self.rng.uniform(0, 24) if e.random_start else e.start_hour
        self.t = float(start_hour) * 3600.0
        self.mob = TVCMobility(self.cfg.mobility, self.dep, self.U, self.rng, r)
        self.traffic = FTP3Traffic(self.cfg.traffic, self.U, self.rng)
        self.chan = LargeScaleChannel(self.dep, r, self.U, self.rng)
        self.o2i_base_db, self.o2i_din = o2i_loss_db(self.rng, self.U, r.fc_ghz, self.mob.high_loss_bld,
                                                     return_parts=True)
        self.k_los = db2lin(r.rician_k_los_db)

        U, S = self.U, self.S
        self.backlog = np.zeros(U)
        self.pf_avg = np.full(U, 1e6)
        self.serving = np.full(U, -1)
        self.prev_serving = np.full(U, -1)          # cell before the last HO (ping-pong detection)
        self.t_last_ho = np.full(U, -np.inf)
        self.ue_handovers = np.zeros(U, int)        # cumulative per-UE HO count (reset after warm-up)
        self.state = np.full(S, ON)
        self.timer = np.zeros(S)
        self.asm_level = np.zeros(S, int)
        self.util = np.zeros(S)
        self.can_sleep_prev = np.zeros(S, bool)
        self.state_age = np.zeros(S)
        self.streams = np.full(S, float(r.max_layers))
        self.mod_bits = np.full(S, 6.0)
        self.wb_sinr_db = np.full(U, -10.0)
        self.G = None
        self.act_last = np.zeros(S)
        self.last = {}
        self.step_count = 0
        if e.warmup_s > 0:
            n = int(round(e.warmup_s / e.decision_interval_s))
            for _ in range(n):
                self.step(np.zeros(S, int))
            self.step_count = 0
            self.ue_handovers[:] = 0
        return self

    # ------------------------------------------------------------------ #
    def period_multiplier(self) -> float:
        return self.cfg.mobility.periods[self.mob.period].traffic_multiplier

    def _apply_actions(self, actions):
        a = np.asarray(actions, int).reshape(-1)
        assert a.shape[0] == self.S, f"expected {self.S} actions, got {a.shape[0]}"
        td4, _, ta4 = ASM_TIMING[4]
        prev = self.state.copy()
        off_cmd = a == 4
        # -> SM4
        go_off = off_cmd & ((self.state == ON) | (self.state == ACT))
        self.timer[go_off] = td4
        self.state[go_off] = DEACT
        # -> wake up
        wake = ~off_cmd & ((self.state == OFF) | (self.state == DEACT))
        self.timer[wake] = ta4 + np.where(self.state[wake] == DEACT, self.timer[wake], 0.0)
        self.state[wake] = ACT
        self.asm_level = np.where(off_cmd, 0, np.clip(a, 0, 3))
        self.state_age = np.where(self.state != prev, 0.0, self.state_age)

    # ------------------------------------------------------------------ #
    def _associate(self, G, acc=None):
        """RSRP association with hysteresis. Returns the number of handovers; if ``acc`` is
        given, the handover breakdown is accumulated into it:

        ho_mobility  serving cell still on, a neighbour is better by > hysteresis (or the
                     serving RSRP fell below ``rsrp_min_dbm``) - A3-style radio-triggered HO
        ho_forced    serving cell left the ON state (SM4 deactivation) - ASM-induced HO
        ho_intra / ho_inter   target on the same / a different site
        ho_pingpong  HO back to the previous cell within ``ho_pingpong_window_s``
        rlf          connected UE lost all coverage (no cell above ``rsrp_min_dbm``)
        reconnect    UE in outage re-attached to a cell
        ho_out_sec / ho_in_sec  per-sector source / target counts
        """
        r = self.cfg.radio
        on = self.state == ON
        rsrp = 10 * np.log10(np.maximum(G / 12.0, 1e-30)) + 30          # RSRP per RE [dBm]
        rsrp_m = np.where(on[None, :], rsrp, -np.inf)
        best = np.argmax(rsrp_m, 1)
        best_v = rsrp_m[np.arange(self.U), best]
        cur = self.serving
        cur_ok = (cur >= 0) & on[np.maximum(cur, 0)]
        cur_v = np.where(cur_ok, rsrp_m[np.arange(self.U), np.maximum(cur, 0)], -np.inf)
        keep = cur_ok & (best_v < cur_v + r.ho_hysteresis_db) & (cur_v >= r.rsrp_min_dbm)
        new = np.where(keep, cur, best)
        new = np.where(rsrp_m[np.arange(self.U), new] < r.rsrp_min_dbm, -1, new)
        is_ho = (cur >= 0) & (new >= 0) & (new != cur)
        ho = int(is_ho.sum())
        if ho:
            src, dst = cur[is_ho], new[is_ho]
            pp = (dst == self.prev_serving[is_ho]) & \
                 (self.t - self.t_last_ho[is_ho] <= r.ho_pingpong_window_s)
            self.prev_serving[is_ho] = src
            self.t_last_ho[is_ho] = self.t
            self.ue_handovers[is_ho] += 1
        if acc is not None:
            acc["ho"] += ho
            if ho:
                forced = ~cur_ok[is_ho]
                site = self.dep.sector_site
                intra = site[src] == site[dst]
                acc["ho_forced"] += int(forced.sum())
                acc["ho_mobility"] += int((~forced).sum())
                acc["ho_intra"] += int(intra.sum())
                acc["ho_inter"] += int((~intra).sum())
                acc["ho_pingpong"] += int(pp.sum())
                acc["ho_out_sec"] += np.bincount(src, minlength=self.S)
                acc["ho_in_sec"] += np.bincount(dst, minlength=self.S)
            acc["rlf"] += int(np.sum((cur >= 0) & (new < 0)))
            acc["reconnect"] += int(np.sum((cur < 0) & (new >= 0)))
        self.serving = new
        return ho

    # ------------------------------------------------------------------ #
    def step(self, actions) -> Dict[str, np.ndarray]:
        cfg, r, pw = self.cfg, self.cfg.radio, self.power
        S, U, B, dt = self.S, self.U, self.B, self.dt_mini
        self._apply_actions(actions)
        P_idle = pw.p_idle
        mix = cfg.power.transition_mix
        P_sm = np.array([P_idle] + [pw.sleep_power[k] for k in (1, 2, 3, 4)])
        P_tr = mix * P_idle + (1 - mix) * P_sm                      # transition power per level
        t_d = np.array([0.0] + [ASM_TIMING[k][0] for k in (1, 2, 3)])
        t_a = np.array([0.0] + [ASM_TIMING[k][2] for k in (1, 2, 3)])

        acc = dict(energy=np.zeros(S), served=np.zeros(U), offered=np.zeros(U), dropped=0.0,
                   outage=0.0, qos=0.0, ho=0, busy=np.zeros(S), sleep_t=np.zeros(S),
                   occ7=np.zeros((S, 7)), wake_lat=np.zeros(S), n_bursts=np.zeros(S),
                   served_sec=np.zeros(S), blocked=0.0,
                   ho_mobility=0, ho_forced=0, ho_intra=0, ho_inter=0, ho_pingpong=0, rlf=0,
                   reconnect=0, ho_out_sec=np.zeros(S, int), ho_in_sec=np.zeros(S, int))
        demand_mask = np.zeros(U, bool)

        ls_every = max(1, cfg.env.n_mini // max(1, cfg.env.ls_updates_per_step))
        for i_mini in range(cfg.env.n_mini):
            # ---------------- mobility + large-scale channel + association
            xy = self.mob.step(self.t, dt)
            geo_bld = self.dep.buildings is not None and r.los_model == "geometry"
            if geo_bld:   # indoor <=> inside an imported footprint; d_in = distance to facade
                indoor = self.dep.buildings.is_indoor(xy)
                d_in = np.minimum(self.dep.buildings.indoor_distance(xy), 30.0)
            else:
                indoor = self.mob.indoor
                d_in = self.o2i_din
            if i_mini % ls_every == 0 or self.G is None:
                o2i = self.o2i_base_db + 0.5 * d_in
                extra = np.where(indoor, o2i, 0.0) + np.where(self.mob.in_car & ~indoor, self.mob.car_loss_db, 0.0)
                self.G = self.chan.update(xy, extra)
                self._indoor = indoor
            G, indoor = self.G, self._indoor
            self._associate(G, acc)
            serv = self.serving
            cov = serv >= 0

            # ---------------- traffic
            n_files, bits = self.traffic.arrivals(self.t, dt, self.period_multiplier())
            acc["offered"] += bits
            acc["blocked"] += bits[~cov].sum()          # no connection -> session blocked
            n_files = np.where(cov, n_files, 0)
            self.backlog += np.where(cov, bits, 0.0)
            demand_mask |= self.backlog > 0

            # ---------------- interference (load-coupled, per-RBG activity)
            on = self.state == ON
            act = np.where(on, np.maximum(self.util, cfg.power.signalling_load), 0.0)
            self.act_last = act
            mask = (self.rng.random((S, B)) < act[:, None]).astype(float)
            if r.interference_fading:
                Iub = np.einsum("us,sb,usb->ub", G, mask, self.rng.exponential(1.0, (U, S, B)))
            else:
                Iub = G @ mask
            sidx = np.maximum(serv, 0)
            g_srv = G[np.arange(U), sidx]
            own = g_srv[:, None] * mask[sidx]
            I = np.maximum(Iub - own, 0.0) + self.chan.noise_prb_w
            rho = np.where(cov[:, None], g_srv[:, None] / I, 0.0)

            # ---------------- 2x2 SU-MIMO link abstraction (only for backlogged, covered UEs)
            self.wb_sinr_db = 10 * np.log10(np.maximum(rho.mean(1), 1e-12))
            idx = np.where(cov & (self.backlog > 0))[0]
            rates = np.zeros((idx.size, B))
            rank = np.zeros(idx.size)
            mod = np.zeros((idx.size, B))
            if idx.size:
                k_lin = np.where(self.chan.los_sector[idx, sidx[idx]] & ~indoor[idx], self.k_los, 0.0)
                H = sample_mimo_channel(self.rng, idx.size, B, r.n_rx_ue, r.n_tx_ports, k_lin)
                bits_slot, rank, mod = link_rates(rho[idx], mimo_eigs(H), r, self.prb_per_rbg)
                rates = bits_slot * self.slots_per_mini

            # ---------------- ASM-aware available service time
            arr_s = np.bincount(serv[cov], weights=(n_files[cov] > 0).astype(float), minlength=S)
            lvl = np.where(on, self.asm_level, 0)
            n_t_est = arr_s * (1.0 - np.clip(self.util, 0, 1))
            wake_frac = np.where(self.can_sleep_prev & (lvl > 0), n_t_est * t_a[lvl] / dt, 0.0)
            t_avail = np.where(on, np.clip(1.0 - wake_frac, 0.2, 1.0), 0.0)

            # ---------------- PF scheduling
            served = np.zeros(U)
            busy = np.zeros(S)
            if idx.size:
                srv_i, busy, share = pf_schedule(serv[idx], rates, self.backlog[idx],
                                                 self.pf_avg[idx], t_avail, S, self.rng,
                                                 mini_s=dt, n_virtual=cfg.env.pf_virtual_slots)
                served[idx] = srv_i
                w = np.maximum(srv_i, 1e-9)
                num = np.bincount(serv[idx], weights=w, minlength=S)
                self.streams = np.where(num > 1e-6, np.bincount(serv[idx], weights=w * rank, minlength=S)
                                        / np.maximum(num, 1e-9), self.streams)
                mod_u = mod.mean(1)
                self.mod_bits = np.where(num > 1e-6, np.bincount(serv[idx], weights=w * mod_u, minlength=S)
                                         / np.maximum(num, 1e-9), self.mod_bits)
            busy = np.where(on, np.minimum(busy, t_avail), 0.0)
            self.backlog -= served
            beta = 1 - np.exp(-dt / 0.1)
            self.pf_avg = (1 - beta) * self.pf_avg + beta * served / dt
            over = np.maximum(self.backlog - cfg.traffic.max_buffer_bits, 0)
            acc["dropped"] += over.sum()
            self.backlog -= over
            acc["served"] += served
            acc["served_sec"] += np.bincount(serv[cov], weights=served[cov], minlength=S)

            # ---------------- energy integration
            idle = np.where(on, np.maximum(t_avail - busy, 0.0), 0.0)   # fraction of dt
            wake = np.where(on, 1.0 - t_avail, 0.0)
            p_busy = pw.active(np.ones(S), self.streams, self.mod_bits).total
            n_ssb = idle * dt / cfg.power.ssb_period_s
            n_t = n_t_est * (lvl > 0)
            n_g = n_ssb + n_t
            gbar = np.where(n_g > 0, idle * dt / np.maximum(n_g, 1e-12), 0.0)
            can = (lvl > 0) & (gbar >= (t_d[lvl] + t_a[lvl])) & on
            e_sleep_ssb = n_ssb * ((t_d[lvl] + t_a[lvl]) * P_tr[lvl] + (gbar - t_d[lvl] - t_a[lvl]) * P_sm[lvl])
            e_sleep_trf = n_t * (t_d[lvl] * P_tr[lvl] + (gbar - t_d[lvl]) * P_sm[lvl])
            e_idle_on = np.where(can, e_sleep_ssb + e_sleep_trf + wake * dt * P_tr[lvl],
                                 (idle + wake) * dt * P_idle)
            sleep_t = np.where(can, n_ssb * (gbar - t_d[lvl] - t_a[lvl]) + n_t * (gbar - t_d[lvl]), 0.0)
            e_on = busy * dt * p_busy + e_idle_on

            # SM4 state machine (DEACT -> OFF, ACT -> ON)
            frac = np.clip(self.timer / dt, 0.0, 1.0)
            P_sm4, P_tr4 = P_sm[4], P_tr[4]
            e_deact = dt * (frac * P_tr4 + (1 - frac) * P_sm4)
            e_act = dt * (frac * P_tr4 + (1 - frac) * P_idle)
            energy = np.select([self.state == ON, self.state == DEACT, self.state == OFF, self.state == ACT],
                               [e_on, e_deact, dt * P_sm4, e_act])
            sleep_t = sleep_t + np.select([self.state == DEACT, self.state == OFF],
                                          [(1 - frac) * dt, np.full(S, dt)], 0.0)
            acc["energy"] += energy
            acc["busy"] += busy * dt
            acc["sleep_t"] += sleep_t
            occ_col = np.where(on, lvl, 3 + self.state)        # 0..3 = on@SMk, 4 deact, 5 sm4, 6 act
            occ_col = np.minimum(occ_col, 6)
            acc["occ7"] += np.eye(7)[occ_col] * dt
            acc["wake_lat"] += np.where(can, n_t * t_a[lvl], 0.0)
            acc["n_bursts"] += np.where(on, arr_s, 0.0)

            in_tr = (self.state == DEACT) | (self.state == ACT)
            self.timer = np.where(in_tr, self.timer - dt, self.timer)
            done = in_tr & (self.timer <= 1e-9)
            self.state = np.where(done & (self.state == DEACT), OFF, self.state)
            self.state = np.where(done & (self.state == ACT), ON, self.state)
            self.state_age += dt

            self.util = busy
            self.can_sleep_prev = can

            # ---------------- QoS bookkeeping
            q_delay = np.where(self.backlog > 0, self.backlog / np.maximum(self.pf_avg, 1e3), 0.0)
            wl = np.where(cov, np.where(self.can_sleep_prev[sidx], t_a[lvl][sidx], 0.0), 0.0)
            viol = cov & (self.backlog > 0) & (q_delay + wl > cfg.traffic.delay_budget_s)
            acc["qos"] += viol.mean()
            acc["outage"] += (~cov).mean()
            self.t += dt

        n = cfg.env.n_mini
        T = cfg.env.decision_interval_s
        p_sector = acc["energy"] / T
        p_total = p_sector.sum() + self.dep.n_sites * cfg.power.site_overhead_w
        tput = acc["served"] / T
        dem = demand_mask & (self.serving >= 0)
        out = dict(
            time_s=self.t, hour=(self.t / 3600) % 24, period=self.mob.period,
            power_sector_w=p_sector, power_total_w=p_total, energy_j=p_total * T,
            power_ref_w=self.p_ref_total, power_norm=p_total / self.p_ref_total,
            offered_bps=acc["offered"].sum() / T, served_bps=tput.sum(),
            served_sector_bps=acc["served_sec"] / T,
            ue_offered_bps=acc["offered"] / T, ue_served_bps=tput,
            dropped_bits=acc["dropped"], drop_frac=acc["dropped"] / max(acc["offered"].sum(), 1.0),
            blocked_frac=acc["blocked"] / max(acc["offered"].sum(), 1.0),
            outage_frac=acc["outage"] / n, qos_violation_frac=acc["qos"] / n, handovers=acc["ho"],
            ho_mobility=acc["ho_mobility"], ho_forced=acc["ho_forced"],
            ho_intra_site=acc["ho_intra"], ho_inter_site=acc["ho_inter"],
            ho_pingpong=acc["ho_pingpong"], ho_pingpong_frac=acc["ho_pingpong"] / max(acc["ho"], 1),
            rlf=acc["rlf"], reconnects=acc["reconnect"],
            ho_rate_per_ue_s=acc["ho"] / (U * T),
            ho_out_sector=acc["ho_out_sec"], ho_in_sector=acc["ho_in_sec"],
            ue_tput_mean_bps=float(tput[dem].mean()) if dem.any() else 0.0,
            ue_tput_p5_bps=float(np.percentile(tput[dem], 5)) if dem.any() else 0.0,
            util_sector=acc["busy"] / T, sleep_frac_sector=acc["sleep_t"] / T,
            occupancy_sector=acc["occ7"] / T,            # cols: noASM, SM1, SM2, SM3, deact, SM4, act
            wake_latency_s=acc["wake_lat"].sum() / max(acc["n_bursts"].sum(), 1.0),
            backlog_bits=self.backlog.sum(),
            n_ue_sector=np.bincount(self.serving[self.serving >= 0], minlength=S),
            state_sector=self.state.copy(),
        )
        self.last = out
        self.step_count += 1
        return out

    # ------------------------------------------------------------------ #
    def sector_offered_bps(self) -> np.ndarray:
        cov = self.serving >= 0
        lam = self.traffic.mean_offered_bps(self.t, self.period_multiplier())
        return np.bincount(self.serving[cov], weights=lam[cov], minlength=self.S)

    def sector_backlog(self) -> np.ndarray:
        cov = self.serving >= 0
        return np.bincount(self.serving[cov], weights=self.backlog[cov], minlength=self.S)

    def sector_mean_sinr_db(self) -> np.ndarray:
        cov = self.serving >= 0
        num = np.bincount(self.serving[cov], weights=self.wb_sinr_db[cov], minlength=self.S)
        cnt = np.bincount(self.serving[cov], minlength=self.S)
        return np.where(cnt > 0, num / np.maximum(cnt, 1), 0.0)

    def sinr_toward(self, ue_idx: np.ndarray, sec_idx: np.ndarray) -> np.ndarray:
        """Wideband SINR [dB] UE u would get from sector s under the current interference
        (for an un-served pair: as if s were on and serving u)."""
        if self.G is None:
            return np.full(len(ue_idx), -10.0)
        g = self.G[ue_idx, sec_idx]
        itot = self.G[ue_idx] @ self.act_last
        i = np.maximum(itot - g * self.act_last[sec_idx], 0.0) + self.chan.noise_prb_w
        return 10 * np.log10(np.maximum(g / i, 1e-12))

    def best_sector_if_all_on(self) -> np.ndarray:
        """Sector each UE would pick if every cell were on (for coverage-aware policies)."""
        return np.argmax(self.G, 1) if self.G is not None else np.zeros(self.U, int)
