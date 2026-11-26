"""24 h downtown-LA study of the non-learning baselines.

For every hour of the day the network is simulated at full resolution (1 s
decisions, 100 ms scheduling mini-periods) for a ``--window`` of seconds after
a short always-on warm-up; daily energy is the sum of hourly mean power x 1 h.

    python examples/run_baselines.py --out results             # all policies
    python examples/run_baselines.py --out results --policies 0,2 --no-plots   # (parallel shards)
    python examples/run_baselines.py --out results --plot-only

Outputs: summary.json, hourly.csv, daily_profile.png, kpis_hourly.png, dtla_map.png
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from oran_asm_sim import ORanASMEnv  # noqa: E402
from oran_asm_sim.baselines import AlwaysOn, FixedASM, LoadThresholdSM4, run_policy  # noqa: E402
from oran_asm_sim.config import PRESETS  # noqa: E402

ALL = [AlwaysOn(), FixedASM(1), FixedASM(2), FixedASM(3), LoadThresholdSM4()]
KEYS = ("power_total_w", "power_fixed_w", "power_ctrl_w", "offered_bps", "served_bps", "qos_violation_frac", "outage_frac",
        "blocked_frac", "drop_frac", "wake_latency_s", "ue_tput_mean_bps", "ue_tput_p5_bps",
        "handovers", "ho_mobility", "ho_forced", "ho_intra_site", "ho_inter_site", "ho_pingpong",
        "rlf", "ho_rate_per_ue_s", "reward")


def simulate(args, policies):
    cfg = {k: dict(v) for k, v in PRESETS[args.preset].items()}
    cfg.setdefault("env", {}).update(seed=args.seed, warmup_s=args.warmup)
    env = ORanASMEnv(cfg)
    res = {}
    for pol in policies:
        t0 = time.time()
        hours = []
        for h in range(24):
            rows = run_policy(env, pol, args.window, seed=args.seed + h, start_hour=float(h))
            hours.append({"hour": h, "period": rows[-1]["period"],
                          **{k: float(np.mean([r[k] for r in rows])) for k in KEYS}})
        res[pol.name] = hours
        e = sum(x["power_total_w"] for x in hours) / 1e3
        print(f"{pol.name:26s} {time.time() - t0:6.1f}s  E={e:7.1f} kWh/day  "
              f"QoS-viol={np.mean([x['qos_violation_frac'] for x in hours]):.4f}", flush=True)
    return env, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--preset", default="dtla_full", choices=list(PRESETS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--window", type=int, default=60, help="simulated seconds per hour")
    ap.add_argument("--warmup", type=float, default=20.0)
    ap.add_argument("--policies", default="0,1,2,3,4")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.plot_only:
        res = {}
        for f in sorted(glob.glob(os.path.join(args.out, "rows_*.json"))):
            res.update(json.load(open(f)))
        res = {p.name: res[p.name] for p in ALL if p.name in res}
        env = ORanASMEnv({**{k: dict(v) for k, v in PRESETS[args.preset].items()}})
    else:
        pols = [ALL[int(i)] for i in args.policies.split(",")]
        env, res = simulate(args, pols)
        with open(os.path.join(args.out, f"rows_{args.policies.replace(',', '')}.json"), "w") as f:
            json.dump(res, f)
        if args.no_plots:
            return

    names = list(res)
    base = sum(x["power_total_w"] for x in res[names[0]])
    summary = {}
    for name, hours in res.items():
        e = sum(x["power_total_w"] for x in hours)
        summary[name] = dict(
            energy_kwh_per_day=e / 1e3, saving_vs_always_on=1 - e / base,
            qos_violation=float(np.mean([x["qos_violation_frac"] for x in hours])),
            busy_hour_qos_violation=float(max(x["qos_violation_frac"] for x in hours)),
            outage=float(np.mean([x["outage_frac"] for x in hours])),
            served_over_offered=float(sum(x["served_bps"] for x in hours) / sum(x["offered_bps"] for x in hours)),
            macro_energy_kwh_per_day=float(sum(x.get("power_fixed_w", 0.0) for x in hours) / 1e3),
            mmwave_energy_kwh_per_day=float(sum(x.get("power_ctrl_w", 0.0) for x in hours) / 1e3),
            mean_wake_latency_ms=float(1e3 * np.mean([x["wake_latency_s"] for x in hours])),
            ue_tput_mean_mbps=float(np.mean([x["ue_tput_mean_bps"] for x in hours]) / 1e6),
            ue_tput_p5_mbps=float(np.mean([x["ue_tput_p5_bps"] for x in hours]) / 1e6),
            # per-step means x steps/hour (1 s decisions) -> HOs per simulated hour, averaged over the day
            ho_per_ue_hour=float(3600 * np.mean([x["ho_rate_per_ue_s"] for x in hours])),
            ho_forced_share=float(sum(x["ho_forced"] for x in hours) / max(sum(x["handovers"] for x in hours), 1e-9)),
            ho_pingpong_share=float(sum(x["ho_pingpong"] for x in hours) / max(sum(x["handovers"] for x in hours), 1e-9)),
            rlf_per_hour=float(3600 * np.mean([x["rlf"] for x in hours]) / env.cfg.env.decision_interval_s),
            mean_reward=float(np.mean([x["reward"] for x in hours])))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "hourly.csv"), "w") as f:
        f.write("policy,hour,period," + ",".join(KEYS) + "\n")
        for name, hours in res.items():
            for x in hours:
                f.write(f"{name},{x['hour']},{x['period']}," + ",".join(f"{x.get(k, float('nan')):.6g}" for k in KEYS) + "\n")
    print(json.dumps(summary, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from oran_asm_sim.viz import plot_network
    except ImportError:
        return
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # same order as kpis_hourly.png
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for (name, hours), c in zip(res.items(), colors):
        h = [x["hour"] + 0.5 for x in hours]
        ax[0].step(h, [x["power_total_w"] / 1e3 for x in hours], where="mid", c=c, lw=1.4, label=name)
        ax[2].step(h, [100 * x["qos_violation_frac"] for x in hours], where="mid", c=c, lw=1.2)
    hours = res[names[0]]
    ax[1].bar([x["hour"] + 0.5 for x in hours], [x["offered_bps"] / 1e9 for x in hours], width=0.8, color="0.6")
    ax[0].set_ylabel("network power [kW]")
    ax[0].set_ylim(bottom=0)
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].set_ylabel("offered DL [Gb/s]")
    ax[2].set_ylabel("UEs over delay\nbudget [%]")
    ax[2].set_xlabel("hour of day (America/Los_Angeles)")
    ax[2].set_xticks(range(0, 25, 3))
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"Downtown LA - {env.S} sectors, {env.sim.U} UEs - per-sector ASM baselines "
                 f"(hourly {args.window} s snapshots)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "daily_profile.png"), dpi=150)
    plot_kpis(res, env, args)

    env2 = ORanASMEnv({"env": {"seed": args.seed, "start_hour": 3.0, "warmup_s": 60}})
    cmd = LoadThresholdSM4()(env2)
    env2.step(cmd)
    cmd_all = np.zeros(env2.sim.S, int)               # macro sectors: always on
    cmd_all[env2.ctrl] = cmd
    fig, ax = plt.subplots(figsize=(9, 7))
    dep = env2.sim.dep
    plot_network(env2.sim, ax, cmd=cmd_all, title=f"Downtown LA bbox: {int((dep.site_tier == 0).sum())} macro + {int((dep.site_tier == 1).sum())} mmWave sites "
                                              f"(layout={env2.cfg.geometry.layout}, buildings: {dep.buildings_source}), "
                                              "UEs at 03:00, threshold-policy commands per sector")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "dtla_map.png"), dpi=150)


def plot_kpis(res, env, args):
    """Hourly KPIs per policy: energy, handovers, outage, UE throughput (one panel each,
    shared hour axis; one y-scale per panel)."""
    import matplotlib.pyplot as plt
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]     # fixed categorical order
    markers = ["o", "s", "^", "D", "v"]                                  # secondary encoding (CVD / print)
    plt.rcParams.update({"axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    ax = ax.ravel()
    T = env.cfg.env.decision_interval_s
    for (name, hours), c, mk in zip(res.items(), colors, markers):
        h = np.array([x["hour"] + 0.5 for x in hours])
        kw = dict(c=c, lw=2, marker=mk, ms=4, markevery=3, label=name)
        ax[0].plot(h, [x["power_total_w"] / 1e3 for x in hours], **kw)          # mean kW over 1 h = kWh
        ax[1].plot(h, [3600 / T * x["handovers"] / env.sim.U for x in hours], **kw)
        ax[2].plot(h, [100 * x["outage_frac"] for x in hours], **kw)
        ax[3].plot(h, [x["ue_tput_mean_bps"] / 1e6 for x in hours], **kw)
        ax[3].plot(h, [x["ue_tput_p5_bps"] / 1e6 for x in hours], c=c, lw=1.2, ls="--", marker=mk, ms=3,
                   markevery=3)
    hours = next(iter(res.values()))
    if all("power_fixed_w" in x for x in hours):
        h = [x["hour"] + 0.5 for x in hours]
        ax[0].fill_between(h, 0, [x["power_fixed_w"] / 1e3 for x in hours], color="#c3c2b7", alpha=0.5,
                           lw=0, label=f"macro share ({next(iter(res))})")
    ax[0].set_ylabel("energy per hour [kWh]")
    ax[0].set_ylim(bottom=0)
    ax[1].set_ylabel("handovers per UE per hour")
    ax[1].set_ylim(bottom=0)
    ax[2].set_ylabel("UEs in outage [%]")
    ax[2].set_ylim(bottom=0)
    ax[3].set_ylabel("UE throughput [Mb/s]\n(solid: mean, dashed: 5th pct.)")
    ax[3].set_yscale("symlog", linthresh=0.1, linscale=0.5)          # log above 0.1 Mb/s, keeps zeros visible
    ax[3].set_ylim(bottom=0)
    titles = ["Energy", "Handovers", "Outage", "UE throughput (UEs with data)"]
    for a, t in zip(ax, titles):
        a.set_title(t, loc="left", fontsize=10, color=ink)
        a.grid(color=grid, lw=0.8)
        a.set_axisbelow(True)
        a.set_xticks(range(0, 25, 3))
        a.set_xlim(0, 24)
    for a in ax[2:]:
        a.set_xlabel("hour of day (America/Los_Angeles)")
    ax[0].legend(fontsize=7, frameon=False, ncol=2)
    dep = env.sim.dep
    fig.suptitle(f"Downtown LA HetNet - {int((dep.site_tier == 0).sum())} macro + "
                 f"{int((dep.site_tier == 1).sum())} mmWave O-RUs, {env.sim.U} UEs - "
                 f"baseline policies (hourly {args.window} s snapshots)", fontsize=10, color=ink)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "kpis_hourly.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
