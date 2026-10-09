#!/usr/bin/env python3
"""
UTD underground parking garage simulator.

    python main.py model          3-D CAD model, floor plans, STL / OBJ / DXF exports
    python main.py traffic        simulate one class day (queues, occupancy, CO, ventilation)
    python main.py animate        animated top-down replay of a day (GIF)
    python main.py maintenance    20-year maintenance & lifecycle run for one policy
    python main.py compare        Monte-Carlo trade study of maintenance policies
    python main.py all            everything above
    python main.py config         write params.json to edit, then use --config params.json

Add --show to open interactive windows (the 3-D model can be rotated with the mouse).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time


def _common() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", help="JSON parameter file (create one with `python main.py config`)")
    p.add_argument("--out", default="outputs", help="output folder (default: outputs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show", action="store_true", help="open interactive plot windows")
    return p


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = _common()

    m = sub.add_parser("model", parents=[c], help="build the 3-D model and CAD exports")
    m.add_argument("--explode", type=float, default=9.0, help="gap (m) between levels in the exploded view")

    for name, hlp in (("traffic", "simulate one day"), ("animate", "animate one day")):
        t = sub.add_parser(name, parents=[c], help=hlp)
        t.add_argument("--demand", type=float, help="vehicles trying to enter (default from config)")
        t.add_argument("--closed-stalls", type=int, default=0, help="close this many random stalls (maintenance)")
        t.add_argument("--fans-down", type=int, nargs="*", default=None, metavar="N",
                       help="failed exhaust fans per level, e.g. --fans-down 0 3 0")
        t.add_argument("--ventilation", choices=["demand", "constant"])
        if name == "animate":
            t.add_argument("--fps", type=int, default=15)
            t.add_argument("--step", type=float, default=3.0, help="simulated minutes per frame")
            t.add_argument("--format", choices=["gif", "mp4"], default="gif")

    mt = sub.add_parser("maintenance", parents=[c], help="multi-year lifecycle run")
    mt.add_argument("--policy", default="preventive", help="reactive | preventive | predictive | all")
    mt.add_argument("--years", type=int)

    cp = sub.add_parser("compare", parents=[c], help="Monte-Carlo comparison of policies")
    cp.add_argument("--reps", type=int, default=20)
    cp.add_argument("--years", type=int)
    cp.add_argument("--policies", nargs="*")

    a = sub.add_parser("all", parents=[c], help="run everything")
    a.add_argument("--reps", type=int, default=15)

    cf = sub.add_parser("config", help="write the default parameters to a JSON file")
    cf.add_argument("--path", default="params.json")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(argv if argv is not None else sys.argv[1:])
    import matplotlib
    if not getattr(args, "show", False):
        matplotlib.use("Agg")

    from garage_sim.config import SimConfig
    if args.cmd == "config":
        SimConfig().to_json(args.path)
        print(f"Wrote {args.path}. Edit it, then run e.g.  python main.py all --config {args.path}")
        return

    cfg = SimConfig.from_json(args.config) if args.config else SimConfig()
    os.makedirs(args.out, exist_ok=True)
    cmds = ["model", "traffic", "animate", "maintenance", "compare"] if args.cmd == "all" else [args.cmd]
    for cmd in cmds:
        print(f"\n=== {cmd} " + "=" * (60 - len(cmd)))
        t0 = time.time()
        globals()[f"run_{cmd}"](cfg, args)
        print(f"({time.time() - t0:.1f} s)")


def _out(args, name):
    return os.path.join(args.out, name)


def _garage(cfg, explode=0.0):
    from garage_sim.geometry import build_garage
    return build_garage(cfg.geometry, explode_gap=explode)


# --------------------------------------------------------------------------- #
def run_model(cfg, args):
    from garage_sim import geometry as geo
    from garage_sim import visualize as viz
    g = _garage(cfg)
    print(g.describe())
    n_tri = geo.export_stl(g, _out(args, "garage.stl"))
    geo.export_obj(g, _out(args, "garage.obj"))
    geo.export_dxf(g, _out(args, "garage_3d.dxf"))
    geo.export_plan_dxf(g, _out(args, "garage_plans.dxf"))
    print(f"CAD exports: garage.stl ({n_tri} triangles), garage.obj/.mtl, garage_3d.dxf, garage_plans.dxf")
    viz.plot_floorplans(g, _out(args, "floorplans.png"))
    viz.render_3d(g, _out(args, "garage_3d_section.png"), title="Cut-away section (near walls and plaza removed, vertical scale x2.5)",
                  elev=16, azim=-62, z_exaggeration=2.5)
    gx = _garage(cfg, explode=getattr(args, "explode", 9.0))
    viz.render_3d(gx, _out(args, "garage_3d_exploded.png"), show=args.show,
                  title="Exploded view: levels pulled apart to show each floor")
    print("Images: floorplans.png, garage_3d_section.png, garage_3d_exploded.png")


def _simulate(cfg, args):
    import numpy as np
    from garage_sim.traffic import simulate_day
    g = _garage(cfg)
    tp = cfg.traffic
    if getattr(args, "ventilation", None):
        tp.ventilation_mode = args.ventilation
    rng = np.random.default_rng(args.seed)
    n_closed = getattr(args, "closed_stalls", 0) or 0
    closed = rng.choice(len(g.stalls), size=n_closed, replace=False) if n_closed else ()
    fans = None
    fans_down = getattr(args, "fans_down", None)
    if fans_down:
        down = (list(fans_down) + [0] * g.n_levels)[:g.n_levels]
        fans = [max(0, g.p.fans_per_level - d) for d in down]
    res = simulate_day(g, tp, seed=args.seed, demand=getattr(args, "demand", None), closed_stalls=closed,
                       fans_working=fans)
    return g, res


def run_traffic(cfg, args):
    from garage_sim import visualize as viz
    from garage_sim.visualize import USER_COLORS
    g, res = _simulate(cfg, args)
    s = res.summary
    print(f"Arrivals {s['arrivals']}, parked {s['parked']}, turned away {s['denied_total']} "
          f"({s['denied_rate']:.1%})  by type {s['denied_by_type']}")
    print(f"Peak occupancy {s['peak_occupancy']}/{s['open_stalls']} ({s['peak_utilization']:.0%}) at "
          f"{int(s['peak_time_h']):02d}:{int(s['peak_time_h'] % 1 * 60):02d}")
    print(f"Entry gate wait: mean {s['mean_gate_wait_min'] * 60:.0f} s, 95th pct {s['p95_gate_wait_min'] * 60:.0f} s, "
          f"max queue {s['max_entry_queue']}")
    print(f"Gate-to-parked: mean {s['mean_search_min']:.1f} min, 95th pct {s['p95_search_min']:.1f} min")
    print(f"Peak CO by level (ppm): {s['peak_co_ppm']}  minutes above {cfg.traffic.co_alarm_ppm:g} ppm alarm: "
          f"{s['minutes_co_above_alarm']}")
    print(f"Fan energy {s['fan_kwh']:.0f} kWh (${s['fan_energy_cost']:.2f})   vehicle passes per level "
          f"{s['level_passes']}")
    viz.plot_traffic(res, g, cfg.traffic, _out(args, "traffic_day.png"), show=args.show)
    viz.plot_driver_experience(res, _out(args, "driver_experience.png"))
    viz.plot_utilization(res, g, _out(args, "stall_utilization.png"))
    peak_min = res.t[res.occupancy.sum(1).argmax()]
    cars = [(sid, USER_COLORS[u]) for sid, u in res.occupied_stalls_at(peak_min)]
    gx = _garage(cfg, explode=9.0)
    viz.render_3d(gx, _out(args, "garage_3d_peak.png"), cars=cars,
                  title=f"Peak occupancy {int(peak_min) // 60:02d}:{int(peak_min) % 60:02d} - {len(cars)} vehicles")
    with open(_out(args, "traffic_cars.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        fields = ["id", "utype", "ev", "ada", "status", "stall", "t_arrive", "t_gate_in", "t_gate_done",
                  "t_parked", "t_leave", "t_exit_queue", "t_exit_done", "search_min"]
        w.writerow(fields)
        for c in res.cars:
            w.writerow([getattr(c, f) if not isinstance(getattr(c, f), float) else round(getattr(c, f), 2)
                        for f in fields])
    print("Wrote traffic_day.png, driver_experience.png, stall_utilization.png, garage_3d_peak.png, traffic_cars.csv")


def run_animate(cfg, args):
    from garage_sim import visualize as viz
    g, res = _simulate(cfg, args)
    fmt = getattr(args, "format", "gif")
    path = _out(args, f"garage_day.{fmt}")
    print("Rendering animation (this takes a minute)...")
    viz.animate_day(res, g, cfg.traffic, path=path, show=args.show, step_min=getattr(args, "step", 3.0),
                    fps=getattr(args, "fps", 15))
    print(f"Wrote {path}")


def _metamodel(cfg, g, args):
    from garage_sim.traffic import build_metamodel
    print("Calibrating capacity metamodel from the traffic simulation...")
    mm = build_metamodel(g, cfg.traffic, seed=args.seed)
    knee = next((r for r, d in zip(mm.rho, mm.denied_frac) if d > 0.005), mm.rho[-1])
    print(f"  cars start being turned away above ~{knee:.1f} daily vehicles per open stall "
          f"(base demand = {cfg.traffic.daily_demand / len(g.stalls):.2f})")
    return mm


def _write_annual(path, res):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res.annual[0]))
        w.writeheader()
        for row in res.annual:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})


def run_maintenance(cfg, args):
    from garage_sim import visualize as viz
    from garage_sim.maintenance import simulate_lifecycle
    g = _garage(cfg)
    mm = _metamodel(cfg, g, args)
    names = list(cfg.policies) if getattr(args, "policy", "preventive") == "all" else [getattr(args, "policy", "preventive")]
    if args.cmd == "all":
        names = list(cfg.policies)
    for name in names:
        if name not in cfg.policies:
            sys.exit(f"Unknown policy '{name}'. Choose from {list(cfg.policies)} or 'all'.")
        res = simulate_lifecycle(g, cfg, cfg.policies[name], mm, seed=args.seed, years=getattr(args, "years", None))
        s = res.summary
        print(f"\n[{name}] {cfg.policies[name].description}")
        print(f"  Lifecycle cost NPV ${s['lcc_npv'] / 1e6:.2f}M  (undiscounted ${s['owner_cost_total'] / 1e6:.2f}M; "
              f"driver cost of turned-away cars ${s['user_cost_total'] / 1e6:.2f}M)")
        print(f"  Availability {s['availability']:.2%}, {s['stall_days_lost']:,.0f} stall-days lost, "
              f"{s['denied_cars']:,.0f} cars turned away")
        print(f"  Floods {s['floods']}, emergency deck repairs {s['emergency_repairs']}, planned {s['planned_repairs']}, "
              f"spall incidents {s['spall_incidents']}, equipment breakdowns {s['equipment_failures']}")
        print(f"  Deck CI: worst ever {s['min_deck_ci']:.0f}, final average {s['final_mean_deck_ci']:.0f}")
        big = sorted(res.events, key=lambda e: -e[3])[:5]
        if big:
            print("  Largest events:")
            for d, cat, desc, cost in big:
                print(f"    {d}  ${cost:>10,.0f}  {desc}")
        viz.plot_lifecycle(res, cfg, _out(args, f"lifecycle_{name}.png"), show=args.show)
        _write_annual(_out(args, f"lifecycle_{name}_annual.csv"), res)
        with open(_out(args, f"lifecycle_{name}_events.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "category", "description", "cost_usd"])
            w.writerows(res.events)
        print(f"  Wrote lifecycle_{name}.png, lifecycle_{name}_annual.csv, lifecycle_{name}_events.csv")


def run_compare(cfg, args):
    from garage_sim import visualize as viz
    from garage_sim.maintenance import compare_policies, summarize_runs
    g = _garage(cfg)
    mm = _metamodel(cfg, g, args)
    names = getattr(args, "policies", None) or list(cfg.policies)
    reps = getattr(args, "reps", 20)

    def progress(name, r, n):
        if sys.stdout.isatty() or r == n:
            print(f"\r  {name:<12} replication {r}/{n}", end="" if r < n else "\n", flush=True)

    results = compare_policies(g, cfg, mm, names, reps=reps, seed=args.seed, years=getattr(args, "years", None),
                               progress=progress)
    rows = [("lcc_npv", 1e-6, "LCC NPV ($M)"), ("cost_structural", 1e-6, "Structural ($M)"),
            ("cost_corrective", 1e-6, "Breakdown repairs ($M)"), ("cost_preventive", 1e-6, "Preventive ($M)"),
            ("cost_incidents", 1e-6, "Incidents/claims ($M)"), ("availability", 100, "Availability (%)"),
            ("denied_cars", 1e-3, "Cars turned away (k)"), ("floods", 1, "Floods"),
            ("emergency_repairs", 1, "Emergency repairs"), ("equipment_failures", 1, "Breakdowns"),
            ("ada_outage_days", 1, "ADA outage days")]
    print(f"\n{'mean ± 95% CI':<24}" + "".join(f"{n:>22}" for n in names))
    table = []
    for key, sc, label in rows:
        cells = []
        for n in names:
            m, h = summarize_runs(results[n], key)
            cells.append(f"{m * sc:,.2f} ± {h * sc:,.2f}")
        print(f"{label:<24}" + "".join(f"{c:>22}" for c in cells))
        table.append([label, *cells])
    best = min(names, key=lambda n: summarize_runs(results[n], "lcc_npv")[0])
    print(f"\nLowest expected lifecycle cost: {best}")
    with open(_out(args, "policy_comparison.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric (mean ± 95% CI)", *names])
        w.writerows(table)
    viz.plot_compare(results, cfg, _out(args, "policy_comparison.png"), show=args.show)
    print("Wrote policy_comparison.png, policy_comparison.csv")


if __name__ == "__main__":
    main()
