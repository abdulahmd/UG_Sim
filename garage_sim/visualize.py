"""Rendering: 3-D CAD views, floor plans, result charts and the day animation."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .geometry import KIND_COLORS, ZONE_COLORS, Garage, car_solid

# categorical slots, fixed order (validated palette: first three are safe all-pairs)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
USER_COLORS = {"student": SERIES[0], "staff": SERIES[1], "visitor": SERIES[2]}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11, "font.size": 9.5,
    "legend.frameon": False, "lines.linewidth": 2.0,
})


def level_color(k: int) -> str:
    return SERIES[k % len(SERIES)]


def _finish(fig, path, show):
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3-D CAD views
# --------------------------------------------------------------------------- #
def render_3d(garage: Garage, path=None, show=False, title=None, elev=26, azim=-58, cars=None,
              hide_roof=True, z_exaggeration=1.0):
    """cars: optional list of (stall_id, colour) to place vehicles in stalls."""
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False  # matplotlib's depth sort fails on big slabs; paint bottom-up instead

    # painter's order: walls, then each level from the deepest up (its slab first, then what sits on it)
    layers: dict[tuple, list] = {}
    for s in garage.solids:
        if hide_roof and s.level == -1 and s.kind in ("roof", "core"):
            continue
        if s.kind == "wall":  # cut away the two walls nearest the viewer
            cx, cy, _ = s.verts.mean(0)
            if not (cy > garage.W or cx < 0):
                continue
            key = (0, 0, 0)
        else:
            key = (1, -s.level, 0 if s.kind in ("slab", "roof") else 1)
        faces = s.faces()
        if s.kind.startswith("stall_"):
            faces = [faces[1]]
        layers.setdefault(key, []).extend((f, KIND_COLORS.get(s.kind, "#cccccc")) for f in faces)
    for sid, col in cars or []:
        st = garage.stalls[sid]
        layers.setdefault((1, -st.level, 2), []).extend((f, col) for f in car_solid(garage, st).faces())
    for z, key in enumerate(sorted(layers)):
        polys, colors = zip(*layers[key])
        ax.add_collection3d(Poly3DCollection(list(polys), facecolors=list(colors), edgecolors=(0, 0, 0, 0.18),
                                             linewidths=0.15, zorder=z))

    pts = np.concatenate([s.verts for s in garage.solids])
    lo, hi = pts.min(0), pts.max(0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi - lo) * np.array([1, 1, z_exaggeration]))
    for k, z in enumerate(garage.floor_z):
        ax.text(garage.L + 4, garage.W / 2, z + 1.0, f"B{k + 1}", fontsize=12, weight="bold", color=INK)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.grid(False)
    ax.set_title(title or "UTD underground parking garage", pad=0)
    handles = [Rectangle((0, 0), 1, 1, fc=ZONE_COLORS[z]) for z in ZONE_COLORS]
    labels = [f"{z} stalls" for z in ZONE_COLORS]
    for kind in ("ramp", "core", "fan", "column"):
        handles.append(Rectangle((0, 0), 1, 1, fc=KIND_COLORS[kind]))
        labels.append({"core": "stair/elevator core", "fan": "exhaust fan"}.get(kind, kind))
    if cars:
        for u, c in USER_COLORS.items():
            handles.append(Rectangle((0, 0), 1, 1, fc=c))
            labels.append(f"{u} car")
    ax.legend(handles, labels, loc="upper left", fontsize=8, ncol=2)
    _finish(fig, path, show)


# --------------------------------------------------------------------------- #
# floor plans
# --------------------------------------------------------------------------- #
def draw_plan(ax, garage: Garage, level: int, values=None, cmap="Blues", vmin=0.0, vmax=1.0, stall_alpha=1.0):
    g = garage
    ax.add_patch(Rectangle((0, -g.S), g.L, g.W + g.S, fc="#f2f1ec", ec="#3d3d3a", lw=1.6, zorder=0))
    for k in (level, level + 1):
        if k >= g.n_levels:
            continue
        xt, xb = g.ramp_span(k)
        x0 = min(xt, xb)
        ax.add_patch(Rectangle((x0, -g.S), abs(xt - xb), g.S, fc=KIND_COLORS["ramp"], ec="#a06a2c",
                               hatch="///", alpha=0.75, lw=0.8, zorder=1))
        if k == level:
            label = "ramp up to street" if level == 0 else f"ramp up to B{level}"
        else:
            label = f"ramp down to B{level + 2}"
        ax.text(x0 + abs(xt - xb) / 2, -g.S / 2, label, ha="center", va="center", fontsize=7, color=INK)

    stalls = [s for s in g.stalls if s.level == level]
    rects = [Rectangle((s.x0, s.y0), s.x1 - s.x0, s.y1 - s.y0) for s in stalls]
    pc = PatchCollection(rects, edgecolor="white", linewidth=0.6, alpha=stall_alpha, zorder=2)
    if values is None:
        pc.set_facecolor([ZONE_COLORS[s.zone] for s in stalls])
    else:
        pc.set_array(np.asarray([values[s.id] for s in stalls]))
        pc.set_cmap(cmap)
        pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    c = g.p.column_size_m
    for x, y in g.columns:
        ax.add_patch(Rectangle((x - c / 2, y - c / 2), c, c, fc="#4a4945", lw=0, zorder=3))
    for x0, x1, y0, y1 in g.cores:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=KIND_COLORS["core"], zorder=3))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, "stair/\nelev", ha="center", va="center", fontsize=6, color="white")
    for lv, x, y in g.fans:
        if lv == level:
            ax.plot(x, y, marker="s", ms=5, color=KIND_COLORS["fan"], zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(-2, g.L + 2)
    ax.set_ylim(-g.S - 2, g.W + 2)
    ax.axis("off")
    return pc


def plot_floorplans(garage: Garage, path=None, show=False):
    n = garage.n_levels
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.9 * n))
    axes = np.atleast_1d(axes)
    counts = garage.stall_counts()
    for k, ax in enumerate(axes):
        draw_plan(ax, garage, k)
        ax.set_title(f"Level B{k + 1}  ({sum(counts[k].values())} stalls, floor at {garage.floor_z[k]:.1f} m)",
                     loc="left")
    handles = [Rectangle((0, 0), 1, 1, fc=ZONE_COLORS[z]) for z in ZONE_COLORS]
    handles += [Rectangle((0, 0), 1, 1, fc=KIND_COLORS["core"]), plt.Line2D([], [], marker="s", ls="",
                                                                            color=KIND_COLORS["fan"])]
    fig.legend(handles, [*(f"{z} stalls" for z in ZONE_COLORS), "stair/elevator core", "exhaust fan"],
               loc="lower center", ncol=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _finish(fig, path, show)


# --------------------------------------------------------------------------- #
# one-day traffic results
# --------------------------------------------------------------------------- #
def plot_traffic(res, garage: Garage, tp, path=None, show=False):
    hours = res.t / 60
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    s = res.summary

    ax = axes[0, 0]
    ax.stackplot(hours, res.occupancy.T, colors=[level_color(k) for k in range(garage.n_levels)],
                 labels=[f"B{k + 1}" for k in range(garage.n_levels)], edgecolor="#fcfcfb", linewidth=1.0)
    ax.axhline(s["open_stalls"], color=INK2, ls="--", lw=1.2)
    ax.text(hours[0] + 0.2, s["open_stalls"] + 8, f"capacity {s['open_stalls']}", color=INK2, fontsize=8)
    ax.set(title="Parked vehicles by level", xlabel="hour of day", ylabel="vehicles", xlim=(hours[0], hours[-1]))
    ax.legend(loc="center right")

    ax = axes[0, 1]
    ax.plot(hours, res.entry_queue, color=SERIES[0], label="entry gate queue")
    ax.plot(hours, res.exit_queue, color=SERIES[1], label="exit gate queue")
    ax.set(title="Gate queues", xlabel="hour of day", ylabel="vehicles waiting", xlim=(hours[0], hours[-1]))
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    for k in range(garage.n_levels):
        ax.plot(hours, res.co_ppm[:, k], color=level_color(k), label=f"B{k + 1}")
    for val, name in ((tp.co_alarm_ppm, "alarm"), (tp.co_limit_ppm, "1-h limit")):
        ax.axhline(val, color=MUTED, ls="--", lw=1)
        ax.text(hours[-1], val, f" {name} {val:g} ppm", va="center", fontsize=8, color=INK2)
    ax.set(title=f"Carbon monoxide by level ({tp.ventilation_mode}-controlled ventilation)",
           xlabel="hour of day", ylabel="CO (ppm)", xlim=(hours[0], hours[-1]),
           ylim=(0, max(tp.co_limit_ppm * 1.15, res.co_ppm.max() * 1.1)))
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    for k in range(garage.n_levels):
        ax.plot(hours, 100 * res.fan_speed[:, k], color=level_color(k), label=f"B{k + 1}")
    ax.set(title=f"Exhaust fan speed  (energy {s['fan_kwh']:.0f} kWh, ${s['fan_energy_cost']:.2f})",
           xlabel="hour of day", ylabel="% of full speed", ylim=(0, 105), xlim=(hours[0], hours[-1]))
    ax.legend(loc="upper left")
    fig.suptitle(f"One class day: {s['arrivals']} arrivals, {s['denied_total']} turned away, "
                 f"peak {s['peak_occupancy']} parked ({s['peak_utilization']:.0%}) at {s['peak_time_h']:.1f} h",
                 fontweight="bold")
    fig.tight_layout()
    _finish(fig, path, show)


def plot_driver_experience(res, path=None, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, attr, title in ((axes[0], "search_min", "Time from gate to parked"),
                            (axes[1], "gate_wait", "Wait at the entry gate")):
        data = []
        for u in USER_COLORS:
            if attr == "gate_wait":
                v = [c.t_gate_in - c.t_arrive for c in res.cars if c.utype == u and c.t_gate_in == c.t_gate_in]
            else:
                v = [c.search_min for c in res.cars if c.utype == u and c.search_min == c.search_min]
            data.append(v)
        bins = np.linspace(0, max(1e-3, max((max(v) for v in data if v), default=1)), 30)
        for (u, col), v in zip(USER_COLORS.items(), data):
            ax.hist(v, bins=bins, color=col, alpha=0.55, label=f"{u} (mean {np.mean(v) if v else 0:.1f} min)",
                    edgecolor="#fcfcfb")
        ax.set(title=title, xlabel="minutes", ylabel="vehicles")
        ax.legend()
    fig.tight_layout()
    _finish(fig, path, show)


def plot_utilization(res, garage: Garage, path=None, show=False):
    minutes = res.t.size
    util = res.stall_occupied_min / minutes
    n = garage.n_levels
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.9 * n))
    axes = np.atleast_1d(axes)
    pc = None
    for k, ax in enumerate(axes):
        pc = draw_plan(ax, garage, k, values=util)
        lv = [util[s.id] for s in garage.stalls if s.level == k]
        ax.set_title(f"B{k + 1}: mean stall utilization {np.mean(lv):.0%}", loc="left")
    fig.colorbar(pc, ax=list(axes), shrink=0.6, label="fraction of the day occupied")
    _finish(fig, path, show)


# --------------------------------------------------------------------------- #
# lifecycle results
# --------------------------------------------------------------------------- #
COST_COLORS = {"structural": SERIES[0], "preventive": SERIES[2], "corrective": SERIES[1], "incidents": SERIES[7],
               "program": SERIES[6], "energy": SERIES[3], "operations": "#b5b4ae", "lost_revenue": SERIES[4]}


def plot_lifecycle(res, cfg, path=None, show=False):
    mp = cfg.maintenance
    pol = cfg.policies[res.policy]
    t = np.array([np.datetime64(d) for d in res.dates])
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    ax = axes[0, 0]
    for k in range(res.slab_ci.shape[1]):
        ax.plot(t, res.slab_ci[:, k].min(1), color=level_color(k), label=f"B{k + 1} worst section")
    ax.plot(t, res.membrane_ci, color=MUTED, lw=1.4, label="plaza membrane")
    ax.axhline(mp.slab_critical_ci, color=SERIES[7], ls="--", lw=1)
    ax.text(t[0], mp.slab_critical_ci + 1.5, "critical: emergency closure", fontsize=8, color=INK2)
    if pol.inspection_interval_days:
        ax.axhline(pol.slab_repair_ci, color=INK2, ls=":", lw=1)
        ax.text(t[0], pol.slab_repair_ci + 1.5, "planned repair trigger", fontsize=8, color=INK2)
    ax.set(title="Deck condition index", ylabel="CI (100 = new)", ylim=(0, 102))
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[0, 1]
    years = [a["year"] for a in res.annual]
    bottom = np.zeros(len(years))
    for cat, col in COST_COLORS.items():
        vals = np.array([a[cat] for a in res.annual]) / 1e3
        ax.bar(years, vals, bottom=bottom, color=col, label=cat.replace("_", " "), edgecolor="#fcfcfb",
               linewidth=1.0, width=0.8)
        bottom += vals
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(title="Annual owner cost", ylabel="$ thousands")
    ax.legend(ncol=2, fontsize=8, loc="upper left")

    ax = axes[1, 0]
    week = 7
    n = len(res.open_stalls) // week * week
    tw = t[:n:week]
    ax.plot(tw, res.open_stalls[:n].reshape(-1, week).min(1), color=SERIES[0], label="open stalls (weekly min)")
    ax.axhline(res.n_stalls, color=MUTED, ls="--", lw=1)
    ax.set(title="Capacity available", ylabel="stalls")
    ax.legend(loc="lower left")

    ax = axes[1, 1]
    month_denied = {}
    for d, v in zip(res.dates, res.denied):
        month_denied[(d.year, d.month)] = month_denied.get((d.year, d.month), 0) + v
    mt = [np.datetime64(f"{y}-{m:02d}-15") for y, m in month_denied]
    ax.bar(mt, list(month_denied.values()), width=25, color=SERIES[1])
    ax.set(title="Cars turned away per month (closures + demand growth)", ylabel="vehicles")

    s = res.summary
    fig.suptitle(f"Policy '{res.policy}': lifecycle cost NPV ${s['lcc_npv'] / 1e6:.2f}M, "
                 f"availability {s['availability']:.2%}, {s['floods']} floods, "
                 f"{s['emergency_repairs']} emergency deck repairs", fontweight="bold")
    fig.tight_layout()
    _finish(fig, path, show)


def plot_compare(results: dict, cfg, path=None, show=False):
    from .maintenance import summarize_runs
    names = list(results)
    cols = {n: SERIES[i % len(SERIES)] for i, n in enumerate(names)}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    def box(ax, key, scale, title, ylabel):
        data = [[r.summary[key] * scale for r in results[n]] for n in names]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops=dict(color=INK, lw=1.5))
        labels = []
        for n in names:
            m, h = summarize_runs(results[n], key)
            labels.append(f"{n}\nmean {m * scale:,.2f} ± {h * scale:,.2f}")
        ax.set_xticks(range(1, len(names) + 1), labels)
        for patch, n in zip(bp["boxes"], names):
            patch.set_facecolor(cols[n])
            patch.set_alpha(0.75)
        ax.set(title=title, ylabel=ylabel)

    box(axes[0, 0], "lcc_npv", 1e-6, "Lifecycle cost (NPV, owner)", "$ millions")
    box(axes[0, 1], "availability", 100, "Stall availability", "% of stall-days open")

    ax = axes[1, 0]
    for n in names:
        curves = np.array([np.cumsum([a["owner_discounted"] for a in r.annual]) for r in results[n]]) / 1e6
        years = [a["year"] for a in results[n][0].annual]
        ax.plot(years, curves.mean(0), color=cols[n], label=n)
        ax.fill_between(years, np.percentile(curves, 10, 0), np.percentile(curves, 90, 0), color=cols[n], alpha=0.15)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(title="Cumulative discounted cost (mean, 10-90% band)", ylabel="$ millions")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    keys = [("floods", "floods"), ("emergency_repairs", "emergency repairs"), ("planned_repairs", "planned repairs"),
            ("spall_incidents", "spall incidents"), ("ada_outage_days", "ADA outage days")]
    x = np.arange(len(keys))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [np.mean([r.summary[k] for r in results[n]]) for k, _ in keys]
        ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, width=w - 0.03, color=cols[n], label=n)
    ax.set_xticks(x, [lbl for _, lbl in keys])
    ax.set(title="Mean events per lifecycle", ylabel="count")
    ax.legend()
    reps = len(next(iter(results.values())))
    fig.suptitle(f"Maintenance policy trade study ({reps} Monte-Carlo replications each, common random numbers)",
                 fontweight="bold")
    fig.tight_layout()
    _finish(fig, path, show)


# --------------------------------------------------------------------------- #
# animation
# --------------------------------------------------------------------------- #
def _along(path_pts, cum, frac):
    d = frac * cum[-1]
    i = min(int(np.searchsorted(cum, d, side="right")) - 1, len(path_pts) - 2)
    i = max(i, 0)
    seg = cum[i + 1] - cum[i]
    u = 0.0 if seg == 0 else (d - cum[i]) / seg
    (x0, y0), (x1, y1) = path_pts[i], path_pts[i + 1]
    return x0 + u * (x1 - x0), y0 + u * (y1 - y0)


def animate_day(res, garage: Garage, tp, path=None, show=False, step_min=2.0, fps=15,
                t_start=6 * 60, t_end=23 * 60):
    n = garage.n_levels
    cars = [c for c in res.cars if c.stall >= 0 and c.t_parked == c.t_parked]
    inf = np.inf
    lvl = np.array([garage.stalls[c.stall].level for c in cars])
    sx = np.array([garage.stalls[c.stall].cx for c in cars])
    sy = np.array([garage.stalls[c.stall].cy for c in cars])
    col = np.array([USER_COLORS[c.utype] for c in cars])
    t_gd = np.array([c.t_gate_done for c in cars])
    t_pk = np.array([c.t_parked for c in cars])
    t_lv = np.array([c.t_leave if c.t_leave == c.t_leave else inf for c in cars])
    t_xa = np.array([c.t_exit_queue if c.t_exit_queue == c.t_exit_queue else inf for c in cars])
    paths = {}
    for c in cars:
        pts = garage.stalls[c.stall].path
        if c.stall not in paths:
            seg = np.hypot(*np.diff(np.array(pts), axis=0).T)
            paths[c.stall] = (pts, np.concatenate([[0], np.cumsum(seg)]))
    sid = np.array([c.stall for c in cars])

    fig = plt.figure(figsize=(15, 3.3 * n + 0.6))
    gs = fig.add_gridspec(n, 2, width_ratios=[2.3, 1])
    plan_axes = [fig.add_subplot(gs[k, 0]) for k in range(n)]
    parked_sc, moving_sc = [], []
    for k, ax in enumerate(plan_axes):
        draw_plan(ax, garage, k, stall_alpha=0.45)
        ax.set_title(f"B{k + 1}", loc="left")
        parked_sc.append(ax.scatter([], [], s=16, marker="s", zorder=5))
        moving_sc.append(ax.scatter([], [], s=34, marker="o", edgecolors=INK, linewidths=0.8, zorder=6))
    handles = [plt.Line2D([], [], marker="s", ls="", color=c) for c in USER_COLORS.values()]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK))
    plan_axes[0].legend(handles, [*USER_COLORS, "driving"], loc="upper right", bbox_to_anchor=(1.0, 1.25), ncol=4)

    ax_occ = fig.add_subplot(gs[: max(1, n // 2 + (n % 2)), 1])
    hours = res.t / 60
    by_type = [((t_pk[None, :] <= res.t[:, None]) & (res.t[:, None] < t_lv[None, :]) & (col == c)[None, :]).sum(1)
               for c in USER_COLORS.values()]
    ax_occ.stackplot(hours, by_type, colors=list(USER_COLORS.values()), labels=list(USER_COLORS),
                     edgecolor="#fcfcfb", linewidth=0.8)
    ax_occ.axhline(res.summary["open_stalls"], color=INK2, ls="--", lw=1)
    ax_occ.set(title="Parked vehicles by driver type", xlim=(t_start / 60, t_end / 60))
    ax_occ.legend(loc="upper right", fontsize=8)
    cur1 = ax_occ.axvline(t_start / 60, color=INK, lw=1)

    ax_co = fig.add_subplot(gs[max(1, n // 2 + (n % 2)):, 1]) if n > 1 else None
    cur2 = None
    if ax_co is not None:
        inks = [INK, INK2, MUTED, "#b5b4ae"]
        i_lab = int(np.searchsorted(res.t, 10 * 60))
        for k in range(n):
            ax_co.plot(hours, res.co_ppm[:, k], color=inks[k % len(inks)], lw=1.6)
            ax_co.text(hours[i_lab], res.co_ppm[i_lab, k], f"B{k + 1} ", ha="right", va="bottom", fontsize=8,
                       color=INK2)
        ax_co.axhline(tp.co_alarm_ppm, color=MUTED, ls="--", lw=1)
        ax_co.set(title="CO by level (ppm)", xlim=(t_start / 60, t_end / 60), xlabel="hour",
                  ylim=(0, max(tp.co_alarm_ppm * 1.1, res.co_ppm.max() * 1.1)))
        cur2 = ax_co.axvline(t_start / 60, color=INK, lw=1)
    clock = fig.text(0.01, 0.985, "", fontsize=14, weight="bold", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    frames = np.arange(t_start, t_end, step_min)

    def update(t):
        parked = (t_pk <= t) & (t < t_lv)
        inbound = (t_gd <= t) & (t < t_pk)
        outbound = (t_lv <= t) & (t < t_xa)
        for k in range(n):
            m = parked & (lvl == k)
            parked_sc[k].set_offsets(np.column_stack([sx[m], sy[m]]) if m.any() else np.empty((0, 2)))
            parked_sc[k].set_facecolors(col[m])
            pos, cols = [], []
            for i in np.flatnonzero((inbound | outbound) & (lvl == k)):
                pts, cum = paths[sid[i]]
                if inbound[i]:
                    frac = (t - t_gd[i]) / max(1e-6, t_pk[i] - t_gd[i])
                else:
                    frac = 1 - (t - t_lv[i]) / max(1e-6, t_xa[i] - t_lv[i])
                pos.append(_along(pts, cum, float(np.clip(frac, 0, 1))))
                cols.append(col[i])
            moving_sc[k].set_offsets(np.array(pos) if pos else np.empty((0, 2)))
            moving_sc[k].set_facecolors(cols if cols else [])
        hh, mm = divmod(int(t), 60)
        clock.set_text(f"{hh:02d}:{mm:02d}   parked {int(parked.sum())}   driving {int((inbound | outbound).sum())}")
        cur1.set_xdata([t / 60, t / 60])
        if cur2 is not None:
            cur2.set_xdata([t / 60, t / 60])
        return []

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    if path:
        writer = FFMpegWriter(fps=fps) if path.endswith(".mp4") else PillowWriter(fps=fps)
        anim.save(path, writer=writer, dpi=80)
    if show:
        plt.show()
    plt.close(fig)
