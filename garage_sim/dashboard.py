"""
Live dashboard: every result in one window, plus an optional parameter panel.

    python main.py dashboard     3-D model, animated parking map, traffic, CO and policy results in a grid
    python main.py ui            the same grid with sliders to change parameters and re-run

Space pauses the parking-map replay; click the occupancy or CO chart to jump to that time.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider

from . import visualize as viz
from .config import SimConfig
from .geometry import build_garage
from .maintenance import compare_policies, summarize_runs
from .traffic import build_metamodel, simulate_day
from .visualize import INK, INK2, MUTED, USER_COLORS

DASH_RC = {"font.size": 7.5, "axes.titlesize": 9, "legend.fontsize": 7, "xtick.labelsize": 7,
           "ytick.labelsize": 7, "axes.labelsize": 7.5}
T_START, T_END = 6 * 60, 23 * 60


@dataclass
class RunSettings:
    """What-if knobs that are not part of SimConfig (same meaning as the CLI flags)."""
    seed: int = 42
    closed_stalls: int = 0
    fans_down: int = 0          # broken exhaust fans on every level
    reps: int = 5               # Monte-Carlo replications per policy


@dataclass
class Results:
    garage: object
    day: object
    lifecycle: dict
    cfg: SimConfig
    peak_cars: list


def simulate_all(cfg: SimConfig, rs: RunSettings, progress=None) -> Results:
    g = build_garage(cfg.geometry)
    rng = np.random.default_rng(rs.seed)
    n_closed = min(rs.closed_stalls, len(g.stalls) - 1)
    closed = rng.choice(len(g.stalls), size=n_closed, replace=False) if n_closed else ()
    fans = [max(0, g.p.fans_per_level - rs.fans_down)] * g.n_levels if rs.fans_down else None
    day = simulate_day(g, cfg.traffic, seed=rs.seed, closed_stalls=closed, fans_working=fans)
    mm = build_metamodel(g, cfg.traffic, seed=rs.seed)
    life = compare_policies(g, cfg, mm, list(cfg.policies), reps=rs.reps, seed=rs.seed, progress=progress)
    peak_min = day.t[day.occupancy.sum(1).argmax()]
    peak_cars = [(sid, USER_COLORS[u]) for sid, u in day.occupied_stalls_at(peak_min)]
    return Results(g, day, life, cfg, peak_cars)


# --------------------------------------------------------------------------- #
# parameter panel: (label, section, attribute, min, max, step, display scale)
# section "run" refers to RunSettings, the others to SimConfig sections
# --------------------------------------------------------------------------- #
CONTROLS = [
    ("One class day", None),
    ("Daily demand (cars)", ("traffic", "daily_demand", 400, 4000, 50, 1)),
    ("Entry gate lanes", ("traffic", "entry_lanes", 1, 4, 1, 1)),
    ("Stalls closed for repair", ("run", "closed_stalls", 0, 300, 10, 1)),
    ("Broken fans on every level", ("run", "fans_down", 0, 8, 1, 1)),
    ("Garage design", None),
    ("Levels", ("geometry", "n_levels", 1, 5, 1, 1)),
    ("Footprint length (m)", ("geometry", "length_m", 70, 200, 5, 1)),
    ("Aisle modules (rows of stalls)", ("geometry", "n_aisle_modules", 1, 5, 1, 1)),
    ("Exhaust fans per level", ("geometry", "fans_per_level", 1, 8, 1, 1)),
    ("Lifecycle", None),
    ("Years simulated", ("maintenance", "years", 5, 40, 1, 1)),
    ("Enrollment growth (%/yr)", ("maintenance", "demand_growth_per_year", 0, 6, 0.5, 100)),
    ("Sump pumps", ("maintenance", "n_sump_pumps", 1, 6, 1, 1)),
    ("Discount rate (%)", ("maintenance", "discount_rate", 0, 10, 0.5, 100)),
    ("Monte-Carlo runs per policy", ("run", "reps", 2, 30, 1, 1)),
    ("Random seed", ("run", "seed", 1, 200, 1, 1)),
]


class Dashboard:
    def __init__(self, cfg: SimConfig, rs: RunSettings, controls=False, config_path=None):
        self.base_cfg, self.base_rs = cfg, rs
        self.controls = controls
        self.config_path = config_path or "params.json"
        self.fig = plt.figure(figsize=(15, 8.8))
        self.fig.canvas.manager.set_window_title("UTD garage simulator" + (" - parameters" if controls else ""))
        self.left = 0.215 if controls else 0.035
        self.result_axes, self.animated, self.ax3d = [], [], None
        self.playing, self.t = True, float(T_START)
        self.bg = None
        self.sliders, self.radio = {}, None
        with plt.rc_context(DASH_RC):
            self.title = self.fig.text(self.left, 0.985, "", fontsize=12, weight="bold", va="top", color=INK)
            self.kpi = self.fig.text(self.left, 0.952, "", fontsize=8, va="top", color=INK2)
            if controls:
                self._build_controls()
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.timer = self.fig.canvas.new_timer(interval=70)
        self.timer.add_callback(self._tick)
        self.fig.canvas.mpl_connect("close_event", lambda _e: self.timer.stop())
        self._run_now()
        self.timer.start()

    # ------------------------------------------------------------------ results
    def _run_now(self):
        cfg, rs = self._current_settings()
        self._status("Running simulation...")
        try:
            self.res = simulate_all(cfg, rs)
        except Exception as exc:  # bad parameter combination: keep the last results on screen
            if not hasattr(self, "res"):
                raise
            self._status(f"Could not run: {exc}")
            self.fig.canvas.draw_idle()
            return
        with plt.rc_context(DASH_RC):
            self._draw_results()
        self._status("Done. Change the sliders and press Run." if self.controls else "")
        self.fig.canvas.draw_idle()

    def _draw_results(self):
        for ax in self.result_axes:
            ax.remove()
        self.result_axes, self.animated, self.bg = [], [], None
        r, fig = self.res, self.fig
        g, day = r.garage, r.day
        outer = fig.add_gridspec(2, 1, left=self.left, right=0.985, top=0.9, bottom=0.06, height_ratios=[2.1, 1],
                                 hspace=0.2)
        top = outer[0].subgridspec(1, 2, width_ratios=[1, 1.35], wspace=0.04)
        bottom = outer[1].subgridspec(1, 4, wspace=0.3)

        # the 3-D axes is reused: matplotlib never disconnects a removed Axes3D's mouse handlers
        if self.ax3d is None:
            self.ax3d = fig.add_subplot(top[0], projection="3d")
        ax3d = self.ax3d
        ax3d.clear()
        peak_min = day.t[day.occupancy.sum(1).argmax()]
        viz.draw_3d(ax3d, build_garage(r.cfg.geometry, explode_gap=9.0), cars=r.peak_cars, legend_size=6,
                    title=f"3-D model at peak ({int(peak_min) // 60:02d}:{int(peak_min) % 60:02d}, "
                          f"{len(r.peak_cars)} cars) - drag to rotate")
        ax3d.set_axis_off()

        n = g.n_levels
        cols = 1 if n == 1 else 2
        rows = (n + cols - 1) // cols
        plans = top[1].subgridspec(rows + 1, cols, height_ratios=[0.16] + [1] * rows, hspace=0.18, wspace=0.05)
        ax_clock = fig.add_subplot(plans[0, :])
        ax_clock.axis("off")
        self.clock = ax_clock.text(0, 0.5, "", fontsize=11, weight="bold", va="center", color=INK)
        self.result_axes.append(ax_clock)
        plan_axes = [fig.add_subplot(plans[1 + k // cols, k % cols]) for k in range(n)]
        self.result_axes += plan_axes
        self.replay = viz.DayReplay(day, g)
        self.animated += self.replay.setup(plan_axes, parked_size=7, moving_size=16)
        legend_ax = plan_axes[-1]
        if n % cols:  # spare grid cell: use it for the legend
            legend_ax = fig.add_subplot(plans[rows, cols - 1])
            legend_ax.axis("off")
            self.result_axes.append(legend_ax)
            self.replay.legend(legend_ax, loc="center", fontsize=8, ncol=1, title="Parking map")
        else:
            self.replay.legend(ax_clock, loc="center right", fontsize=7)

        ax_occ = fig.add_subplot(bottom[0])
        viz.draw_occupancy(ax_occ, day, g)
        ax_occ.set_ylim(0, max(day.summary["open_stalls"], day.occupancy.sum(1).max()) * 1.15)
        ax_occ.legend(loc="upper right", fontsize=6)
        ax_occ.set_xlabel("hour  (click to jump)")
        ax_co = fig.add_subplot(bottom[1])
        viz.draw_co(ax_co, day, g, r.cfg.traffic, labels_inside=True)
        ax_co.set_title(f"CO by level ({r.cfg.traffic.ventilation_mode} ventilation)")
        ax_co.set_xlabel("hour  (click to jump)")
        self.time_axes = (ax_occ, ax_co)
        self.cursors = [ax.axvline(self.t / 60, color=INK, lw=1) for ax in self.time_axes]
        self.animated += self.cursors + [self.clock]

        ax_cum = fig.add_subplot(bottom[2])
        viz.draw_cumulative_cost(ax_cum, r.lifecycle)
        ax_cum.set_title("Cumulative discounted cost")
        ax_box = fig.add_subplot(bottom[3])
        viz.draw_policy_box(ax_box, r.lifecycle, "lcc_npv", 1e-6, "Lifecycle cost NPV by policy", "$ millions")
        ax_box.set_xticks(range(1, len(r.lifecycle) + 1), list(r.lifecycle))
        self.result_axes += [ax_occ, ax_co, ax_cum, ax_box]

        for a in self.animated:
            a.set_animated(True)
        self._update_frame()
        self._write_header()

    def _write_header(self):
        r = self.res
        s = r.day.summary
        reps = len(next(iter(r.lifecycle.values())))
        years = len(next(iter(r.lifecycle.values()))[0].annual)
        npv = {n: summarize_runs(runs, "lcc_npv")[0] for n, runs in r.lifecycle.items()}
        best = min(npv, key=npv.get)
        self.title.set_text(f"UTD underground garage: {len(r.garage.stalls)} stalls on {r.garage.n_levels} levels")
        self.kpi.set_text(
            f"Class day: {s['arrivals']} arrivals, {s['denied_total']} turned away ({s['denied_rate']:.1%}), "
            f"peak {s['peak_utilization']:.0%} full at {int(s['peak_time_h']):02d}:{int(s['peak_time_h'] % 1 * 60):02d}, "
            f"gate wait {s['mean_gate_wait_min'] * 60:.0f} s, gate-to-stall {s['mean_search_min']:.1f} min, "
            f"peak CO {max(s['peak_co_ppm']):.0f} ppm, fans {s['fan_kwh']:.0f} kWh\n"
            f"{years}-year lifecycle ({reps} runs per policy): "
            + ",  ".join(f"{n} ${v / 1e6:.2f}M" for n, v in npv.items())
            + f"   ->  lowest cost: {best}")

    # ---------------------------------------------------------------- animation
    def _update_frame(self):
        n_parked, n_driving = self.replay.update(self.t)
        hh, mm = divmod(int(self.t), 60)
        self.clock.set_text(f"{hh:02d}:{mm:02d}    parked {n_parked}    driving {n_driving}"
                            + ("" if self.playing else "    (paused - press space)"))
        for c in self.cursors:
            c.set_xdata([self.t / 60, self.t / 60])

    def _on_draw(self, event):
        # a full redraw (resize, 3-D rotation, new results): keep the static background, then paint the cars on top
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        for a in self.animated:
            self.fig.draw_artist(a)

    def _blit(self):
        if self.bg is None:
            return
        cv = self.fig.canvas
        cv.restore_region(self.bg)
        for a in self.animated:
            self.fig.draw_artist(a)
        cv.blit(self.fig.bbox)

    def _tick(self):
        if not self.playing or self.bg is None:
            return
        self.t += 2.0
        if self.t >= T_END:
            self.t = float(T_START)
        self._update_frame()
        self._blit()

    def _on_key(self, event):
        if event.key == " ":
            self.playing = not self.playing
            self._update_frame()
            self._blit()

    def _on_click(self, event):
        if event.inaxes in getattr(self, "time_axes", ()) and event.xdata is not None:
            self.t = float(np.clip(event.xdata * 60, T_START, T_END - 1))
            self._update_frame()
            self._blit()

    # ----------------------------------------------------------------- controls
    def _build_controls(self):
        fig = self.fig
        x0, w = 0.012, 0.135
        fig.patches.append(plt.Rectangle((0, 0), self.left - 0.025, 1, transform=fig.transFigure,
                                         fc="#f2f1ec", ec="none", zorder=-1))
        fig.text(x0, 0.985, "Parameters", fontsize=12, weight="bold", va="top", color=INK)
        y = 0.945
        for label, spec in CONTROLS:
            if spec is None:
                fig.text(x0, y - 0.004, label.upper(), fontsize=7, weight="bold", color=MUTED, va="top")
                y -= 0.026
                continue
            section, attr, lo, hi, step, scale = spec
            ax = fig.add_axes((x0, y - 0.034, w, 0.014))
            sl = Slider(ax, label, lo, hi, valinit=self._get(section, attr) * scale, valstep=step,
                        valfmt="%g", color=viz.SERIES[0], initcolor="none")
            sl.label.set_position((0, 1.25))
            sl.label.set_horizontalalignment("left")
            sl.label.set_verticalalignment("bottom")
            sl.label.set_fontsize(7.5)
            sl.valtext.set_fontsize(7.5)
            self.sliders[(section, attr)] = (sl, scale)
            y -= 0.041
        fig.text(x0, y - 0.004, "Ventilation control", fontsize=7.5, va="top", color=INK)
        rax = fig.add_axes((x0, y - 0.058, w, 0.045))
        rax.set_facecolor("none")
        for sp in rax.spines.values():
            sp.set_visible(False)
        modes = ["demand", "constant"]
        self.radio = RadioButtons(rax, ["CO sensors (demand)", "always full speed"],
                                  active=modes.index(self.base_cfg.traffic.ventilation_mode),
                                  radio_props={"facecolor": viz.SERIES[0]})
        for t in self.radio.labels:
            t.set_fontsize(7.5)
        y -= 0.07
        bw = (w + 0.03 - 0.01) / 3
        self.buttons = []
        for i, (text, cb) in enumerate((("Run", self._on_run), ("Reset", self._on_reset), ("Save", self._on_save))):
            bax = fig.add_axes((x0 + i * (bw + 0.005), y - 0.04, bw, 0.036))
            b = Button(bax, text, color=viz.SERIES[0] if i == 0 else "#e6e5e0",
                       hovercolor="#5b9be3" if i == 0 else "#d4d3cd")
            b.label.set_fontsize(9)
            if i == 0:
                b.label.set_color("white")
                b.label.set_weight("bold")
            b.on_clicked(cb)
            self.buttons.append(b)
        self.status_text = fig.text(x0, y - 0.052, "", fontsize=7.5, va="top", color=INK2, wrap=True)

    def _get(self, section, attr):
        obj = self.base_rs if section == "run" else getattr(self.base_cfg, section)
        return getattr(obj, attr)

    def _current_settings(self):
        cfg, rs = copy.deepcopy(self.base_cfg), copy.deepcopy(self.base_rs)
        for (section, attr), (sl, scale) in self.sliders.items():
            obj = rs if section == "run" else getattr(cfg, section)
            cur = getattr(obj, attr)
            val = sl.val / scale
            setattr(obj, attr, int(round(val)) if isinstance(cur, int) else float(val))
        if self.radio is not None:
            cfg.traffic.ventilation_mode = ["demand", "constant"][self.radio.index_selected]
        rs.fans_down = min(rs.fans_down, cfg.geometry.fans_per_level)
        return cfg, rs

    def _status(self, text):
        if self.controls:
            self.status_text.set_text(text)

    def _on_run(self, _event):
        self._status("Running simulation...")
        self.fig.canvas.draw_idle()
        # let the window repaint the status line before the (blocking) simulation starts
        once = self.fig.canvas.new_timer(interval=60)
        once.single_shot = True
        once.add_callback(self._run_now)
        once.start()
        self._pending = once

    def _on_reset(self, _event):
        for sl, _ in self.sliders.values():
            sl.reset()
        self.radio.set_active(["demand", "constant"].index(self.base_cfg.traffic.ventilation_mode))
        self._status("Sliders reset. Press Run to re-simulate.")
        self.fig.canvas.draw_idle()

    def _on_save(self, _event):
        cfg, rs = self._current_settings()
        cfg.to_json(self.config_path)
        self._status(f"Saved to {self.config_path}\n(use --config {self.config_path})")
        self.fig.canvas.draw_idle()


def run(cfg: SimConfig, rs: RunSettings, controls=False, config_path=None):
    dash = Dashboard(cfg, rs, controls=controls, config_path=config_path)
    plt.show()
    return dash
