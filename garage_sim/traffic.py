"""
Discrete-event simulation of one operating day.

Each vehicle: arrives (time-varying Poisson stream shaped around UTD class
times) -> sees the FULL sign or joins the entry-gate queue -> is served by a
gate lane (permit tag read or visitor ticket) -> drives/cruises to the nearest
free stall it is allowed to use -> parks for a lognormal dwell time -> drives
out -> queues at an exit lane -> leaves.

Air quality is a physical mass balance per level, stepped every minute:
    V dC/dt = G(t) - Q(t) (C - C_outdoor)
where G is CO emitted by cars currently driving on that level and Q is the
exhaust airflow delivered by the working fans (CO-controlled VFDs by default).
"""
from __future__ import annotations

import bisect
import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import TrafficParams
from .geometry import ZONES, Garage

USER_TYPES = ("student", "staff", "visitor")
MG_PER_PPM = 28.01 / 24.45      # CO: mg/m3 per ppm at 25 C, 1 atm
CFM_PER_FT2_TO_M3S_PER_M2 = 0.00508

ARRIVE, GATE_DONE, PARKED, LEAVE, EXIT_ARRIVE, EXIT_DONE, TICK = range(7)


@dataclass
class CarRecord:
    id: int
    utype: str
    ev: bool
    ada: bool
    t_arrive: float
    t_gate_in: float = math.nan
    t_gate_done: float = math.nan
    t_parked: float = math.nan
    t_leave: float = math.nan
    t_exit_queue: float = math.nan
    t_exit_start: float = math.nan
    t_exit_done: float = math.nan
    search_min: float = math.nan
    stall: int = -1
    status: str = "arriving"   # denied_full | balked | searching | parked | leaving | exited


@dataclass
class TrafficResult:
    cars: list
    t: np.ndarray                 # tick times, minutes after midnight
    occupancy: np.ndarray         # (T, n_levels) parked cars
    capacity_by_level: np.ndarray
    entry_queue: np.ndarray
    exit_queue: np.ndarray
    co_ppm: np.ndarray            # (T, n_levels)
    fan_speed: np.ndarray         # (T, n_levels) fraction of full speed
    stall_occupied_min: np.ndarray
    level_passes: np.ndarray
    summary: dict = field(default_factory=dict)

    def occupied_stalls_at(self, minute: float) -> list[tuple[int, str]]:
        return [(c.stall, c.utype) for c in self.cars
                if c.t_parked <= minute and not (c.t_leave <= minute)]


def allowed_zones(utype: str, ev: bool, ada: bool) -> list[str]:
    zones = {"student": ["general"], "staff": ["staff", "general"], "visitor": ["visitor", "general"]}[utype]
    if ev:
        zones = ["ev"] + zones
    if ada:
        zones = ["ada"] + zones
    return zones


def _pick_stall(free: dict, zones: list[str]) -> int | None:
    for z in zones:  # ADA / EV drivers take those stalls first when available
        if z in ("ada", "ev") and free[z]:
            return free[z].pop(0)
    best = None
    for z in zones:
        if free[z] and (best is None or free[z][0] < free[best][0]):
            best = z
    return None if best is None else free[best].pop(0)


def simulate_day(garage: Garage, tp: TrafficParams, seed: int | None = None, demand: float | None = None,
                 closed_stalls=(), fans_working=None) -> TrafficResult:
    rng = np.random.default_rng(seed)
    n_lv = garage.n_levels
    stalls = garage.stalls
    N = len(stalls)
    closed = set(closed_stalls)

    free = {z: [] for z in ZONES}
    for s in stalls:
        if s.id not in closed:
            free[s.zone].append(s.id)
    zone_open = {z: len(v) for z, v in free.items()}
    stall_level = np.array([s.level for s in stalls])
    stall_dist = np.array([s.dist for s in stalls])
    share = np.array([s.level_share for s in stalls])
    share /= share.sum(1, keepdims=True)
    open_mask = np.array([s.id not in closed for s in stalls])
    capacity_by_level = np.bincount(stall_level[open_mask], minlength=n_lv)

    # ---- arrivals: piecewise-constant non-homogeneous Poisson process
    demand = tp.daily_demand if demand is None else demand
    hours = np.arange(24)
    w = np.asarray(tp.hourly_profile, float) * ((hours >= int(tp.start_hour)) & (hours < math.ceil(tp.end_hour)))
    w /= w.sum()
    counts = rng.poisson(demand * w)
    t_arr = np.sort(np.concatenate([h * 60 + rng.uniform(0, 60, n) for h, n in zip(hours, counts)]))
    mix = np.array([tp.user_mix[u] for u in USER_TYPES], float)
    utypes = rng.choice(len(USER_TYPES), size=t_arr.size, p=mix / mix.sum())
    is_ev = rng.random(t_arr.size) < tp.ev_fraction
    is_ada = rng.random(t_arr.size) < tp.ada_fraction
    cars = [CarRecord(i, USER_TYPES[u], bool(e), bool(a), float(t))
            for i, (t, u, e, a) in enumerate(zip(t_arr, utypes, is_ev, is_ada))]
    zones_of = [allowed_zones(c.utype, c.ev, c.ada) for c in cars]

    def service(mean_s):  # gamma, CV = 0.5
        return rng.gamma(4.0, mean_s / 4.0) / 60.0

    def dwell(utype):
        m, cv = tp.dwell_mean_h[utype] * 60, tp.dwell_cv[utype]
        s2 = math.log(1 + cv * cv)
        return max(10.0, rng.lognormal(math.log(m) - s2 / 2, math.sqrt(s2)))

    heap = []
    seq = itertools.count()

    def push(t, kind, i):
        heapq.heappush(heap, (t, next(seq), kind, i))

    for c in cars:
        push(c.t_arrive, ARRIVE, c.id)
    t0, t1 = tp.start_hour * 60, tp.end_hour * 60
    ticks = np.arange(t0, t1, 1.0)
    for k, tk in enumerate(ticks):
        push(tk, TICK, k)

    # ---- ventilation physics set-up
    V = garage.level_volume_m3
    q_design = tp.vent_cfm_per_ft2 * CFM_PER_FT2_TO_M3S_PER_M2 * garage.level_area_m2   # m3/s per level
    fw = np.ones(n_lv) if fans_working is None else np.asarray(fans_working, float) / garage.p.fans_per_level
    c_bg = tp.co_background_ppm * MG_PER_PPM
    C = np.full(n_lv, c_bg)
    emit = np.zeros(n_lv)                      # g/min of CO being released on each level
    g = tp.co_emission_g_per_min
    kwh = 0.0

    T = ticks.size
    occ_ts = np.zeros((T, n_lv), int)
    q_in = np.zeros(T, int)
    q_out = np.zeros(T, int)
    co_ts = np.zeros((T, n_lv))
    fan_ts = np.zeros((T, n_lv))
    stall_occ = np.zeros(N)
    occupied = np.zeros(N, bool)
    level_count = np.zeros(n_lv, int)
    passes = np.zeros(n_lv)
    entry_q, exit_q = deque(), deque()
    entry_busy = exit_busy = 0
    speed = tp.drive_speed_kmh * 1000 / 60    # m/min

    while heap:
        t, _, kind, i = heapq.heappop(heap)
        if t >= t1:
            break
        if kind == TICK:
            occ_ts[i] = level_count
            q_in[i], q_out[i] = len(entry_q), len(exit_q)
            ppm = C / MG_PER_PPM
            if tp.ventilation_mode == "demand":
                spd = np.clip(tp.dcv_min_speed + (ppm - tp.dcv_low_ppm) / (tp.dcv_high_ppm - tp.dcv_low_ppm)
                              * (1 - tp.dcv_min_speed), tp.dcv_min_speed, 1.0)
            else:
                spd = np.ones(n_lv)
            Q = q_design * fw * spd * 60.0                          # m3/min
            C = C + np.maximum(emit, 0) * 1000.0 / V - (Q / V) * (C - c_bg)
            co_ts[i], fan_ts[i] = C / MG_PER_PPM, spd
            kwh += float((tp.fan_kw_per_m3s * q_design * fw * spd ** 3).sum()) / 60.0
            stall_occ += occupied
            continue

        c = cars[i]
        if kind == ARRIVE:
            if not any(free[z] for z in zones_of[i]):
                c.status = "denied_full"
            elif entry_busy < tp.entry_lanes:
                entry_busy += 1
                c.t_gate_in = t
                push(t + service(tp.entry_service_s[c.utype]), GATE_DONE, i)
            else:
                entry_q.append(i)

        elif kind == GATE_DONE:
            c.t_gate_done = t
            entry_busy -= 1
            if entry_q:
                j = entry_q.popleft()
                entry_busy += 1
                cars[j].t_gate_in = t
                push(t + service(tp.entry_service_s[cars[j].utype]), GATE_DONE, j)
            zones = zones_of[i]
            sid = _pick_stall(free, zones)
            if sid is None:
                c.status = "balked"
                continue
            occ = 1 - sum(len(free[z]) for z in zones) / max(1, sum(zone_open[z] for z in zones))
            c.stall, c.status = sid, "searching"
            c.search_min = stall_dist[sid] / speed + tp.cruise_penalty_min * occ ** tp.cruise_exponent * rng.uniform(0.5, 1.5)
            emit += g * share[sid]
            push(t + c.search_min, PARKED, i)

        elif kind == PARKED:
            sid = c.stall
            c.t_parked, c.status = t, "parked"
            emit -= g * share[sid]
            occupied[sid] = True
            level_count[stall_level[sid]] += 1
            passes[:stall_level[sid] + 1] += 1
            push(t + dwell(c.utype), LEAVE, i)

        elif kind == LEAVE:
            sid = c.stall
            c.t_leave, c.status = t, "leaving"
            occupied[sid] = False
            level_count[stall_level[sid]] -= 1
            bisect.insort(free[stalls[sid].zone], sid)
            emit += g * share[sid]
            push(t + stall_dist[sid] / speed, EXIT_ARRIVE, i)

        elif kind == EXIT_ARRIVE:
            sid = c.stall
            emit -= g * share[sid]
            passes[:stall_level[sid] + 1] += 1
            c.t_exit_queue = t
            if exit_busy < tp.exit_lanes:
                exit_busy += 1
                c.t_exit_start = t
                push(t + service(tp.exit_service_s[c.utype]), EXIT_DONE, i)
            else:
                exit_q.append(i)

        elif kind == EXIT_DONE:
            c.t_exit_done, c.status = t, "exited"
            exit_busy -= 1
            if exit_q:
                j = exit_q.popleft()
                exit_busy += 1
                cars[j].t_exit_start = t
                push(t + service(tp.exit_service_s[cars[j].utype]), EXIT_DONE, j)

    res = TrafficResult(cars, ticks, occ_ts, capacity_by_level, q_in, q_out, co_ts, fan_ts, stall_occ, passes)
    res.summary = _summarise(res, tp, kwh, n_open=int(open_mask.sum()))
    return res


def _summarise(r: TrafficResult, tp: TrafficParams, kwh: float, n_open: int) -> dict:
    cars = r.cars

    def col(name):
        return np.array([getattr(c, name) for c in cars], float)

    status = np.array([c.status for c in cars])
    t_arr, t_in, t_park = col("t_arrive"), col("t_gate_in"), col("t_parked")
    gate_wait = (t_in - t_arr)[~np.isnan(t_in)]
    exit_wait = (col("t_exit_start") - col("t_exit_queue"))
    exit_wait = exit_wait[~np.isnan(exit_wait)]
    search = col("search_min")
    search = search[~np.isnan(search)]
    total_occ = r.occupancy.sum(1)
    peak_i = int(total_occ.argmax()) if total_occ.size else 0
    denied = (status == "denied_full") | (status == "balked")
    utypes = np.array([c.utype for c in cars])
    above = r.co_ppm > tp.co_alarm_ppm

    def pct(a, q):
        return float(np.percentile(a, q)) if a.size else 0.0

    return {
        "arrivals": len(cars),
        "parked": int((~np.isnan(t_park)).sum()),
        "denied_full": int((status == "denied_full").sum()),
        "balked_after_gate": int((status == "balked").sum()),
        "denied_total": int(denied.sum()),
        "denied_rate": float(denied.mean()) if cars else 0.0,
        "denied_by_type": {u: int((denied & (utypes == u)).sum()) for u in USER_TYPES},
        "still_parked_at_close": int(np.isin(status, ["parked", "searching"]).sum()),
        "open_stalls": n_open,
        "peak_occupancy": int(total_occ.max()) if total_occ.size else 0,
        "peak_utilization": float(total_occ.max() / max(1, n_open)) if total_occ.size else 0.0,
        "peak_time_h": float(r.t[peak_i] / 60) if total_occ.size else 0.0,
        "mean_gate_wait_min": float(gate_wait.mean()) if gate_wait.size else 0.0,
        "p95_gate_wait_min": pct(gate_wait, 95),
        "max_entry_queue": int(r.entry_queue.max()) if r.entry_queue.size else 0,
        "mean_search_min": float(search.mean()) if search.size else 0.0,
        "p95_search_min": pct(search, 95),
        "mean_exit_wait_min": float(exit_wait.mean()) if exit_wait.size else 0.0,
        "peak_co_ppm": [round(float(x), 1) for x in r.co_ppm.max(0)],
        "minutes_co_above_alarm": [int(x) for x in above.sum(0)],
        "fan_kwh": float(kwh),
        "fan_energy_cost": float(kwh * tp.electricity_usd_per_kwh),
        "level_passes": [int(x) for x in r.level_passes],
    }


@dataclass
class CapacityMetamodel:
    """Response surface fitted from DES runs, so the 20-year model can ask
    'how many cars get turned away at this demand and this many open stalls?'
    without running thousands of full simulations."""
    rho: np.ndarray            # demand / open stalls
    denied_frac: np.ndarray
    search_min: np.ndarray
    fan_kwh: np.ndarray
    pass_per_entry: np.ndarray  # vehicle passes over each level per car that enters
    n_stalls: int

    def denied_fraction(self, demand: float, open_stalls: int) -> float:
        if open_stalls <= 0:
            return 1.0
        return float(np.interp(demand / open_stalls, self.rho, self.denied_frac))

    def fan_kwh_for(self, demand: float) -> float:
        return float(np.interp(demand / self.n_stalls, self.rho, self.fan_kwh))


def build_metamodel(garage: Garage, tp: TrafficParams, seed: int = 0, reps: int = 2,
                    multipliers=np.linspace(0.1, 3.0, 13)) -> CapacityMetamodel:
    N = len(garage.stalls)
    rows = []
    for m in multipliers:
        vals = []
        for r in range(reps):
            s = simulate_day(garage, tp, seed=seed + 1000 * r + int(m * 100), demand=tp.daily_demand * m).summary
            vals.append((s["arrivals"] / N, s["denied_rate"], s["mean_search_min"], s["fan_kwh"]))
        rows.append(np.mean(vals, axis=0))
    rows = np.array(rows)
    rows = rows[np.argsort(rows[:, 0])]
    rows[:, 1] = np.maximum.accumulate(rows[:, 1])  # denial can only grow with load
    base = simulate_day(garage, tp, seed=seed).summary
    pass_per_entry = np.asarray(base["level_passes"], float) / max(1, base["parked"])
    return CapacityMetamodel(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], pass_per_entry, N)
