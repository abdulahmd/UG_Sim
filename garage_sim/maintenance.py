"""
Multi-year maintenance and lifecycle-cost simulation (daily time step).

What wears out
  * Concrete decks: condition index (CI) per level section, lost to moisture,
    groundwater (worse deeper), plaza leaks (B1), and traffic passes. A sealer
    slows it down. Low CI -> spalling incidents; CI below critical -> emergency
    closure and expensive repair.
  * Plaza waterproofing membrane over B1: leaks feed the sump and the B1 deck.
  * Equipment with Weibull lifetimes: exhaust fans, sump pumps, gate arms,
    elevators, LED luminaires. Fans down -> level closed for air quality;
    pumps down during a storm -> lowest level floods.

What a maintenance policy controls
  inspection interval, repair trigger CI, sealer cycle, preventive-maintenance
  (PM) intervals, planned replacement ages, condition-monitoring sensors and
  spares stocking. See PolicyParams in config.py.

Capacity lost to closures feeds back into service: a metamodel fitted from the
one-day discrete-event simulation converts (demand, open stalls) into cars
turned away.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from .config import PolicyParams, SimConfig
from .geometry import Garage
from .traffic import CapacityMetamodel

COST_CATS = ("program", "preventive", "corrective", "structural", "energy", "operations", "incidents",
             "lost_revenue")
CAT = {c: i for i, c in enumerate(COST_CATS)}
UNIT_GROUPS = ("vent_fan", "sump_pump", "gate", "elevator", "light")


class _UnitGroup:
    """A population of identical repairable units with Weibull time-to-failure."""

    def __init__(self, name: str, spec: dict, count: int):
        self.name, self.spec, self.n = name, spec, count
        self.age = np.zeros(count)              # effective age, years
        self.down_until = np.full(count, -1)    # day index the unit is back in service
        self.failures = 0
        self.caught = 0

    def up(self, d: int) -> np.ndarray:
        return self.down_until <= d

    def step(self, d: int, policy: PolicyParams, rng, planned_cost_factor: float):
        sp = self.spec
        dt = 1 / 365
        up = self.up(d)
        b, eta = sp["beta"], sp["eta_years"]
        p_fail = -np.expm1((self.age / eta) ** b - ((self.age + dt) / eta) ** b)
        fail = up & (rng.random(self.n) < p_fail)
        self.age[up] += dt
        planned = corrective = 0.0
        n_broke = 0
        idx = np.flatnonzero(fail)
        if idx.size:
            p_det = policy.detect_prob if self.name in policy.monitored_groups else 0.0
            caught = rng.random(idx.size) < p_det
            planned += caught.sum() * sp["repair_cost"] * planned_cost_factor
            broke = idx[~caught]
            mttr = max(1e-3, sp["mttr_days"] * policy.mttr_factor)
            self.down_until[broke] = d + 1 + rng.exponential(mttr, broke.size).astype(int)
            corrective += broke.size * sp["repair_cost"] * policy.corrective_premium
            self.age[idx] = 0.0
            self.failures += broke.size
            n_broke = broke.size
            self.caught += int(caught.sum())
        interval = policy.pm_interval_days.get(self.name)
        if interval and d > 0 and d % int(interval) == 0:
            up = self.up(d)
            planned += up.sum() * sp["pm_cost"]
            self.age[up] *= 1 - sp["pm_restoration"]
        age_limit = policy.replace_age_years.get(self.name)
        if age_limit and d % 30 == 0:
            old = self.age >= age_limit
            planned += old.sum() * sp["replace_cost"]
            self.age[old] = 0.0
            self.down_until[old] = np.minimum(self.down_until[old], d)
        return planned, corrective, n_broke


@dataclass
class LifecycleResult:
    policy: str
    seed: int
    dates: list
    demand: np.ndarray
    open_stalls: np.ndarray
    denied: np.ndarray
    slab_ci: np.ndarray         # (days, levels, sections)
    membrane_ci: np.ndarray
    light_avail: np.ndarray
    fans_up: np.ndarray         # (days, levels)
    pumps_up: np.ndarray
    annual: list
    events: list
    summary: dict = field(default_factory=dict)
    n_stalls: int = 0


def _calendar(mp, start: date, n_days: int):
    periods = [(tuple(map(int, a.split("-"))), tuple(map(int, b.split("-"))), f) for a, b, f in mp.academic_calendar]
    dates, fac = [], np.empty(n_days)
    for d in range(n_days):
        day = start + timedelta(days=d)
        md = (day.month, day.day)
        f = next((x for a, b, x in periods if a <= md <= b), 1.0)
        fac[d] = f * mp.weekday_factor[day.weekday()]
        dates.append(day)
    return dates, fac


def simulate_lifecycle(garage: Garage, cfg: SimConfig, policy: PolicyParams, metamodel: CapacityMetamodel,
                       seed: int = 0, years: int | None = None) -> LifecycleResult:
    mp, tp, gp = cfg.maintenance, cfg.traffic, garage.p
    rng = np.random.default_rng(seed)
    years = years or mp.years
    n_days = int(years * 365)
    n_years = math.ceil(n_days / 365)
    start = date.fromisoformat(mp.start_date)
    dates, cal = _calendar(mp, start, n_days)
    n_lv, n_sec, N = garage.n_levels, gp.slab_sections_per_level, len(garage.stalls)

    sec_stalls = np.zeros((n_lv, n_sec), int)
    for s in garage.stalls:
        sec_stalls[s.level, s.section] += 1
    sec_area = garage.level_area_m2 / n_sec
    visitor_share = tp.user_mix["visitor"] / sum(tp.user_mix.values())

    counts = {"vent_fan": n_lv * gp.fans_per_level, "sump_pump": mp.n_sump_pumps,
              "gate": tp.entry_lanes + tp.exit_lanes, "elevator": mp.n_elevators, "light": garage.n_lights}
    units = {g: _UnitGroup(g, mp.units[g], counts[g]) for g in UNIT_GROUPS}

    # deck state
    ci = mp.slab_initial_ci - rng.uniform(0, 2, (n_lv, n_sec))
    rate_mult = rng.lognormal(0, mp.slab_rate_sigma, (n_lv, n_sec))
    seal_age = np.zeros((n_lv, n_sec))
    depth = 1 + mp.slab_groundwater_factor * np.arange(n_lv) / max(1, n_lv - 1)
    rep_start = np.full((n_lv, n_sec), -1)
    rep_end = np.full((n_lv, n_sec), -1)
    rep_cost = np.zeros((n_lv, n_sec))
    rep_emerg = np.zeros((n_lv, n_sec), bool)
    closed_from = np.full((n_lv, n_sec), -1)
    closed_until = np.full((n_lv, n_sec), -1)
    membrane, mem_job, mem_emerg = 100.0, -1, False
    level_closed_until = np.full(n_lv, -1)
    vent_closed_prev = np.zeros(n_lv, bool)

    annual_cost = np.zeros((n_years, len(COST_CATS)))
    counters = {k: np.zeros(n_years, int) for k in
                ("floods", "emergency_repairs", "planned_repairs", "spall_incidents", "dark_incidents",
                 "equipment_failures", "vent_closure_days", "ada_outage_days", "sealer_jobs")}
    user_cost = np.zeros(n_years)
    events = []

    rec_open = np.zeros(n_days, int)
    rec_dem = np.zeros(n_days)
    rec_denied = np.zeros(n_days)
    rec_ci = np.zeros((n_days, n_lv, n_sec), np.float32)
    rec_mem = np.zeros(n_days)
    rec_light = np.zeros(n_days)
    rec_fans = np.zeros((n_days, n_lv), int)
    rec_pumps = np.zeros(n_days, int)

    fixed_per_day = (mp.cleaning_per_year + mp.fire_life_safety_per_year + mp.elevator_inspection_per_year) / 365
    served_prev = tp.daily_demand * cal[0]

    def book(d, cat, amount, desc=None):
        annual_cost[d // 365, CAT[cat]] += amount
        if desc:
            events.append((dates[d].isoformat(), cat, desc, round(amount)))

    def schedule_repair(d, l, s, start_day, emergency):
        sev = max(0.0, mp.slab_restored_ci - ci[l, s])
        f_days = mp.emergency_days_factor if emergency else 1.0
        f_cost = mp.emergency_cost_factor if emergency else 1.0
        dur = math.ceil((mp.repair_days_base + mp.repair_days_per_point * sev) * f_days)
        rep_start[l, s], rep_end[l, s] = start_day, start_day + dur
        rep_cost[l, s] = (sec_area * min(1.0, sev / 100 * mp.repair_area_factor) * mp.slab_repair_cost_m2 * f_cost
                          + mp.repair_mobilization)
        rep_emerg[l, s] = emergency
        closed_from[l, s] = d if emergency else start_day
        closed_until[l, s] = start_day + dur

    for d in range(n_days):
        y = d // 365
        dem = tp.daily_demand * cal[d] * (1 + mp.demand_growth_per_year) ** y

        # ---------------- equipment reliability
        for name, grp in units.items():
            planned, corrective, n_broke = grp.step(d, policy, rng, mp.planned_cost_factor)
            if planned:
                book(d, "preventive", planned)
            if corrective:
                book(d, "corrective", corrective)
                counters["equipment_failures"][y] += n_broke
        fans_up = units["vent_fan"].up(d).reshape(n_lv, gp.fans_per_level).sum(1)
        pumps_up = int(units["sump_pump"].up(d).sum())
        light_av = float(units["light"].up(d).mean())
        gates_up = units["gate"].up(d)
        if not gates_up[:tp.entry_lanes].any():
            book(d, "operations", mp.manual_gate_cost_per_day)
        if not units["elevator"].up(d).any():
            counters["ada_outage_days"][y] += 1

        vent_closed = fans_up < mp.min_fan_fraction * gp.fans_per_level
        for l in np.flatnonzero(vent_closed & ~vent_closed_prev):
            events.append((dates[d].isoformat(), "closure", f"B{l + 1} closed: only {fans_up[l]} exhaust fans running", 0))
        counters["vent_closure_days"][y] += int(vent_closed.sum())
        vent_closed_prev = vent_closed

        # ---------------- storms, pumps, flooding
        leak = 1 - membrane / 100
        flood = False
        if rng.random() < mp.heavy_rain_prob_by_month[dates[d].month - 1]:
            intensity = rng.exponential(mp.storm_intensity_mean)
            need = max(1, math.ceil(intensity * (mp.pumps_per_unit_intensity + mp.leak_inflow * leak)))
            flood = pumps_up < need
        elif pumps_up == 0:
            flood = rng.random() < 0.5  # groundwater seepage with no pumps at all
        if flood and level_closed_until[-1] <= d:
            lo, hi = mp.flood_closure_days
            level_closed_until[-1] = d + int(rng.integers(lo, hi + 1))
            claims = (sec_stalls[-1].sum() * mp.flood_occupied_fraction * mp.flood_damage_fraction
                      * mp.flood_claim_per_vehicle)
            book(d, "incidents", mp.flood_cleanup_cost + claims,
                 f"B{n_lv} flooded ({pumps_up} of {mp.n_sump_pumps} sump pumps working)")
            ci[-1] -= mp.flood_slab_damage
            counters["floods"][y] += 1

        # ---------------- plaza membrane
        membrane = max(0.0, membrane - mp.membrane_loss_per_year / 365)
        if mem_job < 0 and membrane < mp.membrane_failed_ci:
            mem_job, mem_emerg = d + mp.emergency_lead_days, True
        if d == mem_job:
            cost = garage.L * (garage.W + garage.S) * mp.membrane_cost_m2 * (mp.emergency_cost_factor if mem_emerg else 1)
            book(d, "structural", cost, "Plaza waterproofing membrane replaced" + (" (emergency)" if mem_emerg else ""))
            membrane, mem_job = 100.0, -1

        # ---------------- deck deterioration
        passes = served_prev * metamodel.pass_per_entry
        wear = mp.slab_wear_per_1000_passes * passes[:, None] / n_sec / 1000
        base = mp.slab_base_loss_per_year / 365 * depth[:, None]
        leak_acc = np.ones((n_lv, 1))
        leak_acc[0] += mp.leak_slab_accel * leak
        accel = 1 + mp.slab_nonlinear * (100 - ci) / 100
        protect = np.where(seal_age < mp.sealant_life_years, 1 - mp.sealant_protection, 1.0)
        working_on = (rep_start >= 0) & (rep_start <= d)
        ci -= rate_mult * (base + wear) * leak_acc * accel * protect * ~working_on
        np.clip(ci, 0, 100, out=ci)
        seal_age += 1 / 365

        # ---------------- inspections (policy) and user-reported hazards (everyone)
        if policy.inspection_interval_days and d > 0 and d % policy.inspection_interval_days == 0:
            book(d, "program", mp.inspection_cost)
            observed = ci + rng.normal(0, mp.inspection_sd, ci.shape)
            for l, s in zip(*np.nonzero((observed < policy.slab_repair_ci) & (rep_start < 0))):
                schedule_repair(d, l, s, d + mp.repair_lead_days, emergency=False)
            if (policy.membrane_replace_ci is not None and mem_job < 0
                    and membrane + rng.normal(0, mp.inspection_sd) < policy.membrane_replace_ci):
                mem_job, mem_emerg = d + mp.repair_lead_days, False
        for l, s in zip(*np.nonzero((ci < mp.slab_critical_ci) & (rep_start < 0))):
            schedule_repair(d, l, s, d + mp.emergency_lead_days, emergency=True)
            events.append((dates[d].isoformat(), "closure",
                           f"B{l + 1} section {s + 1} closed: spalling hazard (CI {ci[l, s]:.0f})", 0))

        # ---------------- repair jobs
        for l, s in zip(*np.nonzero(rep_start == d)):
            kind = "Emergency" if rep_emerg[l, s] else "Planned"
            book(d, "structural", rep_cost[l, s],
                 f"{kind} deck repair B{l + 1} section {s + 1} (CI {ci[l, s]:.0f}, {rep_end[l, s] - d} days)")
            counters["emergency_repairs" if rep_emerg[l, s] else "planned_repairs"][y] += 1
        for l, s in zip(*np.nonzero(rep_end == d)):
            ci[l, s] = mp.slab_restored_ci
            rate_mult[l, s] = rng.lognormal(0, mp.slab_rate_sigma)
            seal_age[l, s] = 0.0
            rep_start[l, s] = rep_end[l, s] = -1

        # ---------------- sealer (scheduled for low-demand days)
        if policy.sealant_interval_years and cal[d] <= 0.5:
            due = (seal_age >= policy.sealant_interval_years) & (rep_start < 0)
            for l, s in zip(*np.nonzero(due)):
                book(d, "preventive", sec_area * mp.sealant_cost_m2)
                seal_age[l, s] = 0.0
                closed_from[l, s] = d
                closed_until[l, s] = max(closed_until[l, s], d + mp.sealant_closure_days)
                counters["sealer_jobs"][y] += 1

        # ---------------- capacity and service
        sec_closed = (closed_from >= 0) & (closed_from <= d) & (d < closed_until)
        lvl_closed = (level_closed_until > d) | vent_closed
        closed_mask = sec_closed | lvl_closed[:, None]
        open_st = int(N - (sec_stalls * closed_mask).sum())
        frac = metamodel.denied_fraction(dem, open_st)
        denied = dem * frac
        served_prev = dem - denied
        book(d, "lost_revenue", denied * visitor_share * mp.visitor_fee)
        user_cost[y] += denied * mp.denied_user_cost

        # ---------------- safety incidents
        p_spall = mp.spall_incident_base_per_day * np.exp(-(ci - mp.slab_critical_ci) / 8.0)
        n_sp = int(((rng.random(ci.shape) < p_spall) & ~closed_mask).sum())
        if n_sp:
            book(d, "incidents", n_sp * mp.spall_claim_cost, "Falling-concrete (spall) damage claim")
            counters["spall_incidents"][y] += n_sp
        p_dark = mp.dark_incident_base_per_day * (1 - light_av) / 0.05
        if rng.random() < p_dark:
            book(d, "incidents", mp.dark_claim_cost, f"Incident in poorly lit area ({light_av:.0%} lights working)")
            counters["dark_incidents"][y] += 1

        # ---------------- energy and fixed costs
        kwh = metamodel.fan_kwh_for(dem) + garage.n_lights * light_av * mp.light_watts * 24 / 1000
        book(d, "energy", kwh * tp.electricity_usd_per_kwh)
        book(d, "operations", fixed_per_day)
        book(d, "program", policy.annual_program_cost / 365)

        rec_open[d], rec_dem[d], rec_denied[d] = open_st, dem, denied
        rec_ci[d], rec_mem[d], rec_light[d] = ci, membrane, light_av
        rec_fans[d], rec_pumps[d] = fans_up, pumps_up

    # ---------------- roll-up
    r = mp.discount_rate
    annual = []
    for y in range(n_years):
        sl = slice(y * 365, min((y + 1) * 365, n_days))
        owner = float(annual_cost[y].sum())
        row = {"year": start.year + y, **{c: float(annual_cost[y, CAT[c]]) for c in COST_CATS},
               "owner_total": owner, "owner_discounted": owner / (1 + r) ** (y + 1),
               "user_cost": float(user_cost[y]),
               "availability": float(rec_open[sl].mean() / N),
               "denied_cars": float(rec_denied[sl].sum()),
               "min_deck_ci": float(rec_ci[sl].min()), "mean_deck_ci": float(rec_ci[sl].mean()),
               "lighting_avail": float(rec_light[sl].mean())}
        row.update({k: int(v[y]) for k, v in counters.items()})
        annual.append(row)

    res = LifecycleResult(policy.name, seed, dates, rec_dem, rec_open, rec_denied, rec_ci, rec_mem, rec_light,
                          rec_fans, rec_pumps, annual, events, n_stalls=N)
    res.summary = {
        "policy": policy.name,
        "lcc_npv": sum(a["owner_discounted"] for a in annual),
        "owner_cost_total": float(annual_cost.sum()),
        **{f"cost_{c}": float(annual_cost[:, CAT[c]].sum()) for c in COST_CATS},
        "user_cost_total": float(user_cost.sum()),
        "availability": float(rec_open.mean() / N),
        "stall_days_lost": float((N - rec_open).sum()),
        "denied_cars": float(rec_denied.sum()),
        **{k: int(v.sum()) for k, v in counters.items()},
        **{f"failures_{g}": units[g].failures for g in UNIT_GROUPS},
        **{f"predicted_{g}": units[g].caught for g in UNIT_GROUPS},
        "min_deck_ci": float(rec_ci.min()),
        "final_mean_deck_ci": float(rec_ci[-1].mean()),
        "mean_lighting_avail": float(rec_light.mean()),
    }
    return res


def compare_policies(garage: Garage, cfg: SimConfig, metamodel: CapacityMetamodel, policies: list[str],
                     reps: int = 20, seed: int = 0, years: int | None = None, progress=None) -> dict:
    """Monte-Carlo trade study. The same seeds are used for every policy (common
    random numbers), so differences between policies are not just luck."""
    out = {}
    for name in policies:
        runs = []
        for r in range(reps):
            runs.append(simulate_lifecycle(garage, cfg, cfg.policies[name], metamodel, seed=seed + r, years=years))
            if progress:
                progress(name, r + 1, reps)
        out[name] = runs
    return out


def summarize_runs(runs: list[LifecycleResult], key: str) -> tuple[float, float]:
    """Mean and 95% confidence half-width across replications."""
    x = np.array([r.summary[key] for r in runs], float)
    if x.size < 2:
        return float(x.mean()), 0.0
    t = 1.96 + 2.4 / (x.size - 1)  # close approximation of the Student-t critical value
    return float(x.mean()), float(t * x.std(ddof=1) / math.sqrt(x.size))
