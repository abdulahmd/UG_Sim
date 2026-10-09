"""
Every tunable parameter for the simulation lives here.

Edit the defaults directly, or run `python main.py config` to dump them to
params.json, edit that file, and pass it back with `--config params.json`.
Units: metres, minutes (traffic), days/years (maintenance), US dollars.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields


# --------------------------------------------------------------------------- #
# Physical layout
# --------------------------------------------------------------------------- #
@dataclass
class GeometryParams:
    n_levels: int = 3                   # B1, B2, B3 ...
    length_m: float = 120.0             # west-east footprint
    n_aisle_modules: int = 3            # each module = stall row + drive aisle + stall row
    stall_width_m: float = 2.7          # ~8.9 ft
    stall_depth_m: float = 5.5          # ~18 ft
    aisle_width_m: float = 7.3          # ~24 ft two-way aisle, 90-degree parking
    cross_aisle_width_m: float = 7.5    # end aisles joining the modules
    ramp_strip_width_m: float = 7.5     # drive strip along the south wall that holds the ramps
    ramp_length_m: float = 24.0         # horizontal run of each inter-level ramp (~14% slope)
    floor_to_floor_m: float = 3.4
    slab_thickness_m: float = 0.30
    wall_thickness_m: float = 0.45      # retaining wall
    column_size_m: float = 0.60
    stalls_per_bay: int = 3             # column grid spacing, in stalls
    core_stalls: int = 3                # stalls displaced by each stair/elevator core
    slab_sections_per_level: int = 4    # maintenance zones per level (closed one at a time)
    fans_per_level: int = 4             # exhaust fans per level
    light_spacing_m: float = 5.4        # LED luminaire spacing along drive aisles
    # stall zoning (UTD-style permits)
    ada_fraction: float = 0.02          # ADA 208.2: 2% of spaces when 501-1000 total
    ev_stalls: int = 24
    visitor_stalls: int = 40
    staff_fraction: float = 0.30        # share of stalls reserved for faculty/staff permits


# --------------------------------------------------------------------------- #
# One day of traffic + air quality
# --------------------------------------------------------------------------- #
@dataclass
class TrafficParams:
    daily_demand: int = 1650            # vehicles that try to enter on a typical Mon-Wed class day
    start_hour: float = 5.0
    end_hour: float = 24.0
    # relative arrival intensity per clock hour (0..23); peaks ahead of UTD class start times
    hourly_profile: list = field(default_factory=lambda: [
        0.05, 0.02, 0.02, 0.02, 0.05, 0.3, 1.5, 5.0, 9.5, 10.0, 9.0, 8.0,
        8.5, 7.0, 6.0, 5.0, 5.0, 5.5, 4.0, 2.0, 1.0, 0.5, 0.3, 0.1])
    user_mix: dict = field(default_factory=lambda: {"student": 0.62, "staff": 0.25, "visitor": 0.13})
    dwell_mean_h: dict = field(default_factory=lambda: {"student": 3.2, "staff": 7.5, "visitor": 1.5})
    dwell_cv: dict = field(default_factory=lambda: {"student": 0.55, "staff": 0.25, "visitor": 0.7})
    ev_fraction: float = 0.08
    ada_fraction: float = 0.03
    entry_lanes: int = 2
    exit_lanes: int = 2
    entry_service_s: dict = field(default_factory=lambda: {"student": 6, "staff": 6, "visitor": 28})
    exit_service_s: dict = field(default_factory=lambda: {"student": 5, "staff": 5, "visitor": 22})
    drive_speed_kmh: float = 10.0
    cruise_penalty_min: float = 6.0     # extra search time when the garage is nearly full
    cruise_exponent: float = 3.0
    # carbon monoxide mass balance per level
    co_emission_g_per_min: float = 1.2  # per moving vehicle (warm-ish engine, ASHRAE Ch.16 range)
    co_background_ppm: float = 0.5
    co_alarm_ppm: float = 25.0          # ACGIH TLV / typical garage alarm set-point
    co_limit_ppm: float = 35.0          # ASHRAE 1-hour design ceiling
    vent_cfm_per_ft2: float = 0.75      # IMC 403.3 enclosed parking garages
    ventilation_mode: str = "demand"    # "demand" (CO-controlled VFDs) or "constant"
    dcv_min_speed: float = 0.3
    dcv_low_ppm: float = 9.0
    dcv_high_ppm: float = 25.0
    fan_kw_per_m3s: float = 1.1         # fan power at full speed per m3/s moved
    electricity_usd_per_kwh: float = 0.09


# --------------------------------------------------------------------------- #
# Multi-year maintenance / lifecycle
# --------------------------------------------------------------------------- #
@dataclass
class MaintenanceParams:
    start_date: str = "2027-01-01"
    years: int = 20
    discount_rate: float = 0.04
    demand_growth_per_year: float = 0.02   # enrollment growth
    weekday_factor: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 0.95, 0.7, 0.18, 0.12])
    # [start MM-DD, end MM-DD, demand factor] - approximate UTD academic calendar
    academic_calendar: list = field(default_factory=lambda: [
        ["01-01", "01-18", 0.10], ["01-19", "03-14", 1.00], ["03-15", "03-21", 0.25],
        ["03-22", "05-08", 1.00], ["05-09", "05-24", 0.25], ["05-25", "08-07", 0.45],
        ["08-08", "08-17", 0.25], ["08-18", "11-22", 1.00], ["11-23", "11-29", 0.20],
        ["11-30", "12-12", 1.00], ["12-13", "12-31", 0.10]])

    # concrete deck condition index (CI: 100 = new, 0 = failed)
    slab_initial_ci: float = 100.0
    slab_base_loss_per_year: float = 2.2      # moisture / carbonation / chloride
    slab_wear_per_1000_passes: float = 0.012  # traffic abrasion & fatigue
    slab_rate_sigma: float = 0.25             # section-to-section variability (lognormal)
    slab_groundwater_factor: float = 0.4      # deepest level deteriorates this much faster
    slab_nonlinear: float = 1.0               # acceleration once corrosion starts
    sealant_life_years: float = 6.0           # traffic membrane / sealer effectiveness
    sealant_protection: float = 0.5           # fraction of deterioration prevented while sealed
    sealant_cost_m2: float = 22.0
    sealant_closure_days: int = 2
    slab_critical_ci: float = 30.0            # spalling hazard -> emergency closure
    slab_restored_ci: float = 95.0
    slab_repair_cost_m2: float = 380.0        # partial-depth concrete repair
    repair_area_factor: float = 1.2           # repaired area fraction per CI point lost
    repair_mobilization: float = 25000.0
    repair_days_base: float = 7.0
    repair_days_per_point: float = 0.35
    repair_lead_days: int = 45                # design + bid for a planned repair
    emergency_lead_days: int = 5
    emergency_cost_factor: float = 1.8
    emergency_days_factor: float = 1.3
    inspection_cost: float = 12000.0
    inspection_sd: float = 4.0                # CI measurement error

    # plaza waterproofing membrane above B1
    membrane_loss_per_year: float = 3.5
    membrane_cost_m2: float = 95.0
    membrane_failed_ci: float = 15.0
    leak_slab_accel: float = 1.5              # B1 deterioration multiplier at 100% leakage

    # storms, sump pumps, flooding
    heavy_rain_prob_by_month: list = field(default_factory=lambda: [
        0.02, 0.025, 0.035, 0.045, 0.06, 0.04, 0.02, 0.02, 0.035, 0.045, 0.03, 0.025])
    storm_intensity_mean: float = 0.5
    pumps_per_unit_intensity: float = 0.8
    leak_inflow: float = 1.0                  # extra pumping demand from a leaking membrane
    n_sump_pumps: int = 3
    flood_closure_days: list = field(default_factory=lambda: [3, 8])
    flood_cleanup_cost: float = 45000.0
    flood_claim_per_vehicle: float = 5000.0
    flood_occupied_fraction: float = 0.4
    flood_damage_fraction: float = 0.2
    flood_slab_damage: float = 4.0

    # safety incidents
    spall_incident_base_per_day: float = 0.002   # per open section at the critical CI
    spall_claim_cost: float = 20000.0
    dark_incident_base_per_day: float = 0.0005   # at 95% lighting availability
    dark_claim_cost: float = 15000.0

    # mechanical / electrical equipment (Weibull life, years)
    min_fan_fraction: float = 0.5           # level closes if fewer fans than this run
    n_elevators: int = 2
    planned_cost_factor: float = 0.6        # planned fix vs breakdown repair
    manual_gate_cost_per_day: float = 400.0
    light_watts: float = 40.0
    units: dict = field(default_factory=lambda: {
        "vent_fan":  {"beta": 2.0, "eta_years": 9.0, "repair_cost": 6500, "replace_cost": 14000,
                      "pm_cost": 450, "pm_restoration": 0.35, "mttr_days": 10},
        "sump_pump": {"beta": 1.7, "eta_years": 6.0, "repair_cost": 8500, "replace_cost": 15000,
                      "pm_cost": 350, "pm_restoration": 0.40, "mttr_days": 6},
        "gate":      {"beta": 1.3, "eta_years": 3.0, "repair_cost": 2200, "replace_cost": 9000,
                      "pm_cost": 250, "pm_restoration": 0.30, "mttr_days": 2},
        "elevator":  {"beta": 1.5, "eta_years": 1.2, "repair_cost": 3500, "replace_cost": 60000,
                      "pm_cost": 400, "pm_restoration": 0.25, "mttr_days": 4},
        "light":     {"beta": 3.0, "eta_years": 7.0, "repair_cost": 320, "replace_cost": 210,
                      "pm_cost": 0, "pm_restoration": 0.0, "mttr_days": 21},
    })

    # fixed operating costs per year
    cleaning_per_year: float = 60000.0
    fire_life_safety_per_year: float = 9000.0
    elevator_inspection_per_year: float = 3000.0

    # economics of turning cars away
    visitor_fee: float = 6.0
    denied_user_cost: float = 12.0          # value of a driver's time hunting for another lot


@dataclass
class PolicyParams:
    name: str = "custom"
    description: str = ""
    inspection_interval_days: int | None = None
    slab_repair_ci: float = 0.0             # planned repair when inspected CI is below this
    membrane_replace_ci: float | None = None
    sealant_interval_years: float | None = None
    pm_interval_days: dict = field(default_factory=dict)     # unit group -> days between PM visits
    replace_age_years: dict = field(default_factory=dict)    # unit group -> planned replacement age
    monitored_groups: list = field(default_factory=list)     # groups with condition sensors
    detect_prob: float = 0.0                # chance a sensor flags a failure before it happens
    mttr_factor: float = 1.0                # <1 when spares are stocked
    corrective_premium: float = 1.5         # overtime / call-out multiplier on breakdowns
    annual_program_cost: float = 0.0        # CMMS, sensors, staff time to run the program


def default_policies() -> dict:
    return {
        "reactive": PolicyParams(
            name="reactive",
            description="Run to failure: fix things when they break or become hazardous.",
            corrective_premium=1.5),
        "preventive": PolicyParams(
            name="preventive",
            description="Annual inspections, scheduled PM, sealer every 5 years, group relamping.",
            inspection_interval_days=365, slab_repair_ci=50.0, membrane_replace_ci=35.0,
            sealant_interval_years=5.0,
            pm_interval_days={"vent_fan": 180, "sump_pump": 90, "gate": 180, "elevator": 30},
            replace_age_years={"light": 6.0},
            mttr_factor=0.6, corrective_premium=1.15, annual_program_cost=15000.0),
        "predictive": PolicyParams(
            name="predictive",
            description="Condition-based: semiannual inspections plus vibration/current sensors on equipment.",
            inspection_interval_days=182, slab_repair_ci=60.0, membrane_replace_ci=45.0,
            sealant_interval_years=5.0,
            pm_interval_days={"vent_fan": 365, "sump_pump": 180, "elevator": 30},
            monitored_groups=["vent_fan", "sump_pump", "elevator", "gate", "light"],
            detect_prob=0.75, mttr_factor=0.5, corrective_premium=1.1, annual_program_cost=40000.0),
    }


@dataclass
class SimConfig:
    geometry: GeometryParams = field(default_factory=GeometryParams)
    traffic: TrafficParams = field(default_factory=TrafficParams)
    maintenance: MaintenanceParams = field(default_factory=MaintenanceParams)
    policies: dict = field(default_factory=default_policies)

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "SimConfig":
        cfg = cls()
        with open(path) as fh:
            data = json.load(fh)
        for section in ("geometry", "traffic", "maintenance"):
            _merge(getattr(cfg, section), data.get(section, {}))
        for name, pol in data.get("policies", {}).items():
            base = cfg.policies.get(name, PolicyParams(name=name))
            _merge(base, pol)
            cfg.policies[name] = base
        return cfg


def _merge(obj, updates: dict) -> None:
    valid = {f.name for f in fields(obj)}
    for key, val in updates.items():
        if key not in valid:
            raise KeyError(f"Unknown parameter '{key}' for {type(obj).__name__}")
        cur = getattr(obj, key)
        if isinstance(cur, dict) and isinstance(val, dict):
            merged = dict(cur)
            for k, v in val.items():
                merged[k] = {**merged[k], **v} if isinstance(merged.get(k), dict) and isinstance(v, dict) else v
            setattr(obj, key, merged)
        else:
            setattr(obj, key, val)
