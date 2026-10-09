# UTD Underground Parking Garage Simulator

A parametric CAD model and a simulation of a three-level underground parking garage. It covers daily traffic, air quality, and 20 years of maintenance and lifecycle cost. It was built for SYSE 2300 (Systems Engineering Management).

Requires only Python 3.10+, `numpy` and `matplotlib` (`pip install -r requirements.txt`).

## Quick start

One command installs everything, runs the whole simulation and opens each result in a window
(rotatable 3-D garage model, traffic charts, live animated parking map, lifecycle and policy comparison).
Close each window to move on to the next; everything is also saved to `./outputs`:

```bash
pip install -r requirements.txt && python main.py all --show
```

Or run the pieces on their own:

```bash
python main.py all                 # build everything into ./outputs  (~2 min, mostly the GIF)
python main.py model --show        # rotatable 3-D model window
python main.py traffic             # one class day
python main.py animate             # outputs/garage_day.gif
python main.py maintenance --policy reactive      # or preventive | predictive | all
python main.py compare --reps 30   # Monte-Carlo policy trade study
```

### Change parameters

```bash
python main.py config              # writes params.json with every parameter
# edit params.json (stall counts, levels, demand, failure rates, costs, policies...)
python main.py all --config params.json --out outputs_myscenario
```

You can also run quick what-if checks from the command line:

```bash
python main.py traffic --demand 2200                 # demand surge
python main.py traffic --fans-down 0 3 0             # 3 of 4 exhaust fans dead on B2 -> watch CO
python main.py traffic --closed-stalls 120           # a deck section closed for repair
python main.py traffic --ventilation constant        # compare fan energy against CO-controlled VFDs
python main.py maintenance --policy all --years 30
```

## What each piece models

| Module | What it is | Key outputs |
|---|---|---|
| `garage_sim/geometry.py` | Parametric 3-D model: slabs, ramps (alternating ends), retaining walls, column grid, stair/elevator cores, exhaust fans, LED lights, sump pit, entry gates, and stalls zoned for ADA, EV, visitor, staff and general permits | `garage.stl`, `garage.obj/.mtl`, `garage_3d.dxf` (3-D, AutoCAD layers), `garage_plans.dxf` (2-D plans with stall numbers), renders, floor plans |
| `garage_sim/traffic.py` | Discrete-event simulation of one day. Arrivals follow a time-varying Poisson process that peaks before UTD class times. Driver types are student, staff and visitor; the model includes gate queues, choice of the nearest allowed stall, cruising delay as the garage fills, and lognormal parking durations. **Physics:** a CO mass balance on each level, `V dC/dt = G - Q(C - C_out)`, with CO-controlled fan speed and fan power that scales with the cube of speed | occupancy, queues, search time, turned-away cars, CO ppm, fan kWh, stall utilization heat map, per-car CSV |
| `garage_sim/maintenance.py` | Daily simulation over 20+ years. Concrete deck condition index per section wears down from moisture, groundwater and traffic, slowed by sealer. The plaza waterproofing membrane leaks as it ages. Fans, sump pumps, gates, elevators and lights fail on Weibull lifetimes. Heavy storms can flood the lowest level if too few pumps are working. Fan failures close levels, and spalling concrete or poor lighting cause incident claims | lifecycle cost (NPV), availability, floods, emergency repairs, annual cost CSV, event log |
| `garage_sim/traffic.py: build_metamodel` | A response surface fitted from DES runs that maps (demand, open stalls) to the fraction of cars turned away, so the 20-year model reflects capacity lost to closures | printed "knee" of the capacity curve |

### Maintenance policies (edit or add your own in `config.py` / `params.json`)

- **reactive**: run to failure. No inspections; repair only when something breaks or the concrete becomes hazardous.
- **preventive**: annual deck surveys, planned repair when CI < 50, sealer every 5 years, scheduled PM on fans, pumps, gates and elevators, and group relamping of lights at 6 years.
- **predictive**: surveys every 6 months, repair when CI < 60, and sensors that catch 75% of equipment failures before they happen. Stocked spares cut repair time.

`compare` runs every policy with the **same random seeds** (common random numbers). Differences between policies therefore come from the policy rather than from luck. It reports means with 95% confidence intervals.

## Systems-engineering concepts you can point to

- **Measures of effectiveness:** stall availability, cars turned away, gate wait, time from gate to stall, peak CO and minutes above the alarm level, ADA (elevator) outage days.
- **Lifecycle cost / NPV:** cost categories are program, preventive, corrective, structural, energy, operations, incidents and lost revenue. Driver cost is reported separately.
- **Reliability and maintainability:** Weibull failure models, MTTR, imperfect PM (each PM visit restores part of a unit's age), and k-out-of-n redundancy (pumps against storm demand, fans against ventilation needs).
- **Trade study:** Monte-Carlo comparison of policies, with boxplots and confidence intervals.
- **Capacity planning:** with 2%/yr enrollment growth, demand passes the capacity "knee" around year 8–10 under every policy. That points to a design or TDM decision, not only a maintenance one.

## Main assumptions (change any of them)

The dimensions follow common US practice: 2.7 × 5.5 m stalls, 7.3 m two-way aisles, 3.4 m floor-to-floor, and ramps of about 14%. Ventilation is 0.75 cfm/ft², per the International Mechanical Code (IMC) for enclosed garages. Unit costs, failure rates, storm probabilities and demand are **illustrative estimates, not UTD data**. Replace them with real values from UTD Parking & Transportation or Facilities Management if you can get them, and cite those sources in your report.

## Output files (`outputs/`)

`garage_3d_exploded.png`, `garage_3d_section.png`, `garage_3d_peak.png` (cars parked at peak), `floorplans.png`,
`traffic_day.png`, `driver_experience.png`, `stall_utilization.png`, `garage_day.gif`,
`lifecycle_<policy>.png/_annual.csv/_events.csv`, `policy_comparison.png/.csv`, plus the CAD files listed above.
