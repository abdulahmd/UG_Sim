"""
Parametric 3-D (CAD) model of the underground garage.

Coordinate system (metres): x runs along the garage length (west -> east),
y across its width (south -> north), z is elevation with street grade at 0.
Level index 0 is B1 (first basement), 1 is B2, and so on.

Plan of every level:

    y=W  +-------------------------------------------------------+
         | X | core |  stall row B                   | core |  X |
         | X |----------- drive aisle ----------------------|  X |   <- one "aisle module"
         | X |        stall row A                           |  X |      (repeated n times)
         | X |  ...                                         |  X |
    y=0  +---+----------------------------------------------+----+
         |   [ramp to/from level above]   [ramp to level below]  |   <- ramp strip
    y=-S +-------------------------------------------------------+
         x=0  ^cross aisle                          cross aisle^ x=L

Ramps alternate ends (west, east, west...) so drivers spiral down.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np

from .config import GeometryParams

ZONES = ("ada", "ev", "visitor", "staff", "general")

KIND_COLORS = {
    "slab": "#bdbcb6", "roof": "#9fb48f", "wall": "#8a8178", "guard": "#8a8178",
    "column": "#dcd8cf", "ramp": "#e0a458", "core": "#5b7db1", "fan": "#3f8f80",
    "light": "#f2e46b", "pump": "#2f5f9a", "gate": "#d1495b", "booth": "#7d5ba6",
    "stall_general": "#ececec", "stall_staff": "#e9c46a", "stall_visitor": "#b392ac",
    "stall_ada": "#4a90d9", "stall_ev": "#57b46d",
}
ZONE_COLORS = {z: KIND_COLORS["stall_" + z] for z in ZONES}

# AutoCAD colour index per layer for DXF export
ACI = {"slab": 8, "roof": 3, "wall": 9, "guard": 9, "column": 7, "ramp": 30, "core": 5, "fan": 4,
       "light": 2, "pump": 150, "gate": 1, "booth": 6, "stall_general": 254, "stall_staff": 40,
       "stall_visitor": 210, "stall_ada": 140, "stall_ev": 90, "car": 1}

# vertex order of a hexahedron: bottom 0-3 counter-clockwise seen from above, top 4-7 above them
HEX_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


@dataclass
class Solid:
    kind: str
    verts: np.ndarray  # (8, 3)
    level: int = -1

    def faces(self) -> list[np.ndarray]:
        return [self.verts[list(f)] for f in HEX_FACES]


def box(kind, x0, y0, z0, x1, y1, z1, level=-1) -> Solid:
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    return Solid(kind, v, level)


def ramp_solid(x_top, x_bot, y0, y1, z_top, z_bot, t, level) -> Solid:
    """Sloped slab whose driving surface falls from z_top at x_top to z_bot at x_bot."""
    xa, xb = sorted((x_top, x_bot))
    za, zb = (z_top, z_bot) if xa == x_top else (z_bot, z_top)
    v = np.array([[xa, y0, za - t], [xb, y0, zb - t], [xb, y1, zb - t], [xa, y1, za - t],
                  [xa, y0, za], [xb, y0, zb], [xb, y1, zb], [xa, y1, za]], float)
    return Solid("ramp", v, level)


@dataclass
class Stall:
    id: int
    level: int
    x0: float
    y0: float
    x1: float
    y1: float
    aisle_y: float
    section: int
    zone: str = "general"
    dist: float = 0.0                       # driving distance from the entry gate (m)
    path: list = field(default_factory=list)  # in-level route from ramp foot to stall
    level_share: np.ndarray | None = None   # metres driven inside each level to get here

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Garage:
    p: GeometryParams
    solids: list
    stalls: list
    L: float
    W: float
    S: float
    floor_z: list            # top-of-slab elevation of each level
    roof_z: float
    cores: list              # (x0, x1, y0, y1)
    columns: list            # (x, y)
    lights: list             # (level, x, y)
    fans: list               # (level, x, y)
    gate_xy: tuple
    explode_gap: float

    @property
    def n_levels(self) -> int:
        return self.p.n_levels

    @property
    def level_area_m2(self) -> float:
        return self.L * (self.W + self.S)

    @property
    def level_volume_m3(self) -> float:
        return self.level_area_m2 * (self.p.floor_to_floor_m - self.p.slab_thickness_m)

    @property
    def n_lights(self) -> int:
        return len(self.lights)

    def ramp_span(self, k: int) -> tuple[float, float]:
        """(x at top, x at bottom) of the ramp that arrives at level k."""
        ca, RL = self.p.cross_aisle_width_m, self.p.ramp_length_m
        return (ca + RL, ca) if k % 2 == 0 else (self.L - ca - RL, self.L - ca)

    def stall_counts(self) -> dict:
        out = {}
        for s in self.stalls:
            out.setdefault(s.level, {z: 0 for z in ZONES})[s.zone] += 1
        return out

    def describe(self) -> str:
        lines = [f"Footprint {self.L:.0f} m x {self.W + self.S:.1f} m, {self.n_levels} levels below grade, "
                 f"deepest floor at {self.floor_z[-1]:.1f} m",
                 f"Total stalls: {len(self.stalls)}  |  lights: {self.n_lights}  |  exhaust fans: {len(self.fans)}"]
        for lvl, zc in sorted(self.stall_counts().items()):
            parts = ", ".join(f"{z} {n}" for z, n in zc.items() if n)
            lines.append(f"  B{lvl + 1}: {sum(zc.values()):4d} stalls  ({parts})")
        return "\n".join(lines)


def build_garage(p: GeometryParams, explode_gap: float = 0.0) -> Garage:
    """Build the garage. explode_gap > 0 pulls the levels apart for viewing."""
    n, L = p.n_levels, p.length_m
    D, A, sw = p.stall_depth_m, p.aisle_width_m, p.stall_width_m
    ca, S, RL = p.cross_aisle_width_m, p.ramp_strip_width_m, p.ramp_length_m
    t, ftf, c = p.slab_thickness_m, p.floor_to_floor_m, p.column_size_m
    Mw = 2 * D + A
    W = p.n_aisle_modules * Mw
    if 2 * (ca + RL) > L:
        raise ValueError("length_m is too short to fit two ramps plus the cross aisles")
    n_cols = int((L - 2 * ca) // sw)
    x_end = ca + n_cols * sw

    def floor_z(k):  # k = -1 is the plaza/roof at street grade
        return -(k + 1) * ftf + (n - 1 - k) * explode_gap

    def ramp_span(k):
        return (ca + RL, ca) if k % 2 == 0 else (L - ca - RL, L - ca)

    solids: list[Solid] = []

    # --- slabs: full deck over the parking grid, ramp strip with an opening for the ramp below
    for k in range(-1, n):
        z = floor_z(k)
        kind = "roof" if k == -1 else "slab"
        thick = 2 * t if k == n - 1 else t  # mat foundation under the lowest level
        solids.append(box(kind, 0, 0, z - thick, L, W, z, k))
        segs = [(0.0, L)]
        if k + 1 < n:
            h0, h1 = sorted(ramp_span(k + 1))
            segs = [(0.0, h0), (h1, L)]
            # guard wall between the stall grid and the opening
            solids.append(box("guard", h0, -0.25, z, h1, 0.0, z + 1.1, k))
        for a, b in segs:
            if b > a:
                solids.append(box(kind, a, -S, z - thick, b, 0.0, z, k))

    # --- ramps
    for k in range(n):
        xt, xb = ramp_span(k)
        solids.append(ramp_solid(xt, xb, -S, 0.0, floor_z(k - 1), floor_z(k), t, k))

    # --- perimeter retaining walls (cut-away view hides the near ones)
    if explode_gap == 0:
        wt = p.wall_thickness_m
        zb = floor_z(n - 1) - 2 * t
        solids += [box("wall", -wt, -S - wt, zb, L + wt, -S, 0.0),
                   box("wall", -wt, W, zb, L + wt, W + wt, 0.0),
                   box("wall", -wt, -S, zb, 0.0, W, 0.0),
                   box("wall", L, -S, zb, L + wt, W, 0.0)]

    # --- column grid on the back-to-back stall lines
    bay = p.stalls_per_bay * sw
    col_x = [ca + i * bay for i in range(n_cols // p.stalls_per_bay + 1)]
    if col_x[-1] < x_end - 1e-6:
        col_x.append(x_end)
    col_y = [max(m * Mw, c / 2) for m in range(p.n_aisle_modules)]
    columns = [(x, y) for x in col_x for y in col_y]
    for k in range(n):
        z0, z1 = floor_z(k), floor_z(k - 1) - t
        for x, y in columns:
            solids.append(box("column", x - c / 2, y - c / 2, z0, x + c / 2, y + c / 2, z1, k))

    # --- stair / elevator cores at the north-west and north-east corners
    cw = p.core_stalls * sw
    cores = [(ca, ca + cw, W - D, W), (x_end - cw, x_end, W - D, W)]
    for x0, x1, y0, y1 in cores:
        for k in range(n):
            solids.append(box("core", x0, y0, floor_z(k), x1, y1, floor_z(k - 1) - t, k))
        solids.append(box("core", x0, y0, floor_z(-1), x1, y1, floor_z(-1) + 3.2, -1))  # headhouse

    # --- exhaust fans hung under the deck along the north wall
    fans = []
    for k in range(n):
        zc = floor_z(k - 1) - t
        for i in range(p.fans_per_level):
            x = ca + (i + 0.5) * (x_end - ca) / p.fans_per_level
            fans.append((k, x, W - 1.0))
            solids.append(box("fan", x - 0.7, W - 1.7, zc - 0.9, x + 0.7, W - 0.3, zc, k))

    # --- LED luminaires along drive aisles, the ramp strip and the cross aisles
    lights = []
    for k in range(n):
        zc = floor_z(k - 1) - t
        lines = [(ca / 2, x_end + ca / 2, m * Mw + D + A / 2) for m in range(p.n_aisle_modules)]
        lines.append((ca / 2, L - ca / 2, -S / 2))
        for xa, xb, y in lines:
            for x in np.arange(xa + p.light_spacing_m / 2, xb, p.light_spacing_m):
                lights.append((k, float(x), y))
                solids.append(box("light", x - 0.6, y - 0.08, zc - 0.1, x + 0.6, y + 0.08, zc, k))
        for x in (ca / 2, L - ca / 2):
            for y in np.arange(p.light_spacing_m / 2, W, p.light_spacing_m):
                lights.append((k, x, float(y)))
                solids.append(box("light", x - 0.08, y - 0.6, zc - 0.1, x + 0.08, y + 0.6, zc, k))

    # --- sump pit in the lowest level, entry gates + booth on the plaza
    zl = floor_z(n - 1)
    solids.append(box("pump", L - 4.0, -S + 0.5, zl - 1.5, L - 1.0, -S + 3.0, zl + 0.15, n - 1))
    gx, gy, zr = ca + RL + 6.0, -S / 2, floor_z(-1)
    solids.append(box("booth", gx - 1.0, gy - 0.8, zr, gx + 1.0, gy + 0.8, zr + 2.6, -1))
    solids.append(box("gate", gx - 0.05, -S + 0.2, zr + 0.95, gx + 0.05, gy - 0.8, zr + 1.05, -1))
    solids.append(box("gate", gx - 0.05, gy + 0.8, zr + 0.95, gx + 0.05, -0.2, zr + 1.05, -1))

    # --- stalls
    n_sec = p.slab_sections_per_level
    sec_len = (x_end - ca) / n_sec
    stalls: list[Stall] = []
    for k in range(n):
        for m in range(p.n_aisle_modules):
            ya = m * Mw
            aisle_y = ya + D + A / 2
            for y0, y1 in ((ya, ya + D), (ya + D + A, ya + Mw)):
                for i in range(n_cols):
                    x0 = ca + i * sw
                    x1 = x0 + sw
                    if any(x0 < cx1 and x1 > cx0 and y0 < cy1 and y1 > cy0 for cx0, cx1, cy0, cy1 in cores):
                        continue
                    sec = min(int((x0 + sw / 2 - ca) // sec_len), n_sec - 1)
                    stalls.append(Stall(-1, k, x0, y0, x1, y1, aisle_y, sec))

    # --- driving distances from the gate (used for stall choice, travel time and CO emissions)
    ys = -S / 2
    ramp_len = math.hypot(RL, ftf)
    approach = np.full(n, ramp_len)               # ramp descending into each level
    approach[0] += gx - (ca + RL)                 # gate to top of the entry ramp
    transit = np.array([abs(ramp_span(j)[1] - ramp_span(j + 1)[0]) for j in range(n - 1)] + [0.0])
    for s in stalls:
        xa = ramp_span(s.level)[1]
        best = None
        for xc in (ca / 2, L - ca / 2):
            d_in = abs(xa - xc) + abs(ys - s.aisle_y) + abs(xc - s.cx) + abs(s.aisle_y - s.cy)
            if best is None or d_in < best[0]:
                best = (d_in, [(xa, ys), (xc, ys), (xc, s.aisle_y), (s.cx, s.aisle_y), (s.cx, s.cy)])
        share = np.zeros(n)
        share[:s.level] = approach[:s.level] + transit[:s.level]
        share[s.level] = approach[s.level] + best[0]
        s.level_share, s.path, s.dist = share, best[1], float(share.sum())

    # --- permit zoning: ADA by the elevators, visitors and staff closest to the entrance
    core_c = [((a + b) / 2, (c0 + c1) / 2) for a, b, c0, c1 in cores]
    by_dist = sorted(stalls, key=lambda s: s.dist)
    n_ada = max(2, math.ceil(p.ada_fraction * len(stalls)))
    near_core = sorted((s for s in stalls if s.level == 0),
                       key=lambda s: min(math.hypot(s.cx - x, s.cy - y) for x, y in core_c))
    for s in near_core[:n_ada]:
        s.zone = "ada"
    for s in [s for s in by_dist if s.level == 0 and s.zone == "general"][:p.visitor_stalls]:
        s.zone = "visitor"
    ev_level = min(1, n - 1)
    for s in [s for s in by_dist if s.level == ev_level and s.zone == "general"][:p.ev_stalls]:
        s.zone = "ev"
    for s in [s for s in by_dist if s.zone == "general"][:round(p.staff_fraction * len(stalls))]:
        s.zone = "staff"
    for i, s in enumerate(by_dist):
        s.id = i  # ids ordered by distance, so the lowest free id is the closest stall
    stalls = by_dist

    for s in stalls:  # painted stall surfaces, inset so the slab shows through as striping
        z = floor_z(s.level)
        solids.append(box("stall_" + s.zone, s.x0 + 0.06, s.y0 + 0.06, z + 0.005,
                          s.x1 - 0.06, s.y1 - 0.06, z + 0.015, s.level))

    return Garage(p, solids, stalls, L, W, S, [floor_z(k) for k in range(n)], floor_z(-1),
                  cores, columns, lights, fans, (gx, gy), explode_gap)


def car_solid(garage: Garage, stall: Stall, kind="car") -> Solid:
    """A simple vehicle block parked in a stall."""
    z = garage.floor_z[stall.level]
    return box(kind, stall.cx - 0.92, stall.cy - 2.3, z + 0.02, stall.cx + 0.92, stall.cy + 2.3, z + 1.45,
               stall.level)


# --------------------------------------------------------------------------- #
# CAD exports
# --------------------------------------------------------------------------- #
def _triangles(solids):
    for s in solids:
        for f in s.faces():
            yield f[0], f[1], f[2]
            yield f[0], f[2], f[3]


def export_stl(garage: Garage, path: str) -> int:
    """Binary STL (opens in Fusion 360, SolidWorks, FreeCAD, Blender, any slicer)."""
    tris = list(_triangles(garage.solids))
    with open(path, "wb") as fh:
        fh.write(b"UTD underground parking garage".ljust(80, b" "))
        fh.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            nrm = np.cross(b - a, c - a)
            ln = np.linalg.norm(nrm)
            nrm = nrm / ln if ln else nrm
            fh.write(struct.pack("<12fH", *nrm, *a, *b, *c, 0))
    return len(tris)


def export_obj(garage: Garage, path: str) -> None:
    """Wavefront OBJ + MTL with one group/material per component type (keeps colours)."""
    mtl_path = path.rsplit(".", 1)[0] + ".mtl"
    with open(mtl_path, "w") as fh:
        for kind, hexc in KIND_COLORS.items():
            r, g, b = (int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))
            fh.write(f"newmtl {kind}\nKd {r:.3f} {g:.3f} {b:.3f}\nKa 0.1 0.1 0.1\n\n")
    by_kind: dict[str, list] = {}
    for s in garage.solids:
        by_kind.setdefault(s.kind, []).append(s)
    with open(path, "w") as fh:
        fh.write(f"# UTD underground parking garage\nmtllib {mtl_path.rsplit('/', 1)[-1]}\n")
        base = 1
        for kind, solids in by_kind.items():
            fh.write(f"g {kind}\nusemtl {kind}\n")
            for s in solids:
                for v in s.verts:
                    fh.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
                for f in HEX_FACES:
                    fh.write("f " + " ".join(str(base + i) for i in f) + "\n")
                base += 8


def export_dxf(garage: Garage, path: str) -> None:
    """AutoCAD R12 DXF of the 3-D model (3DFACE entities, one layer per component)."""
    out = ["0", "SECTION", "2", "ENTITIES"]
    for s in garage.solids:
        layer = s.kind.upper()
        for f in s.faces():
            out += ["0", "3DFACE", "8", layer, "62", str(ACI.get(s.kind, 7))]
            for i, v in enumerate(f):
                out += [f"1{i}", f"{v[0]:.3f}", f"2{i}", f"{v[1]:.3f}", f"3{i}", f"{v[2]:.3f}"]
    out += ["0", "ENDSEC", "0", "EOF"]
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")


def export_plan_dxf(garage: Garage, path: str) -> None:
    """2-D floor plans of every level side by side (LINE/TEXT entities) for drafting."""
    out = ["0", "SECTION", "2", "ENTITIES"]

    def rect(layer, x0, y0, x1, y1, dy):
        nonlocal out
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        for (a, b), (c, d) in zip(pts, pts[1:]):
            out += ["0", "LINE", "8", layer, "10", f"{a:.3f}", "20", f"{b + dy:.3f}", "30", "0",
                    "11", f"{c:.3f}", "21", f"{d + dy:.3f}", "31", "0"]

    def text(layer, x, y, h, s):
        nonlocal out
        out += ["0", "TEXT", "8", layer, "10", f"{x:.3f}", "20", f"{y:.3f}", "30", "0", "40", str(h), "1", s]

    g, c = garage, garage.p.column_size_m
    for k in range(g.n_levels):
        dy = -k * (g.W + g.S + 15)
        rect("WALLS", 0, -g.S, g.L, g.W, dy)
        text("TEXT", 0, g.W + 3 + dy, 3.0, f"LEVEL B{k + 1}")
        for kk in (k, k + 1):
            if kk < g.n_levels:
                x0, x1 = sorted(g.ramp_span(kk))
                rect("RAMPS", x0, -g.S, x1, 0, dy)
        for x, y in g.columns:
            rect("COLUMNS", x - c / 2, y - c / 2, x + c / 2, y + c / 2, dy)
        for x0, x1, y0, y1 in g.cores:
            rect("CORES", x0, y0, x1, y1, dy)
        for s in g.stalls:
            if s.level == k:
                rect("STALLS_" + s.zone.upper(), s.x0, s.y0, s.x1, s.y1, dy)
                text("STALL_NUMBERS", s.x0 + 0.3, s.cy + dy, 0.6, str(s.id))
    out += ["0", "ENDSEC", "0", "EOF"]
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
