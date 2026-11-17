#!/usr/bin/env python3
# Stand-alone 1-D plug-flow reactor (n-hexane steam cracking, or air) — reference for TBLOCK validation
# N. Karefyllidis 2026
#
# Steady, constant-area tube of diameter D heated by a prescribed wall temperature Tw(x); a square duct with
# "cross_section": "square" (D = diameter_m is then the side, the hydraulic diameter 4A/P; 30 Oct 2026):
#     rho u = G                                   continuity
#     d(p + G u)/dx = -f G u / (2 D)              momentum, Darcy f (0 unless wall_friction)
#     G d(h + u^2/2)/dx = 4 q_w / D               energy, h with the formation enthalpies
#     q_w = htc (Tw - T)                          wall heat flux [W/m2]
#     u dY_k/dx = omega_k                         species [1/s]
# htc is a constant or "gnielinski" (local Re, Pr; laminar Nu 3.66, square 2.98), f is Petukhov (laminar 64/Re,
# square 56.91/Re); the turbulent correlations take the hydraulic diameter as they are;
# htc = 0 is the adiabatic duct. Marched cell by cell as a Lagrangian parcel: each cell integrates the
# chemistry at the cell-inlet T and p for dt = dx/u, adds the wall heat, then solves T, p, u from the
# total enthalpy, momentum and the ideal gas (Newton on u, subsonic branch).
#
# Gas models, same equations:
#   cantera : n-hexane/steam, the detailed 153-species mechanism (the truth RateNet was trained on)
#   ratenet : n-hexane/steam, the RateNet closure as TBLOCK runs it (RATENET 2): 13 hybrid13 lumps, rates from
#             models/<name>/ratenet.txt, lump gas from lump_thermo.txt (any other model name = such a folder)
#   air     : Cantera air.yaml, frozen (no chemistry), mixture-averaged transport
# Comparing TBLOCK against `ratenet` checks the CFD; `ratenet` against `cantera` checks the closure.
#
# Pilot-coil mode, for experiments (17 Nov 2026): "gas_temperature_profile" imposes the measured process-gas
# T(x) in place of the energy equation, "pressure_profile" the measured p(x) in place of the momentum equation,
# "mass_flow_kgs" sets the inlet velocity; "experiment_yields_wtpct" goes beside the model in the yields table.
#
#     python pfr_runner.py [config.json]          (default pfr_hexane.json; pfr_air.json for the air case)
# Writes <output_dir>/<config>_<model>.csv (TBLOCK 1-D names: CONV, Y_<lump>), <config>.png and, for the
# hexane models, <config>_lumps.png (the 13 lumps) and <config>_yields.csv (outlet wt% of the feed by formula).

import json
import re
import sys
import time
from pathlib import Path

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

_ROOT = Path(__file__).resolve().parent
LUMPS = ["feedstock", "ethylene", "propylene", "methane", "hydrogen", "light_paraffins", "butenes",
         "heavy_olefins", "alkynes", "dienes", "aromatics", "heavies", "diluent"]
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")   # yields chart: experiment, then the models, in this order

# _____________________________________________________________
#
# User settings (pfr_hexane.json)
#
#  _____________________________________________________________

CONFIG = Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT / "pfr_hexane.json"
if CONFIG.suffix != ".json" or not CONFIG.is_file():
    sys.exit(f"usage: python pfr_runner.py [config.json]   (e.g. pfr_hexane.json, pfr_air.json); got '{CONFIG}'")
cfg = json.load(open(CONFIG))
MODELS = cfg["models"]                         # "cantera", and RateNet exports by folder name under models/
MECHANISM = _ROOT / cfg.get("mechanism", "mechanism/n-Hexane_Kinetic-Model_species_153.yaml")
FEED, DILUENT = cfg.get("feed_species", "C6H14(1)"), cfg.get("diluent_species", "Water")
STEAM_RATIO = cfg.get("steam_ratio_kg_per_kg", 0.5)     # kg steam / kg n-hexane
AIR = cfg.get("air_composition", "O2:0.21, N2:0.79")   # mole fractions
# Pilot-coil mode (17 Nov 2026): the measured process-gas T(x) replaces the energy equation, and with
# pressure_profile the measured p(x) replaces the momentum equation; the inlet state is the first knot
T_KNOTS, P_KNOTS = cfg.get("gas_temperature_profile"), cfg.get("pressure_profile")
assert T_KNOTS or not P_KNOTS, "pressure_profile needs gas_temperature_profile"
T_0 = T_KNOTS[0]["temperature_K"] if T_KNOTS else cfg["inlet_T_K"]
P_0 = P_KNOTS[0]["pressure_Pa"] if P_KNOTS else cfg["inlet_p_Pa"]
U_0, MDOT = cfg.get("inlet_u_ms"), cfg.get("mass_flow_kgs")   # one of the two; MDOT = feed + steam [kg/s]
LENGTH, DIAMETER, N_CELLS = cfg["length_m"], cfg["diameter_m"], cfg["n_cells"]
MU_288 = cfg.get("tblock_mu288_Pas", 1.256e-5)  # TBLOCK laminar viscosity at 288 K, mu = MU_288 (T/288)^0.62
HTC = cfg.get("wall_htc_Wm2K", 0.0)            # W/m2/K, or "gnielinski"
FRICTION = cfg.get("wall_friction", False)
SQUARE = cfg.get("cross_section", "circle") == "square"   # diameter_m = the side (hydraulic diameter)
NU_LAM, F_LAM = (2.98, 56.91) if SQUARE else (3.66, 64.0)    # fully developed laminar, constant wall T
AREA = DIAMETER**2 if SQUARE else np.pi * DIAMETER**2 / 4.0
OUT_DIR = _ROOT / cfg["output_dir"]
KNOTS = cfg.get("wall_temperature_profile")
EXPERIMENT = cfg.get("experiment_yields_wtpct", {})   # measured outlet yields by formula, wt% of the feed


def profile(knots, key, x):
    """Prescribed value at x [m]: knots at position*LENGTH, 'step' or 'linear' to the next knot."""
    xk = np.array([k["position"] for k in knots]) * LENGTH
    i = min(max(np.searchsorted(xk, x, side="right") - 1, 0), len(knots) - 1)
    if i == len(knots) - 1 or knots[i].get("interpolation_to_next", "linear") == "step":
        return knots[i][key]
    f = (x - xk[i]) / (xk[i + 1] - xk[i])
    return knots[i][key] + f * (knots[i + 1][key] - knots[i][key])


# _____________________________________________________________
#
# RateNet and the hybrid13 lumping (copies: tblock scripts/chem/lump_of_phi.py,
# GeneralizedPFR src/ml/species_lumping.py — keep in sync)
#
#  _____________________________________________________________

class RateNet:
    """numpy twin of TBLOCK's module ratenet (ratenet_export v1 and v2; v2: asinh targets, R = r0 sinh(z s + m)).
    lumps: the model's lump names in its order (13 in hybrid13; 12 in ReactionRateNET v1.1, heavies in aromatics)."""

    def __init__(self, fname):
        lines = open(fname).read().split("\n")
        assert lines[0].strip() in ("ratenet_export v1", "ratenet_export v2"), f"{fname}: not ratenet_export v1 or v2"
        self.asinh = lines[0].strip() == "ratenet_export v2"
        it = iter(range(len(lines)))
        self.layers, self.chk_in, self.chk_out, self.bounds, self.rbounds, self.yidx, self.rsign = [], [], [], {}, [], [], []
        self.lumps = []
        sect = None
        for i in it:
            ln = lines[i].strip()
            if not ln:
                continue
            key = ln.split()[0]
            if key in ("feature_scaler_mean", "feature_scaler_scale", "target_scaler_mean", "target_scaler_scale",
                       "target_asinh_r0"):
                setattr(self, key, np.array(lines[i + 1].split(), float)); next(it)
            elif key in ("feature_bounds_min_max", "rate_bounds_min_max", "rate_to_species", "state_cols", "rate_cols"):
                sect = key
            elif key == "layer":
                nout = int(ln.split()[2])
                w = np.array([lines[i + 1 + r].split() for r in range(nout)], float)
                assert lines[i + 1 + nout].strip() == "bias"
                self.layers.append((w, np.array(lines[i + 2 + nout].split(), float)))
                for _ in range(nout + 2):
                    next(it)
                sect = None
            elif key == "check_vectors":
                sect = key
            elif key == "check_in":
                self.chk_in.append(np.array(lines[i + 1].split(), float)); next(it)
            elif key == "check_out":
                self.chk_out.append(np.array(lines[i + 1].split(), float)); next(it)
            elif sect == "state_cols" and key.startswith("Y_lump_chem_"):
                self.lumps.append(key[12:])
            elif sect == "feature_bounds_min_max":
                self.bounds[key] = (float(ln.split()[1]), float(ln.split()[2]))
            elif sect == "rate_bounds_min_max":
                self.rbounds.append((float(ln.split()[1]), float(ln.split()[2])))
            elif sect == "rate_to_species":
                self.yidx.append(int(ln.split()[1]) - 2); self.rsign.append(float(ln.split()[2]))
        self.rbounds = np.array(self.rbounds)
        self.tmin = 1.0 / self.bounds["inv_temperature_K"][1]
        self.tmax = 1.0 / self.bounds["inv_temperature_K"][0]
        # v1: error per output scale (the check vectors are torch float32, so near-zero outputs carry ~1e-4 noise);
        # v2: relative, with the asinh r0 as the floor, as TBLOCK's lump_of_phi.py --self-check
        out, ref = self(np.array(self.chk_in)), np.array(self.chk_out)
        err = np.max(np.abs(out - ref) / (np.abs(ref) + self.target_asinh_r0 if self.asinh else self.target_scaler_scale))
        assert err < 1e-3, f"RateNet disagrees with the file's check vectors ({err:.1e})"

    def __call__(self, x):
        h = (np.atleast_2d(x) - self.feature_scaler_mean) / self.feature_scaler_scale
        for k, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if k < len(self.layers) - 1:
                h = np.tanh(np.clip(h, -20, 20))
        z = h * self.target_scaler_scale + self.target_scaler_mean
        return self.target_asinh_r0 * np.sinh(np.clip(z, -700.0, 700.0)) if self.asinh else z

    def rates(self, T, P, rho, Y):
        """omega_k [1/s] per lump as ratenet_tb_omega: envelope clamps, sign, zero-lump guard, sum-zero projection."""
        x = np.column_stack([1.0 / np.clip(T, self.tmin, self.tmax), np.clip(P, *self.bounds["pressure_Pa"]),
                             np.clip(rho, *self.bounds["density_kgm3"]), Y])
        r = np.clip(self(x), self.rbounds[:, 0], self.rbounds[:, 1])
        om = np.zeros_like(Y)
        for ir, (k, sg) in enumerate(zip(self.yidx, self.rsign)):
            o = sg * r[:, ir]
            o[(o < 0) & (Y[:, k - 1] <= 0)] = 0.0
            om[:, k - 1] = o
        s, a = om.sum(1), np.abs(om).sum(1)
        om -= np.abs(om) * (np.where(a > 0, s / np.maximum(a, 1e-300), 0.0))[:, None]
        om[np.atleast_1d(T < self.tmin)] = 0.0
        return om


def species_formula(name):
    """(n_C, n_H) from an RMG-style name: C6H14(1), CH3, H-less SMILES CC / C#CC, or a known name."""
    base = name.split("(")[0]
    known = {"Water": (0, 2), "H2": (0, 2), "H": (0, 1), "Benzene": (6, 6), "Toluene": (7, 8), "Styrene": (8, 8)}
    if base in known:
        return known[base]
    m = re.fullmatch(r"C(\d*)H(\d*)", base)
    if m:
        return (int(m.group(1) or 1), int(m.group(2) or 1))
    if base and set(base) <= set("C=#"):
        n_c = base.count("C")
        return (n_c, 2 * n_c + 2 - 2 * base.count("=") - 4 * base.count("#"))
    return None


def classify_species_hybrid13(name, feed=(6, 14)):
    """Lump of a species in lumped_hybrid13 (formula-based; radicals -> parent molecule)."""
    base = name.split("(")[0]
    if base in ("Water", "Ar", "He", "Ne", "N2"):
        return "diluent"
    if base in ("H2", "H"):
        return "hydrogen"
    f = species_formula(name)
    if f is None:
        return "heavies"
    c, h = f
    if c == 0:
        return "diluent"
    radical = h % 2 == 1
    h += radical
    if (c, h) == feed:
        return "feedstock"
    if c == 1:
        return "methane"
    if c == 2:
        return "light_paraffins" if h >= 6 else "ethylene" if h == 4 else "alkynes"
    if c == 3:
        return "light_paraffins" if h >= 8 else "propylene" if h == 6 else "alkynes"
    if c == 4:
        return "butenes" if h >= 8 else "dienes" if h == 6 else "alkynes"
    if c == 5:
        return "heavy_olefins" if h >= 10 else "dienes"
    if c == 6:
        return "heavy_olefins" if h >= 10 else "dienes" if h == 8 else "aromatics"
    if radical:
        return "heavies"
    return "aromatics" if h <= c + 3 else "heavies"


# _____________________________________________________________
#
# Chemistry models: advance(Y, T, p, dt) -> Y at constant T, p;  thermo(Y, T) -> h [J/kg], cp [J/kg/K]
#
#  _____________________________________________________________


class Detailed:
    def __init__(self):
        self.gas = ct.Solution(str(MECHANISM))
        self.W = self.gas.molecular_weights
        self.lump = np.zeros((self.gas.n_species, len(LUMPS)))   # species -> lump mass matrix
        for i, s in enumerate(self.gas.species_names):
            self.lump[i, LUMPS.index(classify_species_hybrid13(s))] = 1.0
        self.gas.TPY = T_0, P_0, {FEED: 1.0, DILUENT: STEAM_RATIO}
        self.Y0 = self.gas.Y

    def advance(self, Y, T, p, dt):
        self.gas.TPY = T, p, Y
        r = ct.IdealGasConstPressureReactor(self.gas, energy="off", clone=False)
        ct.ReactorNet([r]).advance(dt)
        return self.gas.Y

    def thermo(self, Y, T):
        self.gas.TPY = T, P_0, Y
        return self.gas.enthalpy_mass, self.gas.cp_mass

    def mu(self, Y, T):
        self.gas.TPY = T, P_0, Y
        return self.gas.viscosity                   # mixture-averaged, from the mechanism's transport data

    def k(self, Y, T):
        self.gas.TPY = T, P_0, Y
        return self.gas.thermal_conductivity

    def columns(self, Y):
        L = Y @ self.lump
        return dict(CONV=1.0 - L[:, 0] / L[0, 0], **{"Y_" + k: L[:, j] for j, k in enumerate(LUMPS)})

    def yields(self, Y):
        """wt% of the feed by formula, largest first: isomers and radicals are summed, the mechanism names them by index."""
        out = {}
        for s, y in zip(self.gas.species(), Y):
            if s.name != DILUENT:
                f = "".join(el + (str(int(n)) if n != 1 else "") for el, n in sorted(s.composition.items()))
                out[f] = out.get(f, 0.0) + 100.0 * y / self.Y0[self.gas.species_index(FEED)]
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


class Air(Detailed):
    """Cantera air.yaml with its chemistry frozen."""

    def __init__(self):
        self.gas = ct.Solution("air.yaml")
        self.W = self.gas.molecular_weights
        self.gas.TPX = T_0, P_0, AIR
        self.Y0 = self.gas.Y

    def advance(self, Y, T, p, dt):
        return Y

    def columns(self, Y):
        return {}


class Lumped:
    def __init__(self, folder):
        d = _ROOT / "models" / folder
        self.net = RateNet(str(d / "ratenet.txt"))
        self.lumps = self.net.lumps                 # the model's own lumps, feedstock first and diluent last
        assert (self.lumps[0], self.lumps[-1]) == ("feedstock", "diluent") and set(self.lumps) <= set(LUMPS), self.lumps
        rows = [ln.split() for ln in open(d / "lump_thermo.txt") if ln.strip().startswith("Y_lump_chem_")][:len(self.lumps)]
        assert [r[0][12:] for r in rows] == self.lumps, "lump_thermo.txt lump order"
        self.W = np.array([float(r[1]) for r in rows])
        self.a = np.array([[float(v) for v in r[2:8]] for r in rows])     # NASA-7 a1..a6, one range
        self.Y0 = np.zeros(len(self.lumps))
        self.Y0[0], self.Y0[-1] = 1.0 / (1.0 + STEAM_RATIO), STEAM_RATIO / (1.0 + STEAM_RATIO)

    def advance(self, Y, T, p, dt):
        def rhs(t, y):
            rho = p / (ct.gas_constant * T * np.sum(y / self.W))
            return self.net.rates(T, p, rho, y[None, :])[0]
        return solve_ivp(rhs, (0.0, dt), Y, method="RK45", rtol=1e-8, atol=1e-12).y[:, -1]

    def thermo(self, Y, T):
        a, R = self.a, ct.gas_constant / self.W
        h = R * T * (a[:, 0] + a[:, 1] * T / 2 + a[:, 2] * T**2 / 3 + a[:, 3] * T**3 / 4 + a[:, 4] * T**4 / 5 + a[:, 5] / T)
        cp = R * (a[:, 0] + a[:, 1] * T + a[:, 2] * T**2 + a[:, 3] * T**3 + a[:, 4] * T**4)
        return Y @ h, Y @ cp

    def mu(self, Y, T):
        return MU_288 * (T / 288.0) ** 0.62         # TBLOCK's laminar law (deck REYNO = -MU_288); lumps carry no transport

    def k(self, Y, T):
        raise ValueError("the lump gas has no thermal conductivity: give wall_htc_Wm2K a number, not a correlation")

    def columns(self, Y):
        return dict(CONV=1.0 - Y[:, 0] / Y[0, 0], **{"Y_" + k: Y[:, j] for j, k in enumerate(self.lumps)})

    def yields(self, Y):
        """wt% of the feed for the lumps that are one molecule (with its radicals); the other lumps have no formula."""
        one = {"C6H14": "feedstock", "C2H4": "ethylene", "C3H6": "propylene", "CH4": "methane", "H2": "hydrogen"}
        return {f: 100.0 * Y[self.lumps.index(k)] / self.Y0[0] for f, k in one.items()}


# _____________________________________________________________
#
# PFR
#
#  _____________________________________________________________

def wall(chem, Y, T, G):
    """htc [W/m2/K], Darcy f and Re at a state: Gnielinski / Petukhov, laminar Nu 3.66 / 64/Re."""
    mu = chem.mu(Y, T)
    Re = G * DIAMETER / mu
    f = F_LAM / Re if Re < 2300 else (0.790 * np.log(Re) - 1.64) ** -2
    if HTC == "gnielinski":
        # ponytail: fully developed, no entrance-length or property-ratio correction
        k = chem.k(Y, T)
        Pr = chem.thermo(Y, T)[1] * mu / k
        Nu = NU_LAM if Re < 2300 else max(NU_LAM, f / 8 * (Re - 1000) * Pr / (1 + 12.7 * np.sqrt(f / 8) * (Pr ** (2 / 3) - 1)))
        htc = Nu * k / DIAMETER
    else:
        htc = HTC
    return htc, (f if FRICTION else 0.0), Re


def run(chem):
    x = np.linspace(0.0, LENGTH, N_CELLS + 1)
    rho0 = P_0 / (ct.gas_constant * T_0 * np.sum(chem.Y0 / chem.W))
    u0 = MDOT / (rho0 * AREA) if MDOT else U_0
    G = rho0 * u0
    C = P_0 + G * u0
    H = chem.thermo(chem.Y0, T_0)[0] + u0**2 / 2
    Y, T, p, u, t, wl = [chem.Y0], [T_0], [P_0], [u0], [0.0], [wall(chem, chem.Y0, T_0, G)]
    for i in range(N_CELLS):
        # ponytail: first order in dx (chemistry, wall heat and friction at the cell-inlet state); raise n_cells to converge
        dx = x[i + 1] - x[i]
        dt = dx / u[-1]
        htc, f, _ = wl[-1]
        Y.append(chem.advance(Y[-1], T[-1], p[-1], dt))
        C -= f * G * u[-1] / (2.0 * DIAMETER) * dx
        R = ct.gas_constant * np.sum(Y[-1] / chem.W)
        if T_KNOTS:                              # imposed gas T: u from p (imposed) or from momentum, no energy equation
            Ti = profile(T_KNOTS, "temperature_K", x[i + 1])
            if P_KNOTS:
                ui = G * R * Ti / profile(P_KNOTS, "pressure_Pa", x[i + 1])
                C = G * R * Ti / ui + G * ui     # so that p = C - G u below is the imposed p
            else:                                # subsonic root of u (C - G u) = G R T
                disc = C * C - 4.0 * G * G * R * Ti
                if disc < 0.0:
                    raise RuntimeError(f"no subsonic state at x = {x[i + 1]:.3f} m (choking): lower the mass flow")
                ui = (C - np.sqrt(disc)) / (2.0 * G)
            cp = chem.thermo(Y[-1], Ti)[1]
        else:
            H += 4.0 * htc * (profile(KNOTS, "temperature_K", x[i] + 0.5 * dx) - T[-1]) * dx / (G * DIAMETER)
            ui = u[-1]
            for _ in range(50):                  # h(T(u), Y) + u^2/2 = H with T(u) = u (C - G u) / (G R)
                Ti = ui * (C - G * ui) / (G * R)
                h, cp = chem.thermo(Y[-1], Ti)
                du = -(h + ui**2 / 2 - H) / (cp * (C - 2.0 * G * ui) / (G * R) + ui)
                ui += du
                if abs(du) < 1e-10 * ui:
                    break
            else:
                raise RuntimeError(f"no subsonic state at x = {x[i + 1]:.3f} m (choking): lower inlet_u_ms")
        if ui * ui >= cp / (cp - R) * R * Ti:    # Mach >= 1: past the sonic point (Rayleigh / Fanno)
            raise RuntimeError(f"choking at x = {x[i + 1]:.3f} m: lower inlet_u_ms")
        u.append(ui)
        T.append(ui * (C - G * ui) / (G * R))
        p.append(C - G * ui)
        t.append(t[-1] + dt)
        wl.append(wall(chem, Y[-1], T[-1], G))
    Y, T, p, u, t = (np.array(v) for v in (Y, T, p, u, t))
    htc, f, Re = np.array(wl).T
    assert np.abs(Y.sum(1) - 1.0).max() < 1e-6, "mass fractions drifted"

    h, cp = np.array([chem.thermo(Yi, Ti) for Yi, Ti in zip(Y, T)]).T
    if T_KNOTS:                                  # the heat the wall had to supply; the wall T is not solved
        Tw, qw = np.full(len(x), np.nan), np.gradient(h + u**2 / 2, x) * G * DIAMETER / 4.0
    else:
        Tw = np.array([profile(KNOTS, "temperature_K", xi) for xi in x])
        qw = htc * (Tw - T)
    W = 1.0 / np.sum(Y / chem.W, axis=1)
    R = ct.gas_constant / W
    gamma = cp / (cp - R)
    return dict(x_m=x, t_s=t, T_K=T, TW_K=Tw, QW_Wm2=qw, HTC_Wm2K=htc, RE=Re, F_DARCY=f,
                P_Pa=p, U_ms=u, RHO_kgm3=G / u, W_kgkmol=W, CP_JkgK=cp, GAMMA=gamma,
                MU_Pas=np.array([chem.mu(Yi, Ti) for Yi, Ti in zip(Y, T)]), MACH=u / np.sqrt(gamma * R * T),
                **chem.columns(Y)), Y[-1]


def loose_ylim(a, ys, rel=0.02, floor=0.0):
    """y axis at least +-rel of the value (and at least floor wide): a near-flat line stays flat, no 1e-13 offsets."""
    a.ticklabel_format(axis="y", useOffset=False)
    lo, hi = min(y.min() for y in ys), max(y.max() for y in ys)
    half = max(rel * max(abs(lo), abs(hi)), 0.5 * floor)
    if hi - lo < 2 * half:
        c = 0.5 * (lo + hi)
        a.set_ylim(c - half, c + half)


def main():
    out = OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    stem = CONFIG.stem
    print(f"config {CONFIG}")
    res, yld = {}, {}
    for m in MODELS:
        t0 = time.time()
        chem = Detailed() if m == "cantera" else Air() if m == "air" else Lumped(m)
        r, y_out = run(chem)
        res[m] = r
        if m != "air":
            yld[m] = chem.yields(y_out)
        csv = out / f"{stem}_{m}.csv"
        np.savetxt(csv, np.column_stack(list(r.values())), delimiter=",", header=",".join(r), comments="")
        chem = f" | CONV {r['CONV'][-1]:.4f} | C2H4 {r['Y_ethylene'][-1]:.4f}" if "CONV" in r else ""
        print(f"{m:8s} {time.time() - t0:6.1f} s | tau {r['t_s'][-1]*1e3:.2f} ms | T {r['T_K'][-1]:.1f} K | "
              f"p {r['P_Pa'][-1]/1e5:.4f} bar | u {r['U_ms'][-1]:.2f} m/s | M {r['MACH'][-1]:.3f} | "
              f"Re {r['RE'][0]:.3g}-{r['RE'][-1]:.3g} | htc {r['HTC_Wm2K'].mean():.0f} W/m2K{chem} -> {csv.name}")

    if yld:   # outlet yields by formula against the experiment: rows = the measured ones, then the first model's
        cols = {"experiment": EXPERIMENT, **yld} if EXPERIMENT else yld
        rows = list(EXPERIMENT) + [k for k in next(iter(yld.values())) if k not in EXPERIMENT]
        lines = [",".join(["formula", *cols])] + [
            ",".join([k, *(f"{c[k]:.4f}" if k in c else "" for c in cols.values())]) for k in rows]
        (out / f"{stem}_yields.csv").write_text("\n".join(lines) + "\n")
        print("outlet yields, wt% of the feed:")
        for ln in lines[:1 + (len(EXPERIMENT) or 12)]:
            print("  " + "".join(f"{v:>12s}" for v in ln.split(",")))
        print(f"yields -> {stem}_yields.csv")
        bars = [k for k in rows[:len(EXPERIMENT) or 12] if k != "C6H14"]    # the unconverted feed goes in the title
        fig, a = plt.subplots(figsize=(11, 4.6), facecolor="#fcfcfb")
        w = 0.8 / len(cols)
        for j, (name, c) in enumerate(cols.items()):
            a.bar(np.arange(len(bars)) + (j - (len(cols) - 1) / 2) * w, [c.get(k, np.nan) for k in bars], 0.88 * w,
                  color=SERIES[j], label=name)
        a.set_xticks(np.arange(len(bars)), bars)
        a.set_ylabel("outlet yield [wt% of the feed]", color="#52514e")
        a.set_title(f"{stem}: conversion " + ", ".join(f"{n} {100 - c['C6H14']:.1f} %" for n, c in cols.items()
                                                       if "C6H14" in c), loc="left", color="#0b0b0b")
        a.set_facecolor("#fcfcfb")
        a.grid(axis="y", color="#e1e0d9", lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color("#c3c2b7")
        a.tick_params(colors="#52514e", length=0)
        a.legend(frameon=False, labelcolor="#0b0b0b")
        fig.tight_layout()
        fig.savefig(out / f"{stem}_yields.png", dpi=130)
        print(f"plot -> {out / (stem + '_yields.png')}")

    panels = [("T_K", "T [K] (wall dotted)"), ("P_Pa", "p [Pa]"), ("U_ms", "u [m/s]"),
              ("W_kgkmol", "molar mass [kg/kmol]"), ("CP_JkgK", "cp [J/kg/K]"), ("GAMMA", "gamma [-]"),
              ("MU_Pas", "laminar viscosity [Pa s]"), ("RE", "Re [-]"),
              ("HTC_Wm2K", "htc [W/m2/K]"), ("QW_Wm2", "wall heat flux [W/m2]"), ("MACH", "Mach [-]"),
              ("CONV", "conversion [-]")]
    panels = [pn for pn in panels if all(pn[0] in res[m] for m in MODELS)]
    fig, ax = plt.subplots(3, 4, figsize=(19, 11))
    for a, (k, lab) in zip(ax.flat, panels):
        for m, ls in zip(MODELS, ("-", "--", "-.", ":")):
            a.plot(res[m]["t_s"] * 1e3, res[m][k], ls, label=m)
        a.set(xlabel="residence time [ms]", ylabel=lab)
        a.grid(alpha=0.3)
        loose_ylim(a, [res[m][k] for m in MODELS])
    for a in ax.flat[len(panels):]:
        a.axis("off")
    if HTC != 0:
        ax[0, 0].plot(res[MODELS[0]]["t_s"] * 1e3, res[MODELS[0]]["TW_K"], "k:", label="wall")
    ax[0, 0].legend(frameon=False)
    fig.suptitle(f"{stem}: T0 {T_0:.0f} K, p0 {P_0/1e5:.2f} bar, u0 {res[MODELS[0]]['U_ms'][0]:.0f} m/s, L {LENGTH} m, D {DIAMETER*1e3:.0f} mm, "
                 f"htc {HTC}, friction {'on' if FRICTION else 'off'}, {N_CELLS} cells")
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=130)
    print(f"plot -> {out / (stem + '.png')}")

    if not all("Y_feedstock" in res[m] for m in MODELS):
        return
    fig, ax = plt.subplots(3, 5, figsize=(20, 10))
    for a, k in zip(ax.flat, LUMPS):
        for m, ls in zip(MODELS, ("-", "--", "-.", ":")):
            if "Y_" + k in res[m]:                # a 12-lump model has no heavies (they are in its aromatics)
                a.plot(res[m]["x_m"], res[m]["Y_" + k], ls, label=m)
        a.set(xlabel="x [m]", title="Y " + k)
        a.grid(alpha=0.3)
        a.ticklabel_format(axis="y", useOffset=False)
        top = max(res[m]["Y_" + k].max() for m in MODELS if "Y_" + k in res[m])
        a.set_ylim(0.0, max(1.1 * top, 1e-3))     # mass fractions from 0, at least 0.001 tall
    for a in ax.flat[len(LUMPS):]:
        a.axis("off")
    ax[0, 0].legend(frameon=False)
    fig.suptitle("hybrid13 lumps (Cantera species summed into the RateNet lumps)")
    fig.tight_layout()
    fig.savefig(out / f"{stem}_lumps.png", dpi=130)
    print(f"plot -> {out / (stem + '_lumps.png')}")


if __name__ == "__main__":
    main()
