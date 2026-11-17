#!/usr/bin/env python3
# Van Geem 2006 pilot run (pfr_vangeem2006.json): what the unknown shape of the gas temperature profile does.
#
# The thesis gives only the coil inlet and outlet temperature, not the 20 thermocouple readings. This runs the
# case with `cantera` for
#     T(x) = T_in + (T_out - T_in) (1 - exp(-a x/L)) / (1 - exp(-a)),    a = 0 (straight line), 2, 4, 6, 8
# and writes to outputs/:
#     pfr_vangeem2006_profiles.png      T(x) and the conversion along the coil, with the measured conversion
#     pfr_vangeem2006_selectivity.png   ethylene, propylene and methane yield against conversion, with the measured
#
#     python pfr_vangeem2006_profiles.py        (five Cantera runs side by side, about a minute)

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
A = (0, 2, 4, 6, 8)
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")        # one hue, darker with a
SPECIES = {"C2H4": ("ethylene", "#2a78d6"), "C3H6": ("propylene", "#eb6834"), "CH4": ("methane", "#1baf7a")}
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"

base = json.load(open(ROOT / "pfr_vangeem2006.json"))
T_in, T_out = (base["gas_temperature_profile"][i]["temperature_K"] for i in (0, -1))
exp = base["experiment_yields_wtpct"]
conv_exp = 100.0 - exp["C6H14"]

res = {}
with tempfile.TemporaryDirectory() as tmp:
    runs = []
    for a in A:
        xi = np.linspace(0.0, 1.0, 41)
        shape = xi if a == 0 else (1.0 - np.exp(-a * xi)) / (1.0 - np.exp(-a))
        cfg = dict(base, models=["cantera"], output_dir=tmp, gas_temperature_profile=[
            {"position": float(x), "temperature_K": float(T_in + (T_out - T_in) * s)} for x, s in zip(xi, shape)])
        json.dump(cfg, open(f"{tmp}/a{a}.json", "w"))
        runs.append(subprocess.Popen([sys.executable, str(ROOT / "pfr_runner.py"), f"{tmp}/a{a}.json"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True))
    for a, p in zip(A, runs):
        err = p.communicate()[1]
        assert p.returncode == 0, err[-2000:]
        res[a] = np.genfromtxt(f"{tmp}/a{a}_cantera.csv", delimiter=",", names=True)

feed = res[A[0]]["Y_feedstock"][0]
print(f"measured: conversion {conv_exp:.1f} %, " + ", ".join(f"{k} {exp[k]}" for k in SPECIES))
for a in A:
    r = res[a]
    print(f"a = {a}: conversion {100 * r['CONV'][-1]:.1f} %, "
          + ", ".join(f"{k} {100 * r['Y_' + lump][-1] / feed:.2f}" for k, (lump, _) in SPECIES.items()))


def clean(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, length=0)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)


out = ROOT / base["output_dir"]
out.mkdir(parents=True, exist_ok=True)
L = base["length_m"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)
for a, c in zip(A, RAMP):
    r = res[a]
    a1.plot(r["x_m"], r["T_K"], color=c, lw=2,
            label=f"a = {a}{' (straight line, the config)' if a == 0 else ''}: {100 * r['CONV'][-1]:.1f} %")
    a2.plot(r["x_m"], 100 * r["CONV"], color=c, lw=2)
a2.plot(L, conv_exp, "o", ms=10, color=INK, mec=SURFACE, mew=2, clip_on=False, zorder=5)
a2.annotate(f"measured {conv_exp:.1f} %", (L, conv_exp), xytext=(-10, 10), textcoords="offset points",
            ha="right", va="bottom", color=INK)
a2.set_ylim(0, 60)
clean(a1, "x [m]", "gas temperature [K]")
clean(a2, "x [m]", "n-hexane conversion [%]")
a1.set_title("Assumed temperature profiles", loc="left", color=INK)
a2.set_title("Conversion along the coil", loc="left", color=INK)
a1.legend(frameon=False, labelcolor=INK, loc="lower right", title="profile: outlet conversion")
fig.suptitle(f"Van Geem 2006 pilot run, cantera: only the inlet ({T_in:.0f} K) and outlet ({T_out:.0f} K) "
             "temperatures are published", x=0.01, ha="left", color=INK)
fig.tight_layout()
fig.savefig(out / "pfr_vangeem2006_profiles.png", dpi=130)

fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
for k, (lump, c) in SPECIES.items():
    for a, ls in ((A[-1], "-"), (A[0], "--")):
        r = res[a]
        ax.plot(100 * r["CONV"], 100 * r["Y_" + lump] / feed, ls, color=c, lw=2)
    ax.plot(conv_exp, exp[k], "o", ms=10, color=c, mec=SURFACE, mew=2, zorder=5)
    end = res[A[-1]]
    ax.annotate(lump, (100 * end["CONV"][-1], 100 * end["Y_" + lump][-1] / feed), xytext=(14, 0),
                textcoords="offset points", va="center", color=INK)
ax.plot([], [], "-", color=INK2, lw=2, label=f"model, a = {A[-1]} (fast heat-up)")
ax.plot([], [], "--", color=INK2, lw=2, label=f"model, a = {A[0]} (straight line)")
ax.plot([], [], "o", ms=10, color=INK2, mec=SURFACE, mew=2, label="measured (thesis Table 2.6)")
ax.set_xlim(0, 62)
ax.set_ylim(0, None)
clean(ax, "n-hexane conversion [%]", "yield [wt% of the feed]")
ax.set_title("Yields against conversion: the product split barely depends on the profile", loc="left", color=INK)
ax.legend(frameon=False, labelcolor=INK, loc="upper left")
fig.tight_layout()
fig.savefig(out / "pfr_vangeem2006_selectivity.png", dpi=130)
print(f"plots -> {out / 'pfr_vangeem2006_profiles.png'}, {out / 'pfr_vangeem2006_selectivity.png'}")
