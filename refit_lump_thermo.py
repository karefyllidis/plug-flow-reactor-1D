#!/usr/bin/env python3
# Refit the single-species lumps of a RateNet lump_thermo.txt to the mechanism's exact species thermo.
# N. Karefyllidis 2026
#
# The export fits one NASA-7 range per lump over t_range. For a lump that is one species (composition
# block share >= 0.999: feedstock C6H14, ethylene, hydrogen, steam) the exact Cantera species is known,
# so its polynomial is refitted to it (least squares on cp/R and h/RT over t_range, same format). The
# steam fit of the export is 2.2 % off in cp at 950 K; the refit is ~0.1 %. A refit is kept only where
# it beats the old fit. The 64 check vectors (mixture cp, h) are recomputed so TBLOCK's
# thermo_selfcheck (1e-10) still passes; W is unchanged, so RateNet's inputs are untouched.
#
#     python refit_lump_thermo.py in.txt out.txt
# Optional, NOT applied: models/ratenet/ stays a byte copy of TBLOCK's files so the `ratenet` PFR is
# TBLOCK's gas. Use it on a new RateNet export, for TBLOCK and the 2-D repo together; it changes h, so
# restart dumps of the old gas start fresh.

import sys
from pathlib import Path

import cantera as ct
import numpy as np

_ROOT = Path(__file__).resolve().parent
SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
MECH = _ROOT / "mechanism" / "n-Hexane_Kinetic-Model_species_153.yaml"


def cp_h(a, T):
    """cp/R and h/RT of NASA-7 coefficients a1..a6."""
    return (a[0] + a[1] * T + a[2] * T**2 + a[3] * T**3 + a[4] * T**4,
            a[0] + a[1] * T / 2 + a[2] * T**2 / 3 + a[3] * T**3 / 4 + a[4] * T**4 / 5 + a[5] / T)


def fit(thermo, T):
    cpR = np.array([thermo.cp(t) for t in T]) / ct.gas_constant
    hRT = np.array([thermo.h(t) for t in T]) / (ct.gas_constant * T)
    A = np.vstack([np.column_stack([T**0, T, T**2, T**3, T**4, 0 * T]),
                   np.column_stack([T**0, T / 2, T**2 / 3, T**3 / 4, T**4 / 5, 1 / T])])
    s = 1.0 / np.array([1.0, 1e3, 1e6, 1e9, 1e12, 1e-3])          # column scaling for conditioning
    b, *_ = np.linalg.lstsq(A * s, np.r_[cpR, hRT], rcond=None)
    return b * s, cpR, hRT


def mix_cp_h(a, W, T, Y):
    """Mixture cp [J/kg/K] and h [J/kg] as TBLOCK's lump_si (T inside t_range)."""
    R = ct.gas_constant / W
    cp, h = cp_h(a.T, T)
    return Y @ (R * cp), Y @ (R * T * h)


def main():
    lines = SRC.read_text().split("\n")
    tlo, thi = map(float, next(l for l in lines if l.startswith("t_range ")).split()[1:3])
    ilump = [i for i, l in enumerate(lines) if l.strip().startswith("Y_lump_chem_") and len(l.split()) == 8]
    names = [lines[i].split()[0] for i in ilump]
    W = np.array([float(lines[i].split()[1]) for i in ilump])
    a_old = np.array([[float(v) for v in lines[i].split()[2:8]] for i in ilump])
    icv = next(i for i, l in enumerate(lines) if l.startswith("check_vectors "))
    ncv = int(lines[icv].split()[1])
    cv = np.array([[float(v) for v in lines[icv + 1 + j].split()] for j in range(ncv)])
    comp = {l.split()[0]: l.split()[1:] for l in lines[lines.index("composition") + 1:] if l.strip()}

    # the stored check vectors must be what the old polynomials give (proves cp_h/mix_cp_h match TBLOCK)
    for row in cv:
        cp, h = mix_cp_h(a_old, W, row[0], row[1:14])
        assert abs(cp / row[14] - 1) < 1e-9 and abs(h - row[15]) / (cp * row[0]) < 1e-9, "check vector mismatch"

    gas = ct.Solution(str(MECH))
    T = np.linspace(tlo, thi, 600)
    a_new = a_old.copy()
    for k, name in enumerate(names):
        sp, share = comp[name][0].split(":")
        if float(share) < 0.999 or len(comp[name]) > 1:
            continue
        thermo = gas.species(gas.species_index(sp)).thermo
        a, cpR, hRT = fit(thermo, T)
        err = {}
        for tag, aa in (("old", a_old[k]), ("new", a)):
            cp, h = cp_h(aa, T)
            err[tag] = (np.abs(cp / cpR - 1).max() * 100, np.abs((h - hRT) * T / cpR).max())
        keep = err["new"][0] < err["old"][0]
        if keep:
            a_new[k] = a
        print(f"{name:28s} = {sp:9s} cp max err {err['old'][0]:5.2f}% -> {err['new'][0]:5.2f}%   "
              f"h max err {err['old'][1]:6.2f} K -> {err['new'][1]:5.2f} K   {'refit' if keep else 'kept'}")

    for i, k in zip(ilump, range(len(names))):
        if (a_new[k] != a_old[k]).any():                 # W text kept as exported
            lines[i] = "  " + " ".join([names[k], lines[i].split()[1]] + [repr(float(v)) for v in a_new[k]])
    for j, row in enumerate(cv):
        row[14], row[15] = mix_cp_h(a_new, W, row[0], row[1:14])
        lines[icv + 1 + j] = " ".join(repr(float(v)) for v in row)
    DST.write_text("\n".join(lines), newline="\n")

    # self-check as TBLOCK's thermo_selfcheck on the written file
    out = DST.read_text().split("\n")
    a_chk = np.array([[float(v) for v in out[i].split()[2:8]] for i in ilump])
    for j in range(ncv):
        row = np.array(out[icv + 1 + j].split(), float)
        cp, h = mix_cp_h(a_chk, W, row[0], row[1:14])
        assert abs(cp / row[14] - 1) < 1e-10 and abs(h - row[15]) / (cp * row[0]) < 1e-10, "thermo_selfcheck would fail"
    print(f"wrote {DST}  (t_range {tlo:.0f}-{thi:.0f} K, {ncv} check vectors recomputed, self-check ok)")


if __name__ == "__main__":
    main()
