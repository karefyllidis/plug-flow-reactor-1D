# plug-flow-reactor-1D — n-hexane PFR reference for TBLOCK

A stand-alone 1-D plug-flow reactor for n-hexane steam cracking. It is the reference for validating the
reacting TBLOCK runs (`09 tblock-cfd/tblock`, `RATENET 1/2`) on a straight duct. It takes after
`GeneralizedPFR/Main_1_run_pfr.py` but keeps everything in one script and one config, with no imports from
other repos.

Nikolas Karefyllidis, PhD

## Run

```powershell
pip install -r requirements.txt
python pfr_hexane.py                 # reads pfr_hexane.json
python pfr_hexane.py my_case.json    # another config
```

The script writes these files to `outputs/`:

| File | Content |
|---|---|
| `pfr_hexane_cantera.csv` | detailed mechanism, species summed into the 13 lumps |
| `pfr_hexane_ratenet.csv` | the RateNet closure (`models/ratenet/`) |
| `pfr_hexane.png` | against residence time: T (and wall T), p, u, molar mass, cp, gamma, laminar viscosity, conversion, wall heat flux |
| `pfr_hexane_lumps.png` | all 13 lumps against x, both models |

CSV columns: `x_m t_s T_K TW_K QW_Wm2 P_Pa U_ms RHO_kgm3 W_kgkmol CP_JkgK GAMMA MU_Pas MACH CONV
Y_feedstock ... Y_diluent`. `CONV` and the `Y_<lump>` names match TBLOCK's 1-D file (`1DPavg4tec`).

## TBLOCK validation case (the default `pfr_hexane.json`)

For RateNet checkpoint `c321788e` (`models/ratenet/`):

| | |
|---|---|
| Walls | adiabatic (`wall_htc_Wm2K` 0): TBLOCK has no isothermal walls with the lump gas |
| Inlet | static 1150 K, 2.000 bar, 100 m/s, steam 0.5 kg/kg (Y_feed 0.666667) |
| TBLOCK inlet (lump gas, isentropic) | T0 1151.42 K, p0 204 023 Pa |
| Exit static p (RateNet PFR) | 199 454 Pa |
| Duct | 3.0 m, residence time 28.4 ms, Mach 0.19-0.20 |
| Outlet, Cantera / RateNet | CONV 0.334 / 0.341, T 1012.4 / 1012.7 K, Y_ethylene 0.0839 / 0.0863 |
| Along the duct | max \|dCONV\| 0.013, max \|dT\| 4.8 K |

Every RateNet input stays inside its training box (TBLOCK's `PINN_EXTRAP` stays 0), and the feed rate is
never clamped.

Outlet lumps, RateNet against Cantera:

| Lump | Cantera | RateNet | Error |
|---|---|---|---|
| feedstock | 0.4441 | 0.4393 | -1 % |
| ethylene | 0.0839 | 0.0863 | +3 % |
| propylene | 0.0513 | 0.0466 | -9 % |
| methane | 0.0255 | 0.0281 | +10 % |
| butenes | 0.0304 | 0.0326 | +7 % |
| heavy olefins | 0.0181 | 0.0211 | +17 % |
| dienes | 0.0044 | 0.0036 | -19 % |
| hydrogen | 0.0018 | 0.0021 | +16 % |

### How the case was chosen

The sweep was adiabatic, 100 m/s, scored against Cantera over the whole window:

| Inlet T, p | Window | Max \|dCONV\| | Ethylene | Propylene | Worst major lump |
|---|---|---|---|---|---|
| 1050 K, 2 bar | 30 ms | 0.028 | +20 % | 12 % | methane 34 % |
| 1100 K, 2 bar | 30 ms | 0.027 | +16 % | 5 % | hydrogen 31 % |
| **1150 K, 2 bar** | **30 ms** | **0.013** | **3 %** | **9 %** | dienes 19 % |
| 1150 K, 3 bar | 30 ms | 0.013 | 5 % | 12 % | hydrogen 23 % |
| 1150 K, 1.5 bar | 30 ms | 0.012 | 2 % | 8 % | heavy olefins 22 % |
| 1200 K, 1.5 bar | 30 ms | 0.078 | 0 % | 14 % | propylene 14 %, dT 30 K early |

Pressure has little effect. Below 1150 K the network drifts over time, and above it the first millisecond
is too fast.

### Why the match is not perfect here

Inside the training campaign, RateNet's own evaluation matches Cantera almost perfectly. Those test runs
are heated-wall runs. This duct is adiabatic, so the gas cools as it cracks, which is a different part of
state space.

Fed Cantera's own adiabatic states (teacher-forced, as in the RateNet evaluation), the network's lump rates
are 0.5-2.3 times Cantera's. That is the network itself, not this code: the raw output and the output after
TBLOCK's clamps and sum-zero projection differ by a few percent at most.

The checkpoint is at epoch 40 of 250. Re-export when training finishes and rerun the sweep.

TBLOCK against the `ratenet` CSV tests the solver. That comparison holds at any length, because both run the
same closure. Against the `cantera` CSV it also tests the chemistry, to the accuracy in the tables above.

## Model

A steady, constant-area, inviscid tube of diameter D. The gas enters at `inlet_T_K` and is heated by a
**prescribed wall temperature Tw(x)** through a heat-transfer coefficient:

```
rho u = G                              continuity
p + G u = C                            momentum
G d(h + u^2/2)/dx = 4 q_w / D          energy (h includes the formation enthalpies)
q_w = htc (Tw - T)                     wall heat flux
u dY_k/dx = omega_k                    species
```

With `wall_htc_Wm2K` 0 the duct is adiabatic: the gas cools as it cracks.

The march is cell by cell, following a Lagrangian parcel:

1. The chemistry runs at the cell-inlet T and p for `dt = dx/u`.
2. The wall heat is added to the total enthalpy.
3. T, p and u are solved from total enthalpy, momentum and the ideal-gas law (Newton on u, subsonic branch).

The script stops if the flow reaches Mach 1 (thermal choking). The march is first order in dx: raise
`n_cells` until the result stops changing. On the isothermal test, 500 and 2000 cells agreed to 4 digits.

Laminar viscosity (output only; the inviscid model does not use it):

- `cantera`: the mixture-averaged viscosity from the mechanism's transport data.
- `ratenet`: TBLOCK's law, `mu = tblock_mu288_Pas (T/288)^0.62` (deck `REYNO -1.256E-05`). The lumps carry
  no transport data.

### Two chemistries, same equations

| `models` | What it is |
|---|---|
| `cantera` | `mechanism/n-Hexane_Kinetic-Model_species_153.yaml`, the mechanism RateNet was trained on, integrated with Cantera (CVODES). Species are summed into the hybrid13 lumps for output. |
| `ratenet` | the RateNet closure as TBLOCK runs it: 13 hybrid13 lumps, rates from `models/ratenet/ratenet.txt`, lump gas from `lump_thermo.txt`. It applies the same clamps, zero-lump guard and sum-zero projection as `ratenet_tb_omega`. Any other name in `models` is read as a RateNet export folder under `models/`. |

How to read the comparison:

- **TBLOCK vs `ratenet`** tests the CFD solver: same chemistry, so differences come from the flow solution.
- **`ratenet` vs `cantera`** tests the closure itself.

When `ratenet` loads, the network checks itself against the check vectors in `ratenet.txt` and stops if
they disagree.

## Config (`pfr_hexane.json`)

| Key | Meaning |
|---|---|
| `models` | `"cantera"` and RateNet export folders under `models/`, e.g. `["cantera", "ratenet"]` |
| `steam_ratio_kg_per_kg` | kg steam per kg n-hexane (RateNet trained at 0.40-0.60) |
| `inlet_T_K`, `inlet_p_Pa`, `inlet_u_ms` | static inlet temperature, pressure and velocity |
| `length_m`, `diameter_m`, `n_cells` | tube length, diameter and number of march cells |
| `wall_htc_Wm2K` | gas-wall heat transfer coefficient, constant along the tube (0 = adiabatic) |
| `wall_temperature_profile` | Tw(x) knots in Main_1's format: `position` relative 0..1, `temperature_K`, `interpolation_to_next` `linear` or `step` |
| `tblock_mu288_Pas` | TBLOCK's laminar viscosity at 288 K, used for the `ratenet` viscosity output |

RateNet's training envelope (checkpoint c321788e) is T up to 1601 K, p 1-4 bar, rho 0.16-19 kg/m3, steam ratio
0.40-0.60; the dense part is T 495-1182 K (p01-p99, `07 RateNet/docs/data.md`). Outside the envelope, inputs are
clamped as in TBLOCK, so results there are extrapolation.

## Keeping in sync

These files are copies. Re-copy them when the source changes:

- `models/ratenet/ratenet.txt`: checkpoint `c321788e` (campaign `20261005_173100`, epoch 40 of 250) from
  `07 RateNet/RateNet/models/RateNet/`, exported with
  `python scripts/export_fortran_weights.py --no-thermo -o <here>/models/ratenet/ratenet.txt`.
  Re-export when training finishes.
- `models/ratenet/lump_thermo.txt`: TBLOCK's, from the previous export. The thermo only depends on the lumps and
  the mechanism, which are unchanged. The matching export needs the training pickle, which is not on this machine.
- The `RateNet` class comes from `tblock/scripts/chem/lump_of_phi.py`.
- `classify_species_hybrid13` comes from `GeneralizedPFR/src/ml/species_lumping.py`.

**TBLOCK still runs the previous checkpoint** (`bd35ff68`, heated-tube campaign `20260830_134749`). Copy
`models/ratenet/ratenet.txt` to `tblock/input/models/ratenet/` before comparing TBLOCK with this PFR.

`refit_lump_thermo.py in.txt out.txt` is an optional tool, not applied. It refits the single-species lumps
(hexane, ethylene, hydrogen, steam) to Cantera over `t_range`, which brings the inlet cp error from 0.35 % to
0.1 %. It also recomputes the check vectors, so TBLOCK's `thermo_selfcheck` still passes. Apply it to TBLOCK
and the 2-D repo together.

## Layout

```
pfr_hexane.py        the model
pfr_hexane.json      the input (the TBLOCK validation case)
refit_lump_thermo.py optional lump-thermo refit for a future RateNet export
mechanism/           n-hexane mechanism (153 species)
models/ratenet/      RateNet network (checkpoint c321788e) + lump thermo
outputs/             results (git-ignored)
```
