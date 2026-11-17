# plug-flow-reactor-1D — PFR reference for TBLOCK

A stand-alone 1-D plug-flow reactor for n-hexane steam cracking, or for air heated by the walls. It is the
reference for validating TBLOCK (`09 tblock-cfd/tblock`) on a straight duct: the reacting runs (`RATENET 1/2`)
and heated-wall air runs. It takes after
`GeneralizedPFR/Main_1_run_pfr.py` but keeps everything in one script and one config, with no imports from
other repos.

Nikolas Karefyllidis, PhD

## Run

```powershell
pip install -r requirements.txt
python pfr_runner.py                 # reads pfr_hexane.json (n-hexane)
python pfr_runner.py pfr_air.json    # air heated by the walls
python pfr_runner.py pfr_air_adiabatic.json   # the same pipe, unheated
python pfr_runner.py pfr_air_square.json      # heated square duct, 2 m (TBLOCK SQUARE_HEATED)
python pfr_runner.py pfr_vangeem2006.json     # n-hexane pilot-coil experiment (Van Geem et al. 2006)
python pfr_vangeem2006_profiles.py            # that experiment for five assumed temperature profiles (two figures)
```

The script writes these files to `outputs/`, named after the config file (`<config>`):

| File | Content |
|---|---|
| `<config>_<model>.csv` | one per model: `cantera` (species summed into the 13 lumps), `ratenet`, `air` |
| `<config>.png` | against residence time: T (and wall T), p, u, molar mass, cp, gamma, laminar viscosity, Re, htc, wall heat flux, Mach, conversion |
| `<config>_lumps.png` | all 13 lumps against x (hexane models only) |
| `<config>_yields.csv`, `.png` | outlet yields in wt% of the feed, by formula, one column per model and one for the experiment (hexane models only); the bar chart leaves out the unconverted feed and gives the conversions in its title |

CSV columns: `x_m t_s T_K TW_K QW_Wm2 HTC_Wm2K RE F_DARCY P_Pa U_ms RHO_kgm3 W_kgkmol CP_JkgK GAMMA MU_Pas
MACH`, then for hexane `CONV Y_feedstock ... Y_diluent`. `CONV` and the `Y_<lump>` names match TBLOCK's 1-D
file (`1DPavg4tec`).

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

A steady, constant-area tube of diameter D, or a square duct (`"cross_section": "square"`, 30 Oct 2026: D =
`diameter_m` is then the side, which is the hydraulic diameter 4A/P). The gas enters at `inlet_T_K` and is heated by a **prescribed
wall temperature Tw(x)** through a heat-transfer coefficient:

```
rho u = G                              continuity
d(p + G u)/dx = -f G u / (2 D)         momentum (f = 0 unless wall_friction)
G d(h + u^2/2)/dx = 4 q_w / D          energy (h includes the formation enthalpies)
q_w = htc (Tw - T)                     wall heat flux
u dY_k/dx = omega_k                    species
```

- `wall_htc_Wm2K` is a number, or `"gnielinski"`: Nu from the local Re and Pr (laminar, Re < 2300: Nu 3.66; square
  duct 2.98),
  htc = Nu k / D. It is fully developed, with no entrance-length or wall-to-bulk property correction.
  It needs a thermal conductivity, so `cantera` and `air` only.
- With `wall_htc_Wm2K` 0 the duct is adiabatic, and the hexane gas cools as it cracks.
- `wall_friction`: Darcy f from Petukhov, `(0.790 ln Re - 1.64)^-2` (laminar 64/Re; square duct 56.91/Re).
- A square duct uses the same equations on its hydraulic diameter: 4/D is its perimeter over its area, and the
  turbulent correlations take D_h as they are. Only the laminar constants and the mass flow (G a^2 instead of
  G pi D^2/4) differ, so a turbulent square duct gives the pipe's T, p and u.

The air cases (30 Oct 2026; 300 K, 2 bar, 50 m/s inlet, Gnielinski heat transfer and Petukhov friction):

| Config | Duct | Exit T | Exit p | Exit u | Mach | Residence | htc | TBLOCK deck |
|---|---|---|---|---|---|---|---|---|
| `pfr_air.json` | pipe D 30 mm, 3 m, 800 K wall | 604.1 K | 1.8513 bar | 108.77 m/s | 0.222 | 38.27 ms | 285 W/m2/K | `PIPE_HEATED.dat` |
| `pfr_air_adiabatic.json` | the same pipe, adiabatic (Fanno) | 299.9 K | 1.9523 bar | 51.21 m/s | 0.147 | 59.29 ms | 0 | `PIPE_ADIABATIC.dat` |
| `pfr_air_square.json` | square duct a 30 mm, 2 m, 800 K wall | 530.0 K | 1.9036 bar | 92.80 m/s | 0.202 | 28.35 ms | 278 W/m2/K | `SQUARE_HEATED.dat` |

Re falls from 1.86e5 to 1.13e5 along the heated pipe (1.24e5 in the shorter duct). The mass flow is 0.081758 kg/s in
the pipe and 0.104098 kg/s in the square duct. The 5 m pipe of 27 Oct 2026 (697.7 K, 1.739 bar) is
`outputs/pfr_air_air_5m.csv`.

Checks on the air model (2 m version of that case):

- At constant htc without friction, T(x) matches the closed form `Tw - (Tw - T0) exp(-4 htc x / (G D cp))`
  within 2 K. The closed form uses a mean cp, while the model uses cp(T).
- Without heating, the friction pressure drop is 4 % above Darcy-Weisbach, as it should be: the gas
  accelerates (Fanno flow).
- 1000 and 4000 cells agree to 0.03 K.

The march is cell by cell, following a Lagrangian parcel:

1. The chemistry runs at the cell-inlet T and p for `dt = dx/u`.
2. The wall heat is added to the total enthalpy.
3. T, p and u are solved from total enthalpy, momentum and the ideal-gas law (Newton on u, subsonic branch).

The script stops if the flow reaches Mach 1 (thermal or friction choking). The march is first order in dx: raise
`n_cells` until the result stops changing. On the isothermal test, 500 and 2000 cells agreed to 4 digits.

Laminar viscosity, used by Re, the correlations and the output:

- `cantera`, `air`: the mixture-averaged viscosity from the mechanism's transport data.
- `ratenet`: TBLOCK's law, `mu = tblock_mu288_Pas (T/288)^0.62` (deck `REYNO -1.256E-05`). The lumps carry
  no transport data.

### Pilot-coil mode: measured gas temperature and pressure (17 Nov 2026)

For validation against a pilot-plant coil, where the process-gas temperature and the pressure are measured
along the tube and the wall heat flux is not known:

- `gas_temperature_profile` imposes the gas T(x). It replaces the energy equation, so `wall_htc_Wm2K` and
  `wall_temperature_profile` are not needed. The knots have the wall profile's format.
- `pressure_profile` (knots with `pressure_Pa`) imposes p(x) as well and replaces the momentum equation; the
  velocity is then `u = G R T / p`. Without it, p comes from the momentum equation as before (with
  `wall_friction` if set).
- The inlet state is the first knot of each profile, so `inlet_T_K` (and `inlet_p_Pa` with a pressure profile)
  are not read.
- `mass_flow_kgs` (feed + steam) sets the inlet velocity in place of `inlet_u_ms`.
- `experiment_yields_wtpct`, e.g. `{"C2H4": 30.1, "C3H6": 17.2}`, is printed beside the models and written to
  `<config>_yields.csv`.

Yields are outlet mass per mass of n-hexane fed, in wt%, summed by formula: the mechanism names isomers by index
only (`C4H6(4)`, `C4H6(217)`, ...), so `C4H6` is all of them, and `C6H14` is the unconverted feed. `ratenet`
only has the lumps that are one molecule: C6H14, C2H4, C3H6, CH4 and H2.

In this mode `QW_Wm2` is the heat the wall had to supply, from the enthalpy balance
`G D/4 d(h + u^2/2)/dx`, and `TW_K` is empty (the wall temperature is not solved).

Checks, each imposing the profiles of an earlier run on the same grid:

- T(x) and p(x) of the default hexane case reproduce it: every column to 2e-15 with `ratenet`, to 7e-10 with
  `cantera`. The balance returns its adiabatic wall, |QW| below 1e-6 W/m2.
- T(x) alone reproduces the pressure through the momentum equation, to 1e-14, for the hexane case and for
  `pfr_air.json` with friction.
- For `pfr_air.json`, QW from the enthalpy balance matches `htc (Tw - T)` within 0.02 %.
- The existing configs give the same CSVs as before the change, to the last digit.

## Experimental validation case (`pfr_vangeem2006.json`, 17 Nov 2026)

n-Hexane steam cracking in the Ghent LPT pilot coil: Van Geem, Reyniers, Marin, Song, Matheu, Green, AIChE J.
52, 718-730 (2006). The journal paper is behind a paywall; the numbers here are from K. Van Geem's PhD thesis,
which carries the same work and is open (Ghent University 2006, `biblio.ugent.be/publication/470983`): Tables
2.5 and 2.6, and the pilot plant in section 4.5.

| | |
|---|---|
| Coil | reaction section 12.4 m, internal diameter 9 mm (section 4.5; chapter 2 says "d = 10 mm" in passing) |
| Instruments | 20 thermocouples in the process gas, 5 manometers along the coil, 7 furnace cells |
| The run (Table 2.6) | coil inlet 873 K, outlet 997 K, 0.27 to 0.24 MPa, n-hexane 4.0 kg/h, steam 0.4 kg/kg |
| Measured | conversion 51.5 %, yields of 12 products |
| The other nine runs | only as ranges (outlet 953-1075 K, 3.0-4.0 kg/h, conversion 25-75 %) and parity plots |

**The thermocouple readings are not printed.** The thesis simulated with the measured T(x) and p(x) imposed
(CHEMKIN PLUG) and reached 50.9 %, but gives only the inlet and outlet values. The config assumes straight
lines between them, and the conversion depends strongly on that assumption. With
`T = T_in + (T_out - T_in) (1 - exp(-a x/L)) / (1 - exp(-a))` (a = 0 is the straight line; at a = 8 the gas is
within 3 K of the outlet temperature from mid-coil on), `cantera`:

| Profile | Conversion | CH4 | C2H4 | C2H6 | C3H6 | C4H6 | C4H8 | C5H10 | H2 |
|---|---|---|---|---|---|---|---|---|---|
| a = 0 (the config) | 28.7 % | 3.62 | 9.34 | 1.32 | 7.07 | 0.47 | 4.09 | 2.00 | 0.14 |
| a = 4 | 44.7 % | 5.91 | 15.63 | 2.10 | 11.24 | 1.24 | 5.38 | 2.03 | 0.21 |
| a = 8 | 51.0 % | 6.87 | 18.32 | 2.40 | 12.90 | 1.67 | 5.68 | 1.88 | 0.24 |
| **Experiment** | **51.5 %** | 5.6 | 19.3 | 3.3 | 10.8 | 2.7 | 6.2 | 2.2 | 0.4 |

Yields in wt% of the n-hexane fed. C4H8 is 1-butene plus 2-butene; the measured C5H10 is 1-pentene only.
`pfr_vangeem2006_profiles.py` reruns this table (a = 0, 2, 4, 6, 8) and writes
`outputs/pfr_vangeem2006_profiles.png` (the profiles and the conversion along the coil) and
`outputs/pfr_vangeem2006_selectivity.png` (ethylene, propylene and methane against conversion, with the
measured points).

- The conversion is not validated: the mechanism reaches the measured 51.5 % only if the gas sits at the outlet
  temperature for most of the coil. Either the real profile is that flat (or peaks above the outlet value), or
  the mechanism is slow near 1000 K. The thermocouple readings would settle it.
- At matched conversion (a = 8) the product split can be compared: ethylene -5 %, propylene +19 %, methane
  +23 %, ethane -27 %, butadiene -38 %.
- A 10 mm tube gives 32.0 % instead of 28.7 % (straight line). 2000 cells give 28.70 % against 28.69 %.
- `v1.1` follows `cantera` here: 30.0 % against 28.7 %, ethylene 9.61 against 9.34. The older `ratenet`
  (epoch 40) gives 83 %: it is not usable at these temperatures and residence times (0.66 s).

### Gas models, same equations

| `models` | What it is |
|---|---|
| `cantera` | `mechanism/n-Hexane_Kinetic-Model_species_153.yaml`, the mechanism RateNet was trained on, integrated with Cantera (CVODES). Species are summed into the hybrid13 lumps for output. |
| `ratenet` | the RateNet closure as TBLOCK runs it: 13 hybrid13 lumps, rates from `models/ratenet/ratenet.txt`, lump gas from `lump_thermo.txt`. It applies the same clamps, zero-lump guard and sum-zero projection as `ratenet_tb_omega`. Any other name in `models` is read as a RateNet export folder under `models/`. |
| `v1.1` | ReactionRateNET v1.1 (`models/v1.1/`, checkpoint `da16424d`, epoch 250 of 250, `ratenet_export v2` with asinh targets): 12 lumps, the 13 without `heavies`, which is in its `aromatics`. Envelope 700-1861 K, 0.85-4 bar, steam ratio from 0.40. On the default hexane case: CONV 0.3354 against Cantera's 0.3338, Y_ethylene 0.0840 against 0.0839. |
| `air` | Cantera `air.yaml` (`air_composition`, default O2 0.21 / N2 0.79 by mole), chemistry frozen, NASA thermo and mixture-averaged transport. |

How to read the comparison:

- **TBLOCK vs `ratenet`** tests the CFD solver: same chemistry, so differences come from the flow solution.
- **`ratenet` vs `cantera`** tests the closure itself.

When `ratenet` loads, the network checks itself against the check vectors in `ratenet.txt` and stops if
they disagree.

## Config (`pfr_hexane.json`, `pfr_air*.json`)

| Key | Meaning |
|---|---|
| `models` | `"cantera"`, `"air"`, and RateNet export folders under `models/`, e.g. `["cantera", "ratenet"]` |
| `steam_ratio_kg_per_kg` | kg steam per kg n-hexane (RateNet trained at 0.40-0.60); hexane only |
| `air_composition` | mole fractions of the `air` model |
| `inlet_T_K`, `inlet_p_Pa`, `inlet_u_ms` | static inlet temperature, pressure and velocity |
| `mass_flow_kgs` | total mass flow, feed + steam [kg/s]; sets the inlet velocity in place of `inlet_u_ms` |
| `gas_temperature_profile` | imposed gas T(x), knots as `wall_temperature_profile`; replaces the energy equation (pilot-coil mode) |
| `pressure_profile` | imposed p(x), knots with `pressure_Pa`; needs `gas_temperature_profile`; replaces the momentum equation |
| `experiment_yields_wtpct` | measured outlet yields by formula, wt% of the feed, for the yields table |
| `length_m`, `diameter_m`, `n_cells` | tube length, diameter (the side of a square duct) and number of march cells |
| `cross_section` | `"circle"` (default) or `"square"` |
| `wall_htc_Wm2K` | gas-wall heat transfer coefficient: a number [W/m2/K] (0 = adiabatic, the default), or `"gnielinski"` |
| `wall_friction` | `true`: Darcy friction in the momentum equation (default `false`) |
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
- `models/v1.1/ratenet.txt` and `lump_thermo.txt`: copied on 17 Nov 2026 from
  `tblock-x-nn/tblock/input/models/ratenet/` (TBLOCK's default model since 16 Nov 2026). The `RateNet` class
  reads `ratenet_export` v1 and v2, as `tblock/scripts/chem/lump_of_phi.py` does, and takes the lump list from
  the file. The default `pfr_hexane.json` and the tables above still use `models/ratenet` (c321788e).
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
pfr_runner.py        the model
pfr_hexane.json      n-hexane input (the TBLOCK reacting validation case)
pfr_air.json         air input (heated wall, Gnielinski, friction)
pfr_air_adiabatic.json  the same air pipe unheated (adiabatic wall, friction: Fanno flow)
pfr_air_square.json  air in a heated square duct, side 30 mm, 2 m
pfr_vangeem2006.json n-hexane pilot-coil experiment (Van Geem et al. 2006), measured yields included
pfr_vangeem2006_profiles.py  that experiment for five assumed temperature profiles: the sensitivity figures
refit_lump_thermo.py optional lump-thermo refit for a future RateNet export
mechanism/           n-hexane mechanism (153 species)
models/ratenet/      RateNet network (checkpoint c321788e) + lump thermo
models/v1.1/         ReactionRateNET v1.1 (checkpoint da16424d, 12 lumps) + lump thermo
outputs/             results (git-ignored)
```
