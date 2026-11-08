# fcs-mpcc-ev-ipmsm-benchmark

Switching-level simulation benchmark of current control for a low-inductance EV traction IPMSM
(Model 3-class rear-drive motor reconstructed from public data). Code and data for the manuscript

> T. H. Vo, *Current-control selection for EV traction inverters under a torque-ripple limit: finite-control-set predictive control versus scheduled-carrier PWM and DPWM over the WLTC* (submitted to *Results in Engineering*).

**Scope:** model-in-the-loop simulation only (no HIL, no hardware). Every number in the paper is
generated from the CSV files produced by this code. A Vietnamese description is in `README_vi.md`.

## Engineering problem
Choose the current-control and switching strategy of the traction inverter so that the torque ripple
stays below an NVH limit while the drive loss over WLTC Class 3b is minimal, subject to DC-link
voltage, current, torque-speed envelope (430 N m / 211 kW) and computation-delay constraints.

## What is compared
- PI-SVPWM, deadbeat-SVPWM (fixed carrier), PI with SVPWM / DPWM1 / DPWM-I and carrier frequency
  scheduled per operating point (VSF);
- FCS-MPCC with fixed weights, with neural-network weights (the original proposal), and with a
  switching-count penalty `lam_sw` (fixed, LUT- or NN-scheduled, optionally with scheduled `Ts`).

Plant: dq IPMSM with constant (saturated) Ld/Lq, two-level VSI with per-phase dead time, switching
and conduction loss model, one-period computation delay with compensation, sensor noise, MTPA +
field weakening, shared speed loop, vehicle road load. Stator iron loss with PWM harmonics
(time-domain Bertotti model driven by the switched voltage). Parameters carry provenance tags in
`configs/default.json`: `[MXP]` MotorXP public analysis, `[EVS]` public specification, `[DER]`
derived here, `[ASM]` assumption.

## Reproduce
```bash
pip install -r requirements.txt
python3 tests/test_sanity.py            # 41 physics / numerics checks
python3 scripts/reproduce.py            # all figures and tables: 15 741 runs, ~3 min on 2 cores
python3 scripts/reproduce.py --quick    # fast pipeline check
python3 scripts/derive_tesla_params.py  # Ld, Lq, psi_f, J from the public MotorXP data
```
Outputs: `results/figures/` (PDF + 300-dpi PNG), `results/tables/` (per-seed CSV and summary tables),
`results/training_*` (training grids, NN weights), `results/run_manifest.json` (every run: seed,
mismatch, Ts, status, versions). Raw waveform logs (`results/raw/`) are not versioned; the pipeline
regenerates them.

## Layout
```
configs/default.json      parameters with source tags
data/wltc_class3b.csv     WLTC Class 3b speed trace (from the JRC `wltp` package)
src/aifcs/core.py         numba closed loop: plant, inverter, all controllers
src/aifcs/params.py       config -> parameter vectors, road load, MTPA/FW operating points
src/aifcs/scenarios.py    test scenarios
src/aifcs/metrics.py      metric definitions
src/aifcs/training.py     NN weighting (as originally proposed) and lam_sw schedulers
src/aifcs/vsf.py          VSF / DPWM baselines and ripple-constrained selection
src/aifcs/ironloss.py     stator iron loss with PWM harmonics
src/aifcs/wltc.py         quasi-static WLTC energy evaluation
scripts/reproduce.py      full pipeline
tests/test_sanity.py      checks
```

## Limitations
Constant inductances (saturation handled only as a sensitivity study), assumed iron-loss coefficients
and stator geometry (harmonic iron loss likely an upper bound), no rotor/magnet loss, assumed inverter
loss data (sensitivity reported), quasi-static WLTC, computing cost of FCS-MPCC not equalised.

## AI assistance
The code was implemented with the assistance of a large-language-model assistant (Claude, Anthropic)
following the models and protocol specified by the author; results were checked by the author.
