# diffrl-flow-control

Code for *Sample-efficient reinforcement learning for active flow control with a differentiable Navier–Stokes solver* (Computer Physics Communications, submitted).

It contains a differentiable incompressible Navier–Stokes solver written in JAX and two active-flow-control problems. Each problem is solved with SHAC (policy gradients through the solver) and with PPO, and SHAC is also compared with full-episode backpropagation through time (BPTT) and gradient-informed PPO (GI-PPO):

- drag reduction of a cylinder with synthetic jets, Re = 100, 200, 400, 500;
- control of Rayleigh–Bénard convection through the bottom-wall temperature, Ra = 1e4, 1e5, 1e6.

The repository holds only the code used in the paper. Trained policies, training histories and all evaluated results are in the Zenodo record [10.5281/zenodo.23051277](https://doi.org/10.5281/zenodo.23051277).

## Layout

```
src/diffrl_flow_control/
  solver/             MAC-grid incompressible solver (RK3, MUSCL advection, DCT/FFT pressure,
                      ghost-cell immersed boundary, Boussinesq temperature)
  cylinder/           environment, SHAC, PPO, held-out initial conditions
  rayleigh_benard/    environment, SHAC, PPO, held-out initial conditions
  baselines/          full-episode BPTT and GI-PPO (Section 4.4)
  configs.py          loads configs/<problem>/<name>.json
configs/
  cylinder/           env_re*.json, shac_re*.json, ppo_re*.json
  rayleigh_benard/    env_ra*.json, shac_ra*.json, ppo_ra*.json,
                      ppo_obsnorm_ra1e6.json, shac_horizon{20,40,80,160}_ra1e4.json
scripts/
  train.py            training entry point
  evaluate/           held-out evaluation of the trained policies (Tables 2, 4)
  analysis/           statistics, gradient diagnostics and figure data
  checks/             solver validation and verification (Section 2.3, appendices, supplement)
tests/                solver and environment tests
```

## Installation

Python 3.11 or newer. The SHAC and PPO runs of the paper were trained with JAX 0.9.1; the other runs, the evaluations and the analyses use Python 3.12 and JAX 0.9.2, all with CUDA 12 on NVIDIA GPUs.

```bash
pip install -r requirements.txt
pip install -e .
pytest            # 16 tests, CPU is enough
```

## Data

Download the data archive from the Zenodo record and extract it at the repository root. This creates

```
runs/<problem>/<algorithm>_<case>_s<seed>/   history.json and checkpoints/final.npz
                        (also best.npz for the SHAC horizon sweep and the GI-PPO
                        checkpoint within SHAC's CFD budget, epoch_0010.npz or epoch_0020.npz)
results/evaluation/     held-out evaluations
results/statistics/     statistics reported in the tables and text
results/diagnostics/    gradient-variance and AD–FD diagnostics
results/checks/         validation and verification outputs
results/figure_data/    arrays plotted in the figures
```

## Training

```bash
python scripts/train.py <problem> <algorithm> <case> --seed <seed>
```

| argument | values |
|---|---|
| problem | `cylinder`, `rayleigh_benard` |
| algorithm | `shac`, `ppo`, `bptt`, `gippo`; for Rayleigh–Bénard also `ppo_obsnorm` (Ra = 1e6) and `shac_horizon20/40/80/160` (Ra = 1e4) |
| `--episode-actions N` | `shac`, `bptt`: training episode of N actions (Fig. 6, Appendix G) |
| `--updates U` | `bptt`: U updates instead of SHAC's CFD budget (Fig. 6, Appendix G) |
| case | `re100`, `re200`, `re400`, `re500`; `ra1e4`, `ra1e5`, `ra1e6` |
| seed | 42, 123, 456, 517, 1223 were used in the paper |

Example: `python scripts/train.py rayleigh_benard shac ra1e5 --seed 42` writes `runs/rayleigh_benard/shac_ra1e5_s42/`, and `python scripts/train.py rayleigh_benard bptt ra1e5 --seed 42 --episode-actions 480 --updates 120` writes `runs/rayleigh_benard/bptt_ep480_u120_ra1e5_s42/`. XLA GPU autotuning is switched off in `train.py`, as in every run of the paper.

## Reproducing the paper

| paper item | script | output |
|---|---|---|
| Table 2, cylinder held-out | `scripts/evaluate/cylinder_heldout.py`, `cylinder_uncontrolled.py` | `results/evaluation/cylinder_*.json` |
| Table 4, RB held-out | `scripts/evaluate/rayleigh_benard_heldout.py`, `rayleigh_benard_uncontrolled.py` | `results/evaluation/rayleigh_benard_*.json` |
| Tables 2 and 4, statistics, Fig. 11b | `scripts/analysis/heldout_statistics.py`, `uncontrolled_variability.py` | `results/statistics/`, `results/figure_data/heldout_reductions_per_seed.json` |
| Section 5.4 and Table A.2, observation-normalised PPO (Ra = 1e6) | `scripts/evaluate/rayleigh_benard_heldout_obsnorm.py`, `scripts/analysis/observation_scales.py` | `results/evaluation/`, `results/statistics/` |
| Section 5.1, cylinder forces | `scripts/analysis/cylinder_force_series.py`, `cylinder_force_spectra.py` | `results/statistics/` |
| Section 5.2 and Fig. 11a,c, gradient variance | `scripts/analysis/gradient_variance.py`, `gradient_variance_horizon.py`, `ppo_gradient_variance.py` | `results/diagnostics/` |
| Section 4.4, Figs. 5–6 and Appendix G, BPTT and GI-PPO | `scripts/evaluate/*_heldout.py --algo bptt gippo --seeds 42`, GI-PPO within SHAC's CFD budget with `--checkpoint epoch_0010` or `epoch_0020`, and the training-episode runs (see the docstring of `differentiable_methods.py`), then `scripts/analysis/differentiable_methods.py`; training times on the RTX 3090 (Figs. 5c,d and 6b) are measured values, with their definition, in the data | `results/statistics/differentiable_methods.json`, `results/checks/training_time_rtx3090.json` |
| Section 5.4, PPO saturation | `scripts/analysis/ppo_saturation.py` | `results/statistics/ppo_saturation.json` |
| Section 5.4, training action interval | `scripts/checks/training_interval_ra1e6.py` | `results/checks/training_interval_ra1e6.json` |
| Appendix D, time to target | `scripts/analysis/training_cost.py` | `results/statistics/training_cost.json` |
| Appendix F, AD–FD | `scripts/analysis/ad_fd_gradients.py`, `ad_fd_step_sweep.py` | `results/diagnostics/ad_fd_*.json` |
| validation and verification checks (Section 2.3, Sections 5.1–5.4, Appendix C, Supplementary Material) | `scripts/checks/*.py` | `results/checks/` |
| Figs. 1–2 background | flow snapshots saved by `scripts/train.py` | `results/figure_data/cylinder_schematic_background.npz`, `rayleigh_benard_schematic_background.npz` |
| Figs. 7–10 data | `scripts/analysis/cylinder_actuation.py`, `rayleigh_benard_fields.py`, `wall_pattern.py`, `rayleigh_benard_heat_flux.py` | `results/figure_data/` |

The scripts in `scripts/` read `runs/` and `results/` and write to `results/`, so each item can be reproduced on its own from the downloaded data, which also holds the arrays plotted in every figure.

## Citation

See `CITATION.cff`.

## License

MIT
