# diffrl-flow-control

Code for the article

> J. Kang, I. Kim, S. Lee, *Sample-efficient reinforcement learning for active
> flow control via differentiable flow simulation*, submitted to Computer
> Physics Communications.

The repository contains a differentiable incompressible Navier–Stokes solver
written in JAX (`src/jax_cfd`), the two control benchmarks used in the article
(cylinder drag reduction at Re = 100–500 and Rayleigh–Bénard convection at
Ra = 10⁴–10⁶), the SHAC and PPO training code, the held-out evaluation
scripts, and the scripts that produce every table and figure of the article.

Trained policies, training histories and evaluation outputs are archived
separately on Zenodo (DOI: 10.5281/zenodo.XXXXXXX).

## Installation

Tested with Python 3.12, JAX/jaxlib 0.9.2 (CUDA 12) and NumPy 1.26 on NVIDIA
RTX 3090 and RTX 4090 GPUs.

```bash
git clone https://github.com/rider37/diffrl-flow-control.git
cd diffrl-flow-control
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # exact environment of the article
pip install -e .                     # installs the jax_cfd package
pytest                               # solver unit tests
```

For a CPU-only installation replace `jax[cuda12]` by `jax` in
`requirements.txt`.  All statistics and figures can be regenerated on a CPU;
training and policy evaluation need a GPU in practice.

## Data

Download the Zenodo archive and extract it in the repository root:

```bash
tar -xzf cpc_data_v1.0.tar.gz        # creates experiments/*/output_*/ and results/
```

The archive holds, for every (algorithm, seed, Re or Ra) run of the article,
the final policy checkpoint and the training history (`history.json`), plus
the evaluation outputs in `results/`.  `MANIFEST.tsv` in the archive lists
every file with its SHA-256 checksum.

## Layout

| Path | Content |
|---|---|
| `src/jax_cfd/` | Solver: MAC grid, RK3 time stepping, pressure projection, immersed boundary, Boussinesq buoyancy, control environments |
| `cases/`, `experiments/` | Benchmark configurations and SHAC/PPO training loops for the cylinder (`afc_cylinder`) and Rayleigh–Bénard (`rb_control`) problems |
| `configs/` | Configurations of the published SHAC runs, copied from their training histories |
| `scripts/` | Training launchers, held-out evaluation, statistics and verification scripts |
| `figures/` | Figure scripts; each writes its PDF into `figures/` |
| `tests/` | Unit tests |

Training seeds are 42, 123, 456, 517 and 1223; held-out initial conditions
use IC seeds 100–104 and are never used in training.

## Reproducing the article

All commands are run from the repository root.

### Training (GPU)

| Runs | Command |
|---|---|
| Cylinder PPO, Re = 100/200/400 | `python scripts/retrain_re100_ppo_768_eps075.py [seeds]` (likewise `re200`, `re400_ppo_1280`) |
| Cylinder PPO, Re = 500 | `python scripts/retrain_re500_ppo_velocity.py --seed 42` |
| Cylinder SHAC | `python scripts/train_wm_shac_cyl.py --Re 100 --seed 42` |
| RB PPO, Ra = 10⁴ | `python scripts/rb_ra1e4_pilot.py [seeds]` |
| RB PPO, Ra = 10⁵, 10⁶ | `python scripts/rb_ppo_expand.py --ra 1e5 --seed 42` |
| RB SHAC, Ra = 10⁴ | `python scripts/train_rb1e4_shac1000.py --seed 42` |
| RB SHAC, Ra = 10⁵, 10⁶ | `python scripts/train_wm_shac_rb.py --ra 1e5 --seed 42` |
| PPO with observation normalisation, Ra = 10⁶ (§5.4) | `python scripts/obsnorm_ppo_ra1e6_launch.py [seeds]` |

Launchers without an explicit seed run all five seeds.  Outputs go to
`experiments/<problem>/output_*/`, with the same directory names as in the
Zenodo archive.

### Held-out evaluation (GPU)

| Output (`results/`) | Script | Content |
|---|---|---|
| `cyl_heldout.json` | `scripts/eval_cyl_heldout.py` | controlled drag per (seed, IC), all cylinder policies |
| `rb_heldout.json` | `scripts/eval_rb_heldout.py` | controlled Nu per (seed, IC), all RB policies |
| `cyl_heldout_matched_baselines.json` | `scripts/cyl_matched_baseline.py` | uncontrolled drag per IC over the same window |
| `rb_heldout_matched_baselines.json` | `scripts/rb_matched_baseline.py` | uncontrolled Nu per IC over the same window |

Both evaluation scripts take `--Re`/`--ra`, `--algo` and `--seeds` to run a
subset.  They store controlled values only; every reduction in the article is
computed from these files by `scripts/heldout_matched_stats.py`.

### Tables and statistics (CPU, seconds)

| Item | Script | Output |
|---|---|---|
| Tables 2 and 4, Welch intervals, Fig. 9b data | `scripts/heldout_matched_stats.py` | `heldout_matched_stats.json`, `figure_data/fig9b_rb_reductions.json` |
| Uncontrolled variability windows (§5.2) | `scripts/uncontrolled_cv_windows.py` | `uncontrolled_cv_windows.json` |
| Table D.1, training cost (§4) | `scripts/time_to_target_wm.py` | `time_to_target_wm.json` |
| Table A.2, observation scales | `scripts/baseline_obs_stats.py` (GPU) | `baseline_obs_stats.json` |
| Table S1, force spectra | `scripts/cyl_drag_spectrum_stats.py` | `cyl_drag_spectrum_stats.json` |
| PPO pre-tanh saturation (§5.4) | `scripts/saturation_5seed.py` (GPU) | `saturation_5seed.json` |
| PPO with observation normalisation (§5.4) | `scripts/v12_obsnorm_ppo_ra1e6_final.py` (GPU) | `verification/v12_obsnorm_ppo_ra1e6_final.json` |
| Gradient variance and PPO critic (§5.2) | `scripts/grad_variance_mechanism_wm.py`, `grad_variance_ppo.py`, `grad_variance_horizon_wm.py` (GPU) | `grad_variance_wm/`, `grad_variance/` |
| AD–FD gradient check (Appendix F) | `scripts/analyze_gradient_quality.py`, `ad_fd_eps_sweep.py` (GPU) | `gradient_quality/` |

### Figures (CPU)

| Figure | Script | Input |
|---|---|---|
| 1, 2 | `figures/make_fig1_schematics.py` | `results/figure_data/fig{1,2}_background_*.npz` |
| 3, 4 | `figures/make_efficiency_split.py` | training histories |
| 5 | `figures/plot_fig_cyl_actuation.py` | `action_both_vs_re.npz` from `scripts/measure_action_both.py` |
| 6 | `figures/make_fig6_temp_grid.py` | `rb_timeavg_ra*.npz` from `scripts/render_rb_timeavg_wm.py` |
| 7 | `figures/replot_fig6_policy.py` | `policy_physics_wm/` from `scripts/analyze_policy_physics_wm.py` |
| 8 | `figures/replot_rb_spectrum_wm.py` | `fig7_rb_spectrum_data.npz` from `scripts/rb_spectrum_compare_wm.py` |
| 9 | `figures/fig_mechanism_wm.py` | `grad_variance_wm/`, `figure_data/fig9b_rb_reductions.json` |
| F.1 | `figures/make_figF8_gradient_quality.py` | `gradient_quality/` |

With the Zenodo data in place every figure script reproduces the published
figure exactly.

### Solver verification (GPU)

`scripts/cyl_re100_validation.py` gives the uncontrolled Re = 100 drag,
Strouhal number and jet momentum coefficient of Section 2.3, and
`scripts/validate_de_vahl_davis.py` compares the buoyancy solver with the de
Vahl Davis benchmark.  The scripts `scripts/v01_*.py` to `scripts/v12_*.py`
produce the numerical checks quoted in the article and the supplementary
material (single versus double precision, CFL and divergence, memory and
wall-clock time, convective onset, Lyapunov exponents, AD–FD gradients along a
trained trajectory, baseline sensitivity, horizon sweep, action power, and the
final-checkpoint observation-normalisation evaluation).  Their outputs are in
`results/verification/`.

## Notes

- `scripts/rb_eval_common.py` and `scripts/cyl_eval_common.py` hold the
  held-out initial conditions, checkpoint loaders and rollout used by all
  evaluation scripts.
- All cylinder runs use the immersed-boundary interface width
  `ibm_eps_factor = 0.75`, passed by the launchers and stored in
  `configs/cyl_re*.json`; `cases/cylinder_afc.make_env_config` falls back to
  the solver default when it is not given.

## License

MIT (see `LICENSE`).
