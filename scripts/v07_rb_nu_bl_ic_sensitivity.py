"""Verification copy of scripts/measure_nu_bl_ic_sensitivity.py; saves results/verification/v07_rb_nu_bl_ic_sensitivity.json.
Run from the repository root."""
"""RB Ra=10^4 baseline Nu sensitivity to initial conditions.

Puzzle: same Shenfun solver, same domain (2π×2), same 96×64 grid, same Pr=0.7.
  - Vignon 2023: Nu_bl = 2.68
  - Markmann 2025: Nu_bl = 3.9 (also reports max "around ~4" for normalization)
  - Ours (MAC FD): Nu_bl = 4.31

Hypothesis: Nu_bl depends strongly on IC (mode mix, amplitude, phase) because
the 2-cell attractor at Ra=10^4 has multiple metastable states with different
plume amplitudes. Markmann averages 20 test ICs; our default uses one fixed IC.

This script measures Nu_bl for 10 different random ICs at Ra=10^4 (our grid).
Reports:
  - Nu_mean (time-averaged over 3000 steps = 30 time-units, after 3000-step warmup)
  - Nu_max (transient peak)
  - Nu_late (last 1000 steps — converged attractor)
  - distribution across ICs
"""
import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiments.rb_control.config import ENV_CONFIG
from jax_cfd.envs.rb_control_env import RBControlEnv, FluidState


def build_ic(env, seed):
    """Random linear T + multi-mode perturbation with random phase.

    Follows env._initial_state but with random mode mix instead of fixed k=1,2,3.
    """
    rng = np.random.default_rng(seed)
    grid = env.grid
    Nx, Ny = grid.shape
    Lx, Ly = grid.domain_size

    u = jnp.zeros(grid.face_shape(0))
    v = jnp.zeros(grid.face_shape(1))
    p = jnp.zeros(grid.cell_shape())

    xc, yc = grid.cell_centers()
    T_lin = env.T_hot + (env.T_cold - env.T_hot) * yc / Ly  # (Ny,)
    T = np.broadcast_to(T_lin[None, :], (Nx, Ny)).copy()

    XC, YC = np.meshgrid(xc, yc, indexing='ij')
    amp = env.perturb_amp

    # Random mix of modes k=1..6 with random amplitude [0.5,1.5] and phase
    for k in range(1, 7):
        a_k = rng.uniform(0.5, 1.5) * amp
        phi_k = rng.uniform(0.0, 2.0 * np.pi)
        T = T + a_k * np.cos(2.0 * np.pi * k * XC / Lx + phi_k) * np.sin(np.pi * YC / Ly)

    return FluidState(velocity=(u, v), pressure=p, scalars={'T': jnp.asarray(T)}, ibm_forces=None)


def measure_ic(env, seed, warmup_steps, baseline_steps):
    """Warmup + baseline Nu measurement for one IC."""
    state = build_ic(env, seed)

    nu_warmup = []
    for i in range(warmup_steps):
        state = env._step_jit(state, env.dt, None)
        if (i + 1) % 500 == 0:
            nu_warmup.append(float(env._compute_nu_jit(state)))

    nu_series = np.empty(baseline_steps)
    for i in range(baseline_steps):
        state = env._step_jit(state, env.dt, None)
        nu_series[i] = float(env._compute_nu_jit(state))

    return {
        'seed': seed,
        'nu_warmup_samples': nu_warmup,
        'nu_mean': float(np.mean(nu_series)),
        'nu_max': float(np.max(nu_series)),
        'nu_min': float(np.min(nu_series)),
        'nu_std': float(np.std(nu_series)),
        'nu_late_mean': float(np.mean(nu_series[-1000:])),
        'nu_late_std': float(np.std(nu_series[-1000:])),
        'n_samples': baseline_steps,
    }


if __name__ == '__main__':
    cfg = {**ENV_CONFIG, 'warmup_steps': 3000, 'baseline_steps': 3000}

    print(f"=== RB Ra=1e4 Nu_bl IC sensitivity ===")
    print(f"Grid {cfg['Nx']}x{cfg['Ny']}, domain {cfg['Lx']:.3f}×{cfg['Ly']}")
    print(f"Pr={cfg['Pr']}, dt={cfg['dt']}, warmup={cfg['warmup_steps']}, "
          f"baseline={cfg['baseline_steps']}")

    env = RBControlEnv(cfg)

    results = []
    seeds = list(range(10))
    for seed in seeds:
        t0 = time.time()
        r = measure_ic(env, seed, cfg['warmup_steps'], cfg['baseline_steps'])
        r['time_min'] = (time.time() - t0) / 60
        print(f"\n  seed={seed}: Nu_mean={r['nu_mean']:.4f} "
              f"(max={r['nu_max']:.4f}, min={r['nu_min']:.4f}, "
              f"late={r['nu_late_mean']:.4f}±{r['nu_late_std']:.4f}) "
              f"[{r['time_min']:.1f}min]")
        results.append(r)

    nu_means = np.array([r['nu_mean'] for r in results])
    nu_late = np.array([r['nu_late_mean'] for r in results])
    nu_maxes = np.array([r['nu_max'] for r in results])

    print("\n" + "=" * 70)
    print(f"Summary over {len(seeds)} ICs (Ra=10^4, MAC FD, 96×64, 2π×2)")
    print("=" * 70)
    print(f"Nu_mean (full baseline):  {nu_means.mean():.4f} ± {nu_means.std():.4f}"
          f"  [min={nu_means.min():.4f}, max={nu_means.max():.4f}]")
    print(f"Nu_mean (late 1000):      {nu_late.mean():.4f} ± {nu_late.std():.4f}"
          f"  [min={nu_late.min():.4f}, max={nu_late.max():.4f}]")
    print(f"Nu_max (peak):            {nu_maxes.mean():.4f} ± {nu_maxes.std():.4f}"
          f"  [min={nu_maxes.min():.4f}, max={nu_maxes.max():.4f}]")
    print(f"Our default Nu_bl:        4.306 (single fixed IC, 1000-step baseline)")
    print(f"Literature Ra=10^4:")
    print(f"  Vignon 2023 (Shenfun):  Nu_bl = 2.68")
    print(f"  Markmann 2025 (Shenfun): Nu_bl = 3.9 (mean over 20 test ICs)")

    # single deterministically initialised IC: the environment's own reset() baseline
    env_default = RBControlEnv(dict(ENV_CONFIG))
    env_default.reset()
    default_bl = float(env_default.nu_baseline) if getattr(env_default, 'nu_baseline', None) is not None else None
    print(f"Default-IC env.reset() baseline: {default_bl}")
    import json
    out = 'results/verification/v07_rb_nu_bl_ic_sensitivity.json'
    with open(out, 'w') as f:
        json.dump(dict(per_ic=results, config=dict(warmup_steps=cfg['warmup_steps'], baseline_steps=cfg['baseline_steps']),
                       nu_mean_ensemble=float(nu_means.mean()), nu_std_ensemble_pop=float(nu_means.std()),
                       nu_min=float(nu_means.min()), nu_max=float(nu_means.max()),
                       default_ic_env_baseline=default_bl), f, indent=2)
    print(f"\nSaved: {out}")
