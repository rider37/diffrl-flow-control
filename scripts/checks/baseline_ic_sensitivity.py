"""Sensitivity of the uncontrolled Ra = 1e4 Nusselt number to the initial
condition (Appendix C): Nu over the reward-baseline window for an ensemble of
IC seeds and for the reference IC of the environment.
Writes results/checks/baseline_ic_sensitivity.json.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import time
import numpy as np
import jax
import jax.numpy as jnp


from diffrl_flow_control.configs import load_config
ENV_RA1E4 = load_config('rayleigh_benard', 'env_ra1e4')
from diffrl_flow_control.rayleigh_benard.env import RBControlEnv, FluidState


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
    cfg = {**ENV_RA1E4, 'warmup_steps': 3000, 'baseline_steps': 3000}

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
    print(f"Literature Ra=10^4:")
    print(f"  Vignon 2023 (Shenfun):  Nu_bl = 2.68")
    print(f"  Markmann 2025 (Shenfun): Nu_bl = 3.9 (mean over 20 test ICs)")

    # single deterministically initialised IC: the environment's own reset() baseline
    env_default = RBControlEnv(dict(ENV_RA1E4))
    env_default.reset()
    default_bl = float(env_default.nu_baseline) if getattr(env_default, 'nu_baseline', None) is not None else None
    print(f"Default-IC env.reset() baseline: {default_bl}")
    import json
    out = 'results/checks/baseline_ic_sensitivity.json'
    with open(out, 'w') as f:
        json.dump(dict(per_ic=results, config=dict(warmup_steps=cfg['warmup_steps'], baseline_steps=cfg['baseline_steps']),
                       nu_mean_ensemble=float(nu_means.mean()), nu_std_ensemble_pop=float(nu_means.std()),
                       nu_min=float(nu_means.min()), nu_max=float(nu_means.max()),
                       default_ic_env_baseline=default_bl), f, indent=2)
    print(f"\nSaved: {out}")
