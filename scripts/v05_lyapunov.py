"""Verification copy of scripts/measure_lyapunov.py on the production cylinder grids with the
calibrated interface (eps = 0.75 dx); saves results/verification/v05_lyap_<problem>_<case>.json."""
"""Chaoticity diagnostic via unnormalised two-trajectory separation growth.

Grounds the CPC paper's "return variance scales with chaoticity" claim with a
first-principles dynamical-systems measurement:
  * Cylinder (stable limit cycle) -> separation rises to a small BOUNDED
    plateau (transverse decay + neutral phase offset); no sustained exponential
    growth, so lambda_max ~ 0.
  * Rayleigh-Benard (chaotic)     -> separation grows EXPONENTIALLY over many
    orders of magnitude before saturating at the attractor size; slope = lambda
    > 0 (growing with Ra).

Method (uncontrolled flow, action off): warm up onto the attractor, clone the
state with a tiny velocity perturbation d0, evolve BOTH WITHOUT renormalisation,
and record the separation d(t) over the (u, v[, T]) fields every step.  Report
(i) the saturation plateau log10(d_max/d0) -- small for a limit cycle, large
(~ full attractor) for chaos -- and (ii) lambda = slope of ln d(t) fitted over
the exponential-growth window (excluding the initial alignment transient and
the saturation tail).

NOTE: a fixed-interval Benettin renorm is unreliable here -- for the limit
cycle the perturbation saturates at the phase-offset amplitude within one
interval, which a long renorm interval misreads as exponential growth.  The
unnormalised trace makes the limit-cycle-vs-chaos distinction unambiguous.

Usage:
  python scripts/measure_lyapunov.py cyl 200
  python scripts/measure_lyapunov.py rb 1e6
"""
import os, sys, json, time
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_V2 = _ROOT   # released solver (eps knob)
sys.path.insert(0, _V2); sys.path.insert(0, os.path.join(_V2, 'src'))
import numpy as np
import jax, jax.numpy as jnp

N_STEPS = int(os.environ.get("N_STEPS", 6000))   # CFD steps to evolve the two trajectories
REC_EVERY = 5             # record separation every this many steps
REL_PERT = 1e-7           # initial perturbation size relative to rms(u)


def perturb_fields(state):
    """Return list of (key, array) leaves we perturb/measure: velocity (+T)."""
    leaves = [('u', state.velocity[0]), ('v', state.velocity[1])]
    if 'T' in state.scalars:
        leaves.append(('T', state.scalars['T']))
    return leaves


def set_fields(state, fields):
    d = dict(fields)
    vel = (d['u'], d['v']) + tuple(state.velocity[2:])
    scal = dict(state.scalars)
    if 'T' in d:
        scal['T'] = d['T']
    return state._replace(velocity=vel, scalars=scal)


def sep_norm(ref, pert):
    diff = []
    for (k, a), (_, b) in zip(perturb_fields(ref), perturb_fields(pert)):
        diff.append((np.asarray(a) - np.asarray(b)).ravel())
    return float(np.linalg.norm(np.concatenate(diff)))


def build_env(problem, case):
    if problem == 'cyl':
        from cases.cylinder_afc import make_env_config
        from jax_cfd.envs.cylinder_flow_env import CylinderFlowEnv
        grid = {100: (768, 144), 200: (768, 144), 400: (1280, 240), 500: (1280, 240)}[int(case)]
        cfg = make_env_config(Re=int(case), cells=grid, ibm_eps_factor=0.75)
        cfg['jet_method'] = 'surface_bc'; cfg['soft_reset_prob'] = 0.0
        env = CylinderFlowEnv(cfg)
        state, _ = env.reset()
        step = lambda s: env._step_jit(s, env.dt, 0.0)
        return env, state, step, env.dt
    else:
        import experiments.rb_control.config as rc
        from jax_cfd.envs.rb_control_env import RBControlEnv
        cfg = {'1e4': rc.ENV_CONFIG, '1e5': rc.ENV_CONFIG_RA1E5,
               '1e6': rc.ENV_CONFIG_RA1E6}[case]
        env = RBControlEnv(cfg)
        state, _ = env.reset()
        step = lambda s: env._step_jit(s, env.dt, None)
        return env, state, step, env.dt


def fit_lambda(t, logd):
    """Slope of ln d vs t over the exponential-growth window.

    Exclude the initial alignment transient (first 15%) and the saturation
    tail (where ln d is within 0.5 of its max).  Returns (lambda, n_fit)."""
    t = np.asarray(t); logd = np.asarray(logd)
    lo = int(0.15 * len(t))
    sat = logd.max() - 0.5
    mask = np.zeros(len(t), bool); mask[lo:] = True
    mask &= logd < sat
    if mask.sum() < 5:
        return 0.0, int(mask.sum())
    A = np.polyfit(t[mask], logd[mask], 1)
    return float(A[0]), int(mask.sum())


def main():
    problem, case = sys.argv[1], sys.argv[2]
    print(f"\n=== chaoticity: {problem} {case} ===", flush=True)
    env, ref, step, dt = build_env(problem, case)

    # initial tiny perturbation on u
    u = np.asarray(ref.velocity[0])
    d0 = REL_PERT * float(np.sqrt(np.mean(u**2)))
    rng = np.random.default_rng(0)
    du = d0 * rng.standard_normal(u.shape) / np.sqrt(u.size)
    pert = set_fields(ref, [('u', jnp.asarray(u + du)), ('v', ref.velocity[1])] +
                     ([('T', ref.scalars['T'])] if 'T' in ref.scalars else []))
    d0 = sep_norm(ref, pert)

    t0 = time.time()
    ts, lds = [], []
    for k in range(N_STEPS):
        ref = step(ref); pert = step(pert)
        if (k + 1) % REC_EVERY == 0:
            jax.block_until_ready(pert.velocity[0])
            d = sep_norm(ref, pert)
            ts.append((k + 1) * dt); lds.append(np.log(d / d0))
            if d / d0 > 1e9:           # fully saturated; stop early
                break
    ts = np.array(ts); lds = np.array(lds)
    lam, nfit = fit_lambda(ts, lds)
    plateau_log10 = float(lds.max() / np.log(10))
    ref_norm = float(np.linalg.norm(np.concatenate([np.asarray(a).ravel() for _, a in perturb_fields(ref)])))
    d_max = float(d0 * 10 ** (lds.max() / np.log(10)))
    out = {'problem': problem, 'case': case, 'dt': dt, 'd0': d0,
           'flow_norm': ref_norm, 'd_max': d_max, 'd_max_over_flow_norm': d_max / ref_norm,
           'lambda_per_time': lam, 'plateau_log10_dmax_d0': plateau_log10,
           'n_fit': nfit, 'final_log10_ratio': float(lds[-1] / np.log(10)),
           'wall_s': time.time() - t0,
           't': ts.tolist(), 'log10_d_over_d0': (lds / np.log(10)).tolist()}
    verdict = ('BOUNDED (limit cycle, lambda~0)' if plateau_log10 < 4 and lam < 0.5
               else f'EXPONENTIAL (chaotic, lambda={lam:.3f}/time)')
    print(f"  plateau log10(d/d0)={plateau_log10:.2f}  lambda_fit={lam:.4f}/time "
          f"(n_fit={nfit})  -> {verdict}", flush=True)
    print(f"  d_max / |flow| = {out['d_max_over_flow_norm']:.4%}", flush=True)
    json.dump(out, open(os.path.join(_ROOT, 'results', 'verification', f'v05_lyap_{problem}_{case}_n{N_STEPS}.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
