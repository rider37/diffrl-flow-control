"""Gradient-estimator variance at the trained SHAC policies (Fig. 11a, Section 5.2).

At the final SHAC checkpoint of each Ra (training seed 1223) and a fixed exploration standard deviation, the gradient of the
expected return is estimated on the same K sampled trajectories by
  - the pathwise estimator, differentiating the rollout through the solver (SHAC), and
  - the score-function estimator with a batch-mean baseline (the basis of PPO).
Reports the per-trajectory variance V, the signal ||gbar||^2, the relative variance
V/||gbar||^2 and the return variance Var[R].  Running sums, no per-trajectory storage.
Run one Ra per process:  python scripts/analysis/gradient_variance.py <1e4|1e5|1e6>
Writes results/diagnostics/gradient_variance_ra{Ra}.json.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json, time
import numpy as np
import jax
import jax.numpy as jnp

from diffrl_flow_control.configs import load_config
ENV_RA1E4 = load_config('rayleigh_benard', 'env_ra1e4')
SHAC_RA1E4 = load_config('rayleigh_benard', 'shac_ra1e4')
ENV_RA1E5 = load_config('rayleigh_benard', 'env_ra1e5')
SHAC_RA1E5 = load_config('rayleigh_benard', 'shac_ra1e5')
ENV_RA1E6 = load_config('rayleigh_benard', 'env_ra1e6')
SHAC_RA1E6 = load_config('rayleigh_benard', 'shac_ra1e6')
from diffrl_flow_control.rayleigh_benard.ppo import mlp_forward
from diffrl_flow_control.rayleigh_benard.shac import (
    make_heater_step_fn, make_rb_get_obs_fn, make_compute_nu_fn,
    make_map_action_fn, differentiable_simulate_segmented)
from diffrl_flow_control.rayleigh_benard.env import RBControlEnv
from diffrl_flow_control.rayleigh_benard.heldout import build_ic, load_params_flexible

RA = sys.argv[1] if len(sys.argv) > 1 else '1e4'
K = int(sys.argv[2]) if len(sys.argv) > 2 else 40          # trajectories
EPOCHS = [1000]                                         # final policy (epoch 1000)
SIGMA = float(sys.argv[4]) if len(sys.argv) > 4 else np.exp(-0.5)   # fixed exploration std
CFG = {'1e4': (ENV_RA1E4, SHAC_RA1E4), '1e5': (ENV_RA1E5, SHAC_RA1E5), '1e6': (ENV_RA1E6, SHAC_RA1E6)}
env_cfg, shac_cfg = CFG[RA]
CKPT_DIR = f'runs/rayleigh_benard/shac_ra{RA}_s1223/checkpoints'

# ── env + pure functions (mirror train_shac.py setup) ──────────────────────
env = RBControlEnv(env_cfg)
shac_step_fn = make_heater_step_fn(env.step_fn, env.T_hot, env.amplitude)
get_obs_fn = make_rb_get_obs_fn(env.probe_ix_c, env.probe_iy_c, env.probe_ix_u,
                                env.probe_iy_u, env.probe_ix_v, env.probe_iy_v)
compute_nu_fn = make_compute_nu_fn(env.grid, env.T_hot, env.T_cold,
                                   segment_map=env.segment_map)
map_action_fn = make_map_action_fn(env.T_hot, env.amplitude)
env.reset(jax.random.PRNGKey(0))           # populates env.nu_baseline (as in train_shac)
nu_baseline = float(env.nu_baseline)
action_dim = int(env.action_size)
h = int(shac_cfg['horizon'])
action_interval = int(shac_cfg['action_interval'])
segment_length = int(shac_cfg['segment_length'])
dt = float(env_cfg['dt'])
gamma = float(shac_cfg['gamma'])
warmup = int(getattr(env, 'warmup_steps', env_cfg.get('warmup_steps', 2000)))

print(f'[gradvar] Ra={RA} epochs={EPOCHS} h={h} ai={action_interval} seg={segment_length} '
      f'obs={env.obs_size} adim={action_dim} sigma={SIGMA:.3f} K={K} warmup={warmup} '
      f'Nu_bl={nu_baseline:.3f}', flush=True)


# ── reparameterized stochastic rollout: returns R (diff'able) + recorded data ──
def rollout(actor_params, state0, noise_seq):
    @jax.checkpoint
    def step(state, noise_t):
        obs = get_obs_fn(state)
        mean = mlp_forward(actor_params, obs)[:action_dim]
        action_raw = mean + SIGMA * noise_t
        action = jnp.tanh(action_raw)
        state_next = differentiable_simulate_segmented(
            shac_step_fn, state, action_interval, dt,
            segment_length=segment_length, action=action)
        nu = compute_nu_fn(state_next, map_action_fn(action))
        r = 1.0 - nu / nu_baseline
        return state_next, (r, obs, action_raw)
    _, (rewards, obs_seq, action_raw_seq) = jax.lax.scan(step, state0, noise_seq)
    R = jnp.sum((gamma ** jnp.arange(h)) * rewards)
    return R, (obs_seq, action_raw_seq)


# PW: value_and_grad of the reparam return (aux = recorded trajectory)
pw_vg = jax.jit(jax.value_and_grad(rollout, argnums=0, has_aux=True))


# SF: gradient of sum_t log pi(a_raw_t | mean(theta,obs_t), sigma) w.r.t. theta,
#     with obs_seq, action_raw_seq treated as fixed sampled data.
def logp_sum(actor_params, obs_seq, action_raw_seq):
    means = jax.vmap(lambda o: mlp_forward(actor_params, o)[:action_dim])(obs_seq)
    return jnp.sum(-0.5 * ((action_raw_seq - means) / SIGMA) ** 2)
sf_grad = jax.jit(jax.grad(logp_sum, argnums=0))

flat = lambda g: jnp.concatenate([x.ravel() for x in jax.tree.leaves(g)])

# ── cache warmed-up ICs + noise ONCE (uncontrolled warmup is epoch-independent) ──
print(f'[gradvar] warming up {K} ICs (warmup={warmup})...', flush=True)
t0 = time.time()
states, noises = [], []
for i in range(K):
    s = build_ic(env, 200 + i)
    for _ in range(warmup):
        s = env._step_jit(s, env.dt, None)
    states.append(s)
    noises.append(jax.random.normal(jax.random.PRNGKey(10_000 + i), (h, action_dim)))
print(f'[gradvar] warmup done [{(time.time()-t0)/60:.1f}m]', flush=True)


def measure(actor):
    """Per-trajectory SF & PW gradient estimator variance at a fixed policy."""
    SgPW = 0.0; Sgg_PW = 0.0
    Sg = 0.0; SRg = 0.0
    Sg2 = 0.0; SRg2 = 0.0; SR2g2 = 0.0
    SR = 0.0; SR2 = 0.0; n = 0
    for i in range(K):
        (R, (obs_seq, action_raw_seq)), gPW = pw_vg(actor, states[i], noises[i])
        R = float(R)
        if not np.isfinite(R):
            continue
        gPW = flat(gPW)
        glp = flat(sf_grad(actor, obs_seq, action_raw_seq))
        g2 = float(jnp.sum(glp ** 2))
        SgPW = SgPW + gPW; Sgg_PW += float(jnp.sum(gPW ** 2))
        Sg = Sg + glp; SRg = SRg + R * glp
        Sg2 += g2; SRg2 += R * g2; SR2g2 += R * R * g2
        SR += R; SR2 += R * R; n += 1
    b = SR / n
    gbar_PW = SgPW / n; sig_PW = float(jnp.sum(gbar_PW ** 2)); V_PW = Sgg_PW / n - sig_PW
    gbar_SF = (SRg - b * Sg) / n; sig_SF = float(jnp.sum(gbar_SF ** 2))
    V_SF = (SR2g2 - 2 * b * SRg2 + b * b * Sg2) / n - sig_SF
    return dict(K=n, mean_return=b, Var_return=SR2 / n - b * b,
                V_PW=V_PW, sig_PW=sig_PW, relV_PW=V_PW / sig_PW,
                V_SF=V_SF, sig_SF=sig_SF, relV_SF=V_SF / sig_SF,
                V_SF_over_V_PW=V_SF / V_PW)


os.makedirs('results/diagnostics', exist_ok=True)
for EPOCH in EPOCHS:
    ckpt = f'{CKPT_DIR}/final.npz'
    if not os.path.exists(ckpt):
        print(f'[ep{EPOCH}] ckpt missing, skip', flush=True); continue
    actor = load_params_flexible(ckpt)[0]['actor']
    te = time.time()
    res = measure(actor)
    out = dict(Ra=RA, epoch=EPOCH, sigma=SIGMA, h=h, nu_baseline=nu_baseline, **res)
    json.dump(out, open(f'results/diagnostics/gradient_variance_ra{RA}.json', 'w'), indent=2)
    print(f'[ep{EPOCH}] relV_PW={res["relV_PW"]:.2f} relV_SF={res["relV_SF"]:.2f} '
          f'sig_PW={res["sig_PW"]:.3e} sig_SF={res["sig_SF"]:.3e} VarR={res["Var_return"]:.3f} '
          f'V_SF/V_PW={res["V_SF_over_V_PW"]:.0f}x [{(time.time()-te)/60:.1f}m]', flush=True)
print(f'[gradvar] Ra={RA} all epochs done [{(time.time()-t0)/60:.1f} min total]', flush=True)
