"""Pathwise-gradient variance against the differentiated horizon (Fig. 11c).

At the final SHAC checkpoint of each Ra (training seed 1223), the relative variance
of the pathwise gradient is measured over K trajectories for horizons of 5 to 160
actions.
Run one Ra per process:  python scripts/analysis/gradient_variance_horizon.py <1e4|1e5|1e6>
Writes results/diagnostics/gradient_variance_horizon_ra{Ra}.json.
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
K = int(sys.argv[2]) if len(sys.argv) > 2 else 32
HORIZONS = ([int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3
            else [5, 10, 20, 40, 80, 120, 160])
SIGMA = np.exp(-0.5)
EPOCH = 1000
CFG = {'1e4': (ENV_RA1E4, SHAC_RA1E4), '1e5': (ENV_RA1E5, SHAC_RA1E5),
       '1e6': (ENV_RA1E6, SHAC_RA1E6)}
env_cfg, shac_cfg = CFG[RA]
CKPT = f'runs/rayleigh_benard/shac_ra{RA}_s1223/checkpoints' + f'/final.npz'

env = RBControlEnv(env_cfg)
shac_step_fn = make_heater_step_fn(env.step_fn, env.T_hot, env.amplitude)
get_obs_fn = make_rb_get_obs_fn(env.probe_ix_c, env.probe_iy_c, env.probe_ix_u,
                                env.probe_iy_u, env.probe_ix_v, env.probe_iy_v)
compute_nu_fn = make_compute_nu_fn(env.grid, env.T_hot, env.T_cold, segment_map=env.segment_map)
map_action_fn = make_map_action_fn(env.T_hot, env.amplitude)
env.reset(jax.random.PRNGKey(0))
nu_baseline = float(env.nu_baseline)
action_dim = int(env.action_size)
action_interval = int(shac_cfg['action_interval'])
segment_length = int(shac_cfg['segment_length'])
dt = float(env_cfg['dt'])
gamma = float(shac_cfg['gamma'])
h_shac = int(shac_cfg['horizon'])
warmup = int(getattr(env, 'warmup_steps', env_cfg.get('warmup_steps', 2000)))
actor = load_params_flexible(CKPT)[0]['actor']
flat = lambda g: jnp.concatenate([x.ravel() for x in jax.tree.leaves(g)])
print(f'[horizon] Ra={RA} h_shac={h_shac} sweep={HORIZONS} K={K} adim={action_dim} '
      f'warmup={warmup} Nu_bl={nu_baseline:.3f}', flush=True)

# cache warmed-up ICs + max-horizon noise once
HMAX = max(HORIZONS)
t0 = time.time()
states, noises = [], []
for i in range(K):
    s = build_ic(env, 200 + i)
    for _ in range(warmup):
        s = env._step_jit(s, env.dt, None)
    states.append(s)
    noises.append(jax.random.normal(jax.random.PRNGKey(10_000 + i), (HMAX, action_dim)))
print(f'[horizon] warmup done [{(time.time()-t0)/60:.1f}m]', flush=True)


def build_measure(h):
    def rollout(actor_params, state0, noise_seq):       # noise_seq: (h, adim)
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
            return state_next, (1.0 - nu / nu_baseline, obs, action_raw)
        _, (rewards, obs_seq, araw) = jax.lax.scan(step, state0, noise_seq)
        R = jnp.sum((gamma ** jnp.arange(h)) * rewards)
        return R, (obs_seq, araw)
    pw_vg = jax.jit(jax.value_and_grad(rollout, argnums=0, has_aux=True))

    def logp_sum(actor_params, obs_seq, araw):
        means = jax.vmap(lambda o: mlp_forward(actor_params, o)[:action_dim])(obs_seq)
        return jnp.sum(-0.5 * ((araw - means) / SIGMA) ** 2)
    sf_grad = jax.jit(jax.grad(logp_sum, argnums=0))

    SgPW = 0.0; Sgg_PW = 0.0; SnPW = 0.0
    Sg = 0.0; SRg = 0.0; Sg2 = 0.0; SRg2 = 0.0; SR2g2 = 0.0
    SR = 0.0; SR2 = 0.0; n = 0
    for i in range(K):
        (R, (obs_seq, araw)), gPW = pw_vg(actor, states[i], noises[i][:h])
        R = float(R)
        if not np.isfinite(R):
            continue
        gPW = flat(gPW); nrm = float(jnp.sum(gPW ** 2))
        glp = flat(sf_grad(actor, obs_seq, araw)); g2 = float(jnp.sum(glp ** 2))
        SgPW = SgPW + gPW; Sgg_PW += nrm; SnPW += nrm
        Sg = Sg + glp; SRg = SRg + R * glp
        Sg2 += g2; SRg2 += R * g2; SR2g2 += R * R * g2
        SR += R; SR2 += R * R; n += 1
    b = SR / n
    gbar_PW = SgPW / n; sig_PW = float(jnp.sum(gbar_PW ** 2)); V_PW = Sgg_PW / n - sig_PW
    gbar_SF = (SRg - b * Sg) / n; sig_SF = float(jnp.sum(gbar_SF ** 2))
    V_SF = (SR2g2 - 2 * b * SRg2 + b * b * Sg2) / n - sig_SF
    return dict(h=h, K=n, mean_return=b, Var_return=SR2 / n - b * b,
                gradnorm2_PW_mean=SnPW / n, V_PW=V_PW, sig_PW=sig_PW, relV_PW=V_PW / sig_PW,
                V_SF=V_SF, sig_SF=sig_SF, relV_SF=V_SF / sig_SF, V_SF_over_V_PW=V_SF / V_PW)


os.makedirs('results/diagnostics', exist_ok=True)
allres = []
for h in HORIZONS:
    te = time.time()
    try:
        res = build_measure(h)
    except Exception as e:
        print(f'[h={h}] FAILED {e!r}', flush=True); continue
    allres.append(res)
    json.dump({'Ra': RA, 'epoch': EPOCH, 'h_shac': h_shac, 'sweep': allres},
              open(f'results/diagnostics/gradient_variance_horizon_ra{RA}.json', 'w'), indent=2)
    print(f'[h={h:>4}] |gPW|^2={res["gradnorm2_PW_mean"]:.3e} V_PW={res["V_PW"]:.3e} '
          f'V_SF={res["V_SF"]:.3e} V_SF/V_PW={res["V_SF_over_V_PW"]:.2e} '
          f'relV_PW={res["relV_PW"]:.1f} relV_SF={res["relV_SF"]:.1f} [{(time.time()-te)/60:.1f}m]',
          flush=True)
print(f'[horizon] Ra={RA} done [{(time.time()-t0)/60:.1f}m]', flush=True)
