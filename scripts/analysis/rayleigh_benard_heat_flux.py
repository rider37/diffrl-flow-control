"""Data of Fig. 8: heat flux and wall actuation at Ra = 1e6.

The uncontrolled flow and the final PPO and SHAC policies (training seed 42) run
for 600 actions from held-out IC seed 100.  Saves the Nu(t) series, the action
norms |a(t)| and their Welch spectra.
Writes results/figure_data/fig08_heat_flux.npz and fig08_heat_flux.json.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json
import numpy as np
import jax
import jax.numpy as jnp


from scipy.signal import welch

from diffrl_flow_control.configs import load_config
ENV_RA1E6 = load_config('rayleigh_benard', 'env_ra1e6')
from diffrl_flow_control.rayleigh_benard.env import RBControlEnv
from diffrl_flow_control.rayleigh_benard.heldout import build_ic, load_params_flexible
from diffrl_flow_control.rayleigh_benard.ppo import mlp_forward

RA = '1e6'
SEED = 42
IC_SEED = 100
N_ACTIONS = 600  # long horizon for spectrum
SHAC_DIR = os.path.join(ROOT, 'runs/rayleigh_benard/shac_ra1e6_s42')
PPO_DIR  = os.path.join(ROOT, 'runs/rayleigh_benard/ppo_ra1e6_s42')


def find_ckpt(d, algo):
    return f'{d}/checkpoints/final.npz'


def rollout(env, n_actions, controller, controller_state=None):
    """Run rollout with controller; return Nu(t), action(t)."""
    state = build_ic(env, IC_SEED)
    for _ in range(env.warmup_steps):
        state = env._step_jit(state, env.dt, None)
    action_step = int(env.action_interval)
    nu_hist = []; act_hist = []
    aux = controller_state
    for _ in range(n_actions):
        action, aux = controller(state, aux)
        T_seg = env.map_action(jnp.asarray(action))
        for _ in range(action_step):
            state = env._step_jit(state, env.dt, T_seg)
        nu_hist.append(float(env.compute_nu(state, T_seg)))
        act_hist.append(np.array(action))
    return np.array(nu_hist), np.stack(act_hist, axis=0)


def baseline_rollout(env, n_actions):
    state = build_ic(env, IC_SEED)
    for _ in range(env.warmup_steps):
        state = env._step_jit(state, env.dt, None)
    action_step = int(env.action_interval)
    nu_hist = []
    for _ in range(n_actions):
        for _ in range(action_step):
            state = env._step_jit(state, env.dt, None)
        nu_hist.append(float(env.compute_nu(state)))
    return np.array(nu_hist)


def make_neural_controller(actor_params, n_seg):
    @jax.jit
    def act(obs):
        out = mlp_forward(actor_params, obs)
        return jnp.tanh(out[:n_seg])
    def controller(state, _aux):
        obs = env._get_obs(state)
        return np.array(act(obs)), None
    return controller


# ── Main ─────────────────────────────────────────────────────────────
env = RBControlEnv(ENV_RA1E6)
n_seg = env.n_segments
print(f'n_seg={n_seg}, action_interval={env.action_interval}')

print('\n[baseline]')
nu_bl_t = baseline_rollout(env, N_ACTIONS)
nu_bl_mean = float(np.mean(nu_bl_t))
print(f'  Nu_baseline mean = {nu_bl_mean:.3f}')

print('\n[PPO]')
p_ppo, _ = load_params_flexible(find_ckpt(PPO_DIR, 'ppo'))
ctrl = make_neural_controller(p_ppo['actor'], n_seg)
nu_ppo_t, a_ppo_t = rollout(env, N_ACTIONS, ctrl)

print('\n[SHAC]')
p_shac, _ = load_params_flexible(find_ckpt(SHAC_DIR, 'shac'))
ctrl = make_neural_controller(p_shac['actor'], n_seg)
nu_shac_t, a_shac_t = rollout(env, N_ACTIONS, ctrl)

# action time = action step × CFD substeps × dt
dt_ctrl = float(env.action_interval) * float(env.dt)
fs = 1.0 / dt_ctrl
print(f'\n  control sampling: dt_ctrl={dt_ctrl:.4f}, fs={fs:.3f} Hz')

def psd(x, fs):
    nperseg = min(256, len(x) // 4)
    return welch(x, fs=fs, nperseg=nperseg, detrend='constant')

# PSDs of Nu(t)
f_bl, P_bl   = psd(nu_bl_t,   fs)
f_ppo, P_ppo = psd(nu_ppo_t,  fs)
f_shac,P_shac= psd(nu_shac_t, fs)

# Wall-action PSD: use spatial-mean magnitude of |a(t)|
mag_ppo  = np.linalg.norm(a_ppo_t, axis=1)
mag_shac = np.linalg.norm(a_shac_t, axis=1)
fa_ppo, Pa_ppo   = psd(mag_ppo,  fs)
fa_shac, Pa_shac = psd(mag_shac, fs)

# Means for table
print('\n=== Means (last 80%) ===')
for label, x in [('baseline', nu_bl_t), ('PPO', nu_ppo_t), ('SHAC', nu_shac_t)]:
    cut = int(0.2 * len(x))
    m = float(np.mean(x[cut:]))
    print(f'  {label}: Nu={m:.3f}  (red={(1-m/nu_bl_mean)*100:+.2f}%)')

# ── spectra (the figure itself is figures/fig08_heat_flux.py) ─
CPC_FIG = os.path.join(ROOT, 'results/figure_data')
os.makedirs(CPC_FIG, exist_ok=True)
np.savez(os.path.join(CPC_FIG, 'fig08_heat_flux.npz'),
         f_bl=f_bl, P_bl=P_bl, f_ppo=f_ppo, P_ppo=P_ppo, f_shac=f_shac, P_shac=P_shac,
         fa_ppo=fa_ppo, Pa_ppo=Pa_ppo, fa_shac=fa_shac, Pa_shac=Pa_shac,
         nu_bl_t=nu_bl_t, nu_ppo_t=nu_ppo_t, nu_shac_t=nu_shac_t,
         mag_ppo=mag_ppo, mag_shac=mag_shac, dt_ctrl=dt_ctrl)
print('  Saved: results/figure_data/fig08_heat_flux.npz')

# ── Save numerics ────────────────────────────────────────────────────
out = {
    'Ra': RA, 'seed': SEED, 'ic_seed': IC_SEED,
    'n_actions': N_ACTIONS, 'dt_ctrl': dt_ctrl,
    'nu_baseline_mean': nu_bl_mean,
    'nu_means_last80pct': {
        'baseline': float(np.mean(nu_bl_t[int(0.2*len(nu_bl_t)):])),
        'PPO':      float(np.mean(nu_ppo_t[int(0.2*len(nu_ppo_t)):])),
        'SHAC':     float(np.mean(nu_shac_t[int(0.2*len(nu_shac_t)):])),
    },
}
with open('results/figure_data/fig08_heat_flux.json', 'w') as f:
    json.dump(out, f, indent=2)
print('  Saved: results/figure_data/fig08_heat_flux.json')
