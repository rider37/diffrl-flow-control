"""Power-spectral density of Nu(t) for baseline / PD / PPO / SHAC at Ra=1e6.

Address C1 (RB physical interpretation depth) by quantifying the
frequency-domain footprint of each controller. We roll out long enough
to estimate Welch PSD on Nu(t) and the per-segment wall-temperature
action, then plot the spectra.

Usage
-----
    python scripts/rb_spectrum_compare.py
"""
import os, sys, json
import numpy as np
import jax
import jax.numpy as jnp

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'src'))
sys.path.insert(0, os.path.join(_root, 'scripts'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

from experiments.rb_control.config import ENV_CONFIG_RA1E6
from jax_cfd.envs.rb_control_env import RBControlEnv
from rb_eval_common import build_ic, load_params_flexible
from experiments.rb_control.train_ppo import mlp_forward

FIG_DIR = os.path.join(_root, 'paper/pof/fig')
os.makedirs(FIG_DIR, exist_ok=True)

RA = '1e6'
SEED = 42
IC_SEED = 100
N_ACTIONS = 600  # long horizon for spectrum
SHAC_DIR = os.path.join(_root, 'experiments/rb_control/output_wm_shac_ra1e6_s42')
PPO_DIR  = os.path.join(_root, 'experiments/rb_control/output_phase3_ppo_ra1e6_s42')
PD_DIR   = os.path.join(_root, 'experiments/rb_control/output_pd_ra1e6')


def best_pd_from_grid(grid_dir):
    best = None
    for fn in os.listdir(grid_dir):
        if not fn.startswith('pd_Kp') or not fn.endswith('.json'):
            continue
        d = json.load(open(os.path.join(grid_dir, fn)))
        red = d.get('nu_reduction_pct')
        if red is None: continue
        if best is None or red > best['nu_reduction_pct']:
            best = d
    return best


def find_ckpt(d, algo):
    if algo == 'shac':
        for cand in ['fans_final_epoch_1000.npz', 'fans_final_epoch_2000.npz']:
            p = f'{d}/checkpoints/{cand}'
            if os.path.exists(p): return p
        eps = sorted([int(f.split('_')[2].split('.')[0])
                      for f in os.listdir(f'{d}/checkpoints')
                      if f.startswith('fans_epoch_')])
        return f'{d}/checkpoints/fans_epoch_{eps[-1]:04d}.npz'
    else:
        p = f'{d}/checkpoints/final_params.npz'
        if os.path.exists(p): return p
        eps = sorted([int(f.split('_')[1].split('.')[0])
                      for f in os.listdir(f'{d}/checkpoints')
                      if f.startswith('ckpt_')])
        return f'{d}/checkpoints/ckpt_{eps[-1]:04d}.npz'


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


def make_pd_controller(env, Kp, Kd, n_seg):
    Nx = env.grid.shape[0]
    seg_w = Nx // n_seg
    aux_init = {'T_prev': None}
    def controller(state, aux):
        T_arr = np.array(state.scalars['T'])[:, 0]
        T_seg = np.array([T_arr[i*seg_w:(i+1)*seg_w].mean() for i in range(n_seg)])
        anom = T_seg - T_seg.mean()
        d_anom = (T_seg - aux['T_prev']) if aux['T_prev'] is not None else np.zeros_like(T_seg)
        action = np.clip(-Kp * anom - Kd * d_anom, -1.0, 1.0)
        aux['T_prev'] = T_seg
        return action, aux
    return controller, dict(aux_init)


# ── Main ─────────────────────────────────────────────────────────────
env = RBControlEnv(ENV_CONFIG_RA1E6)
n_seg = env.n_segments
print(f'n_seg={n_seg}, action_interval={env.action_interval}')

print('\n[baseline]')
nu_bl_t = baseline_rollout(env, N_ACTIONS)
nu_bl_mean = float(np.mean(nu_bl_t))
print(f'  Nu_baseline mean = {nu_bl_mean:.3f}')

print('\n[PD]')
pd = best_pd_from_grid(PD_DIR)
print(f'  Kp={pd["Kp"]}, Kd={pd["Kd"]}')
ctrl, aux0 = make_pd_controller(env, pd['Kp'], pd['Kd'], n_seg)
nu_pd_t, a_pd_t = rollout(env, N_ACTIONS, ctrl, aux0)

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
f_pd, P_pd   = psd(nu_pd_t,   fs)
f_ppo, P_ppo = psd(nu_ppo_t,  fs)
f_shac,P_shac= psd(nu_shac_t, fs)

# Wall-action PSD: use spatial-mean magnitude of |a(t)|
mag_pd   = np.linalg.norm(a_pd_t, axis=1)
mag_ppo  = np.linalg.norm(a_ppo_t, axis=1)
mag_shac = np.linalg.norm(a_shac_t, axis=1)
fa_pd, Pa_pd     = psd(mag_pd,   fs)
fa_ppo, Pa_ppo   = psd(mag_ppo,  fs)
fa_shac, Pa_shac = psd(mag_shac, fs)

# Means for table
print('\n=== Means (last 80%) ===')
for label, x in [('baseline', nu_bl_t), ('PD', nu_pd_t),
                 ('PPO', nu_ppo_t), ('SHAC', nu_shac_t)]:
    cut = int(0.2 * len(x))
    m = float(np.mean(x[cut:]))
    print(f'  {label}: Nu={m:.3f}  (red={(1-m/nu_bl_mean)*100:+.2f}%)')

# ── Plot 2-panel: Nu PSD + action-magnitude PSD (PD dropped; SHAC-PPO) ─
CPC_FIG = os.path.join(_root, 'results/figure_data')
os.makedirs(CPC_FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'font.size': 8.5, 'axes.linewidth': 0.6,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
SH, PP, BL = '#2E5894', '#B6324B', '0.55'
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))

ax = axes[0]
ax.loglog(f_bl,  P_bl,  '-', color=BL, lw=1.0, label='uncontrolled')
ax.loglog(f_ppo, P_ppo, '-', color=PP, lw=1.3, label='PPO')
ax.loglog(f_shac, P_shac, '-', color=SH, lw=1.7, label='SHAC')
ax.set_xlabel(r'$f\,t_{\mathrm{ff}}$'); ax.set_ylabel(r'PSD of $\mathrm{Nu}(t)$')
ax.set_title('(a) heat-flux spectrum', fontsize=9)
ax.grid(alpha=0.2, which='both', lw=0.4)
ax.legend(fontsize=7.5, framealpha=0.9, edgecolor='0.8')

ax = axes[1]
ax.loglog(fa_ppo, Pa_ppo, '-', color=PP, lw=1.3, label='PPO')
ax.loglog(fa_shac, Pa_shac, '-', color=SH, lw=1.7, label='SHAC')
ax.set_xlabel(r'$f\,t_{\mathrm{ff}}$')
ax.set_ylabel(r'PSD of $\|\mathbf{a}(t)\|_2$')
ax.set_title('(b) wall-action spectrum', fontsize=9)
ax.grid(alpha=0.2, which='both', lw=0.4)
ax.legend(fontsize=7.5, framealpha=0.9, edgecolor='0.8')

fig.tight_layout()
fig.savefig(os.path.join(CPC_FIG, 'fig7_rb_spectrum.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(CPC_FIG, 'fig7_rb_spectrum_preview.png'), dpi=170,
            bbox_inches='tight')
plt.close(fig)
# cache arrays so the figure can be re-styled later WITHOUT re-running on GPU
np.savez(os.path.join(CPC_FIG, 'fig7_rb_spectrum_data.npz'),
         f_bl=f_bl, P_bl=P_bl, f_pd=f_pd, P_pd=P_pd, f_ppo=f_ppo, P_ppo=P_ppo,
         f_shac=f_shac, P_shac=P_shac, fa_pd=fa_pd, Pa_pd=Pa_pd,
         fa_ppo=fa_ppo, Pa_ppo=Pa_ppo, fa_shac=fa_shac, Pa_shac=Pa_shac,
         nu_bl_t=nu_bl_t, nu_pd_t=nu_pd_t, nu_ppo_t=nu_ppo_t, nu_shac_t=nu_shac_t,
         mag_pd=mag_pd, mag_ppo=mag_ppo, mag_shac=mag_shac, dt_ctrl=dt_ctrl)
print(f'\n  Saved: {CPC_FIG}/fig7_rb_spectrum.pdf (+ _data.npz)')

# ── Save numerics ────────────────────────────────────────────────────
out = {
    'Ra': RA, 'seed': SEED, 'ic_seed': IC_SEED,
    'n_actions': N_ACTIONS, 'dt_ctrl': dt_ctrl,
    'nu_baseline_mean': nu_bl_mean,
    'nu_means_last80pct': {
        'baseline': float(np.mean(nu_bl_t[int(0.2*len(nu_bl_t)):])),
        'PD':       float(np.mean(nu_pd_t[int(0.2*len(nu_pd_t)):])),
        'PPO':      float(np.mean(nu_ppo_t[int(0.2*len(nu_ppo_t)):])),
        'SHAC':     float(np.mean(nu_shac_t[int(0.2*len(nu_shac_t)):])),
    },
    'PD_gains': {'Kp': pd['Kp'], 'Kd': pd['Kd']},
}
with open('results/figure_data/rb_spectrum_compare.json', 'w') as f:
    json.dump(out, f, indent=2)
print('  Saved: results/figure_data/rb_spectrum_compare.json')
