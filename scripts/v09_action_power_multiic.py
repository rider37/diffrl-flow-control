"""Action-norm spectral power of the final RB policies on all five held-out ICs (Section 5.1).

Final policies of Table 4 (training seed 42): SHAC Ra=1e4 1000 epochs, width-matched
SHAC Ra=1e5/1e6, PPO ckpt_0100.  200-action rollouts per IC.
Writes results/verification/v09_action_power_multiic.json.
"""
import os, sys, json, time
import numpy as np
import jax
import jax.numpy as jnp

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'src'))
sys.path.insert(0, os.path.join(_root, 'scripts'))

from scipy.signal import welch
from experiments.rb_control.config import (
    ENV_CONFIG, ENV_CONFIG_RA1E5, ENV_CONFIG_RA1E6,
)
from jax_cfd.envs.rb_control_env import RBControlEnv
from rb_eval_common import build_ic, load_params_flexible
from experiments.rb_control.train_ppo import mlp_forward

TEST_ICS = [100, 101, 102, 103, 104]
N_ACTIONS = 200  # training horizon

CASES = {
    '1e4': dict(env_cfg=ENV_CONFIG,
                shac='experiments/rb_control/output_phase3_shac_ra1e4_s42',
                ppo='experiments/rb_control/output_phase2_ppo_ra1e4_s42'),
    '1e5': dict(env_cfg=ENV_CONFIG_RA1E5,
                shac='experiments/rb_control/output_wm_shac_ra1e5_s42',
                ppo='experiments/rb_control/output_phase3_ppo_ra1e5_s42'),
    '1e6': dict(env_cfg=ENV_CONFIG_RA1E6,
                shac='experiments/rb_control/output_wm_shac_ra1e6_s42',
                ppo='experiments/rb_control/output_phase3_ppo_ra1e6_s42'),
}


def find_ckpt(d, algo):
    d = os.path.join(_root, d)
    if not os.path.isdir(d):
        return None
    if algo == 'shac':
        for cand in ['fans_final_epoch_1000.npz', 'fans_final_epoch_2000.npz']:
            p = f'{d}/checkpoints/{cand}'
            if os.path.exists(p): return p
        eps = sorted([int(f.split('_')[2].split('.')[0])
                      for f in os.listdir(f'{d}/checkpoints')
                      if f.startswith('fans_epoch_')])
        return f'{d}/checkpoints/fans_epoch_{eps[-1]:04d}.npz' if eps else None
    else:
        p = f'{d}/checkpoints/ckpt_0100.npz'   # final PPO update (as in Table 4)
        if os.path.exists(p): return p
        eps = sorted([int(f.split('_')[1].split('.')[0])
                      for f in os.listdir(f'{d}/checkpoints')
                      if f.startswith('ckpt_')])
        return f'{d}/checkpoints/ckpt_{eps[-1]:04d}.npz' if eps else None


def rollout(env, ic_seed, n_actions, controller):
    state = build_ic(env, ic_seed)
    for _ in range(env.warmup_steps):
        state = env._step_jit(state, env.dt, None)
    action_step = int(env.action_interval)
    nu_hist = []; act_hist = []
    for _ in range(n_actions):
        action = controller(state)
        T_seg = env.map_action(jnp.asarray(action))
        for _ in range(action_step):
            state = env._step_jit(state, env.dt, T_seg)
        nu_hist.append(float(env.compute_nu(state, T_seg)))
        act_hist.append(np.array(action))
    return np.array(nu_hist), np.stack(act_hist, axis=0)


def baseline_nu(env, ic_seed, n_actions):
    state = build_ic(env, ic_seed)
    for _ in range(env.warmup_steps):
        state = env._step_jit(state, env.dt, None)
    action_step = int(env.action_interval)
    nu_hist = []
    for _ in range(n_actions):
        for _ in range(action_step):
            state = env._step_jit(state, env.dt, None)
        nu_hist.append(float(env.compute_nu(state)))
    return np.array(nu_hist)


def make_neural_controller(env, actor_params, n_seg):
    @jax.jit
    def act(obs):
        out = mlp_forward(actor_params, obs)
        return jnp.tanh(out[:n_seg])
    def controller(state):
        obs = env._get_obs(state)
        return np.array(act(obs))
    return controller


def psd_total_power(action_hist, fs):
    mag = np.linalg.norm(action_hist, axis=1)
    nperseg = min(64, len(mag) // 4)
    f, P = welch(mag, fs=fs, nperseg=nperseg, detrend='constant')
    return float(np.trapezoid(P, f))


def main():
    results = {}
    for ra, cfg in CASES.items():
        print(f"\n{'='*60}\nRa={ra}\n{'='*60}")
        env = RBControlEnv(cfg['env_cfg'])
        n_seg = env.n_segments
        dt_ctrl = float(env.action_interval) * float(env.dt)
        fs = 1.0 / dt_ctrl

        # Baselines
        bls = []
        cut = int(0.2 * N_ACTIONS)
        print("  baselines (5 ICs):")
        for ic in TEST_ICS:
            t0 = time.time()
            nu_bl = baseline_nu(env, ic, N_ACTIONS)
            bl = float(np.mean(nu_bl[cut:]))
            bls.append(bl)
            print(f"    ic={ic}: Nu_bl={bl:.3f} [{time.time()-t0:.0f}s]")

        results[ra] = {'fs': fs, 'baseline_per_ic': bls,
                       'baseline_mean': float(np.mean(bls))}

        # Per-algo per-IC
        for algo in ['ppo', 'shac']:
            ck = find_ckpt(cfg[algo], algo)
            if ck is None:
                print(f"  {algo}: ckpt missing, skip")
                continue
            params, _ = load_params_flexible(ck)
            controller = make_neural_controller(env, params['actor'], n_seg)

            print(f"  {algo} (5 ICs):")
            results[ra].setdefault('checkpoints', {})[algo] = ck
            per_ic_data = []
            for ic_idx, ic in enumerate(TEST_ICS):
                t0 = time.time()
                nu_hist, act_hist = rollout(env, ic, N_ACTIONS, controller)
                nu_steady = float(np.mean(nu_hist[cut:]))
                pwr = psd_total_power(act_hist, fs)
                act_mag_mean = float(np.mean(np.linalg.norm(act_hist, axis=1)))
                act_mag_std  = float(np.std(np.linalg.norm(act_hist, axis=1)))
                red = (1 - nu_steady / bls[ic_idx]) * 100
                per_ic_data.append({
                    'ic': ic, 'nu_steady': nu_steady,
                    'reduction_pct': red,
                    'action_psd_total_power': pwr,
                    'action_mag_mean': act_mag_mean,
                    'action_mag_std': act_mag_std,
                })
                print(f"    ic={ic}: red={red:+.2f}%, pwr={pwr:.5f}, |a|={act_mag_mean:.3f}±{act_mag_std:.3f} [{time.time()-t0:.0f}s]")

            pwrs = [r['action_psd_total_power'] for r in per_ic_data]
            reds = [r['reduction_pct'] for r in per_ic_data]
            results[ra][algo] = {
                'per_ic': per_ic_data,
                'pwr_mean': float(np.mean(pwrs)),
                'pwr_std':  float(np.std(pwrs)),
                'red_mean': float(np.mean(reds)),
                'red_std':  float(np.std(reds)),
            }
            print(f"  {algo} 5-IC: action_pwr = {np.mean(pwrs):.5f} ± {np.std(pwrs):.5f} | "
                  f"red = {np.mean(reds):+.2f} ± {np.std(reds):.2f}%")

    out = os.path.join(_root, 'results', 'verification', 'v09_action_power_multiic.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
