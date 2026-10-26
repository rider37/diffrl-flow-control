"""Action-norm spectral power of the final RB policies on all five held-out ICs (Section 5.1).

Final SHAC and PPO policies of Table 4 (training seed 42).  200-action rollouts per IC.
Writes results/checks/action_power.json.
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


from scipy.signal import welch
from diffrl_flow_control.configs import load_config
ENV_RA1E4 = load_config('rayleigh_benard', 'env_ra1e4')
ENV_RA1E5 = load_config('rayleigh_benard', 'env_ra1e5')
ENV_RA1E6 = load_config('rayleigh_benard', 'env_ra1e6')
from diffrl_flow_control.rayleigh_benard.env import RBControlEnv
from diffrl_flow_control.rayleigh_benard.heldout import build_ic, load_params_flexible
from diffrl_flow_control.rayleigh_benard.ppo import mlp_forward

TEST_ICS = [100, 101, 102, 103, 104]
N_ACTIONS = 200  # training horizon

CASES = {
    '1e4': dict(env_cfg=ENV_RA1E4,
                shac='runs/rayleigh_benard/shac_ra1e4_s42',
                ppo='runs/rayleigh_benard/ppo_ra1e4_s42'),
    '1e5': dict(env_cfg=ENV_RA1E5,
                shac='runs/rayleigh_benard/shac_ra1e5_s42',
                ppo='runs/rayleigh_benard/ppo_ra1e5_s42'),
    '1e6': dict(env_cfg=ENV_RA1E6,
                shac='runs/rayleigh_benard/shac_ra1e6_s42',
                ppo='runs/rayleigh_benard/ppo_ra1e6_s42'),
}


def find_ckpt(d, algo):
    p = os.path.join(ROOT, d, 'checkpoints', 'final.npz')
    return p if os.path.exists(p) else None


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
            results[ra].setdefault('checkpoints', {})[algo] = os.path.relpath(ck, ROOT)
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

    out = os.path.join(ROOT, 'results', 'checks', 'action_power.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
