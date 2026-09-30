"""Held-out evaluation of the Ra=1e6 PPO policies trained with running observation
normalisation (Section 5.4), final checkpoints ckpt_0100.npz, held-out ICs 100-104.

The normalisation statistics saved in each checkpoint are applied frozen.  Records
the last-half mean Nu per IC, its reduction against the matched uncontrolled
reference of Table 4, and the pre-tanh policy mean (median |mu|, fraction |mu|>2).
Writes results/verification/v12_obsnorm_ppo_ra1e6_final.json.
"""
import os, sys, time, json
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.rb_control.config import ENV_CONFIG_RA1E6
from experiments.rb_control.train_ppo import mlp_forward
from jax_cfd.envs.rb_control_env import RBControlEnv
from rb_eval_common import build_ic, load_params_flexible

TEST_POOL = [100, 101, 102, 103, 104]
SEEDS = [42, 123, 456, 517, 1223]


def load_obs_rms(ckpt_path):
    d = np.load(ckpt_path)
    if 'obs_rms_mean' not in d.files:
        return None
    return (np.asarray(d['obs_rms_mean'], np.float64),
            np.asarray(d['obs_rms_var'], np.float64))


def eval_on_ic_normed(env, actor_params, obs_rms, seed, warmup, ep_actions, action_dim):
    mean, var = obs_rms
    state = build_ic(env, seed)
    for _ in range(warmup):
        state = env._step_jit(state, env.dt, None)

    @jax.jit
    def raw_mean(obs):
        return mlp_forward(actor_params, obs)  # pre-tanh mean

    ai = env.cfg.get('action_interval', 150)
    nus = np.empty(ep_actions)
    mus = []
    for a in range(ep_actions):
        obs = np.asarray(env._get_obs_jit(state))
        obs_n = np.clip((obs - mean) / np.sqrt(var + 1e-8), -10.0, 10.0).astype(np.float32)
        out = np.asarray(raw_mean(jnp.asarray(obs_n)))
        mus.append(out[:action_dim])
        action = jnp.tanh(jnp.asarray(out[:action_dim]))
        T_segs = env.map_action(action)
        for _ in range(ai):
            state = env._step_jit(state, env.dt, T_segs)
        nus[a] = float(env.compute_nu(state, T_segs))
    mus = np.abs(np.array(mus))
    return nus, mus


def main():
    cfg = {**ENV_CONFIG_RA1E6}
    env = RBControlEnv(cfg)
    warmup, ep_actions = cfg['warmup_steps'], cfg['episode_actions']
    action_dim = env.action_size

    # per-IC uncontrolled baselines (shared across seeds)
    # matched-window uncontrolled reference of Table 4
    mb = json.load(open('results/rb_heldout_matched_baselines.json'))['1e6']
    nu_bl = {r['ic']: r['last_half'] for r in mb}

    results = {}
    for seed in SEEDS:
        ckpt = (f'experiments/rb_control/output_obsnorm_ppo_ra1e6_s{seed}'
                f'/checkpoints/ckpt_0100.npz')   # final update, as for Table 4
        if not os.path.exists(ckpt):
            print(f"[s{seed}] MISSING {ckpt}"); continue
        obs_rms = load_obs_rms(ckpt)
        if obs_rms is None:
            print(f"[s{seed}] WARNING: no obs_rms in ckpt (norm not applied!)"); continue
        params, fmt = load_params_flexible(ckpt)
        print(f"\n[s{seed}] eval ({fmt})", flush=True)
        per_ic_red, nu_lates, mu_meds, mu_frac = [], [], [], []
        for s in TEST_POOL:
            t0 = time.time()
            nus, mus = eval_on_ic_normed(env, params['actor'], obs_rms, s,
                                         warmup, ep_actions, action_dim)
            nu_late = float(np.mean(nus[ep_actions // 2:]))
            red = (1.0 - nu_late / nu_bl[s]) * 100.0
            per_ic_red.append(red)
            nu_lates.append(nu_late)
            mu_meds.append(float(np.median(mus)))
            mu_frac.append(float(np.mean(mus > 2.0)))
            print(f"  IC {s}: Nu_late={nu_late:.3f} red={red:+.1f}% "
                  f"|mu|med={mu_meds[-1]:.2f} frac>2={mu_frac[-1]:.2f} "
                  f"[{(time.time()-t0)/60:.1f}min]", flush=True)
        results[f's{seed}'] = {
            'per_ic_reduction': per_ic_red,
            'mean_reduction_perIC': float(np.mean(per_ic_red)),
            'mu_median': float(np.mean(mu_meds)),
            'mu_frac_beyond2': float(np.mean(mu_frac)),
            'nu_late_per_ic': nu_lates, 'mu_median_per_ic': mu_meds,
        }
        print(f"  >> s{seed}: red(per-IC)={results[f's{seed}']['mean_reduction_perIC']:+.1f}% "
              f"|mu|med={results[f's{seed}']['mu_median']:.2f}", flush=True)

    if results:
        m_pic = [v['mean_reduction_perIC'] for v in results.values()]
        m_pic_arr = np.array(m_pic)
        summary = {'checkpoint': 'ckpt_0100.npz', 'nu_baselines': nu_bl, 'per_seed': results,
                   'reduction_perIC_sd1': float(np.std(m_pic_arr, ddof=1)),
                   'reduction_perIC_mean': float(np.mean(m_pic)),
                   'reduction_perIC_std': float(np.std(m_pic))}
        os.makedirs('results', exist_ok=True)
        with open('results/verification/v12_obsnorm_ppo_ra1e6_final.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n=== obs-norm PPO Ra=1e6 held-out ({len(results)} seed) ===")
        print(f"  per-IC : {summary['reduction_perIC_mean']:+.1f} +/- "
              f"{summary['reduction_perIC_std']:.1f}%")
        print("Saved results/verification/v12_obsnorm_ppo_ra1e6_final.json")


if __name__ == '__main__':
    main()
