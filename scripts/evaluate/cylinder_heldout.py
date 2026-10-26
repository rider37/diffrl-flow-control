"""Held-out evaluation of the final cylinder policies (Table 2).

Each held-out IC (seeds 100-104) is warmed once per Re and every SHAC and PPO
policy runs deterministically from that state (tanh of the relu actor output)
for 80 actions at the environment action interval with the EMA action filter.
The stored quantity is the drag coefficient of the IBM force, averaged over the
last half of the episode.  Reductions are computed against the matched
uncontrolled references by scripts/analysis/heldout_statistics.py.

Final checkpoints (seeds 42 123 456 517 1223, interface eps = 0.75):
  runs/cylinder/{shac,ppo}_re{Re}_s{seed}/checkpoints/final.npz

Usage: python scripts/evaluate/cylinder_heldout.py [--Re 100 200 400 500] [--algo shac ppo] [--seeds ...]
                                                    [--checkpoint epoch_0020]
Writes results/evaluation/cylinder_heldout.json: {Re: {algo: {seed: {ckpt, cd_ctrl_per_ic}}}};
a checkpoint other than final.npz is stored under the key algo@<checkpoint>.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import time, json, argparse
import numpy as np
import jax, jax.numpy as jnp


from diffrl_flow_control.configs import load_config
from diffrl_flow_control.cylinder.env import CylinderFlowEnv
from diffrl_flow_control.cylinder.ppo import mlp_forward_relu
from diffrl_flow_control.cylinder.heldout import build_perturbed_ic, load_actor, TEST_POOL, EPISODE_ACTIONS, SEEDS, RE_GRID

OUT = 'results/evaluation/cylinder_heldout.json'


def ckpt(Re, algo, s, name='final'):
    return f'runs/cylinder/{algo}_re{Re}_s{s}/checkpoints/{name}.npz'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Re', type=int, nargs='+', default=sorted(RE_GRID))
    ap.add_argument('--algo', nargs='+', default=['shac', 'ppo'])
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--checkpoint', default='final', help='checkpoint name, e.g. epoch_0010 (default: final)')
    a = ap.parse_args()
    key = (lambda al: al) if a.checkpoint == 'final' else (lambda al: f'{al}@{a.checkpoint}')
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    act_fn = jax.jit(lambda p, o: jnp.tanh(mlp_forward_relu(p, o)[:1]))
    for Re in a.Re:
        cfg = load_config('cylinder', f'env_re{Re}')
        cfg['soft_reset_prob'] = 0.0; cfg['obs_type'] = 'velocity'
        env = CylinderFlowEnv(cfg)
        warmup = int(env.warmup_steps); ai = int(env.action_interval); ema = float(env.ema_alpha)
        actors = {(al, s): load_actor(ckpt(Re, al, s, a.checkpoint)) for al in a.algo for s in a.seeds}
        print(f"\n=== Re={Re} grid={RE_GRID[Re]} warmup={warmup} action interval={ai} ===", flush=True)
        cd = {k: [] for k in actors}
        for ic in TEST_POOL:
            t0 = time.time(); st = build_perturbed_ic(env, ic)
            for _ in range(warmup):
                st = env._step_jit(st, env.dt, 0.0)
            warmed = st
            for k, p in actors.items():
                st = warmed; Q = 0.0; c = np.empty(EPISODE_ACTIONS)
                for j in range(EPISODE_ACTIONS):
                    act = float(act_fn(p, env._get_obs(st))[0]); ci = []
                    for _ in range(ai):
                        Q = Q + ema * (act - Q); st = env._step_jit(st, env.dt, Q)
                        ci.append(float(st.ibm_forces[0]) / env.q_ref)
                    c[j] = np.mean(ci)
                cd[k].append(float(np.mean(c[EPISODE_ACTIONS // 2:])))
            print(f"  IC{ic} done [{(time.time()-t0)/60:.1f} min]", flush=True)
        for (al, s), v in cd.items():
            out.setdefault(str(Re), {}).setdefault(key(al), {})[str(s)] = {'ckpt': ckpt(Re, al, s, a.checkpoint), 'cd_ctrl_per_ic': v}
        os.makedirs('results', exist_ok=True)
        json.dump(out, open(OUT, 'w'), indent=1)
    print(f"Saved {OUT}")


if __name__ == '__main__':
    main()
