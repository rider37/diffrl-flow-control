"""Width-matched cylinder SHAC (one Re, one seed).

Identical to the published (relu, eps=0.75, velocity-obs) cylinder SHAC recipe
-- env_config and fans_config are loaded field-for-field from that run's saved
history.json (configs/cyl_re{Re}.json) -- with ONE change: the actor/critic
widths are set to PPO's Rabault configuration, actor [512,512] / critic [32,32]
(published SHAC used [256,256] / [256,256]), and the training rollout reproduces the
environment reward path used by PPO: EMA action smoothing (ema_alpha), interval-mean
Cd and |mean Cl| in the reward, and cl_weight = env cl_weight (0.2; published SHAC used 0.1
and end-of-interval instantaneous Cd/Cl).

  python scripts/train_wm_shac_cyl.py --Re 100 --seed 42
"""
import os, sys, time, json, argparse
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src')); os.chdir(ROOT)
from experiments.afc_cylinder.train_fans import train as train_fans_cyl

ACTOR_DIMS, CRITIC_DIMS = [512, 512], [32, 32]      # = PPO (Rabault 2019)
GRID_BY_RE = {100: (768, 144), 200: (768, 144), 400: (1280, 240), 500: (1280, 240)}
FINAL_CKPT = 'fans_final_epoch_1000.npz'


def out_dir(Re, seed):
    g = GRID_BY_RE[Re]
    return f'experiments/afc_cylinder/output_wm_shac_re{Re}_{g[0]}x{g[1]}_s{seed}'


def _clean(cfg):
    """history.json stores ints as floats; restore ints where the code expects them."""
    out = {}
    for k, v in cfg.items():
        if isinstance(v, float) and v.is_integer() and k not in ('actor_lr', 'critic_lr', 'gamma', 'dt',
                                                                  'cl_weight', 'ema_alpha', 'Q_max', 'jet_width',
                                                                  'ibm_eps_factor', 'soft_reset_prob', 'U_inf',
                                                                  'Um', 'R', 'cx', 'cy', 'Lx', 'Ly', 'Re',
                                                                  'max_grad_norm', 'action_penalty'):
            out[k] = int(v)
        elif isinstance(v, list):
            out[k] = [int(x) if isinstance(x, float) and x.is_integer() else x for x in v]
        else:
            out[k] = v
    return out


def run(Re, seed, n_epochs=1000, force=False):
    od = out_dir(Re, seed)
    if os.path.isfile(os.path.join(od, 'checkpoints', FINAL_CKPT)) and not force:
        print(f'[SKIP] wm_shac_re{Re}_s{seed} -- final checkpoint exists', flush=True); return
    c = json.load(open(os.path.join(ROOT, 'configs', f'cyl_re{Re}.json')))
    env_cfg = _clean(c['env_config'])
    fans_cfg = _clean(c['fans_config'])
    fans_cfg.update({'seed': seed, 'n_epochs': n_epochs, 'output_dir': od,
                     'actor_hidden_dims': ACTOR_DIMS, 'critic_hidden_dims': CRITIC_DIMS,
                     'hidden_dims': ACTOR_DIMS,
                     'cl_weight': float(env_cfg['cl_weight'])})   # = PPO/env lift penalty (0.2)
    print(f"\n{'='*72}\nWM-SHAC cylinder  Re={Re}  grid={env_cfg['Nx']}x{env_cfg['Ny']}  eps={env_cfg['ibm_eps_factor']}  "
          f"obs={env_cfg['obs_type']}  actor={ACTOR_DIMS} critic={CRITIC_DIMS}  seed={seed}  n_epochs={n_epochs}\n{'='*72}", flush=True)
    t0 = time.time()
    train_fans_cyl(env_cfg, fans_cfg)
    print(f"WM-SHAC Re={Re} s{seed} done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--Re', type=int, required=True, choices=sorted(GRID_BY_RE))
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--n_epochs', type=int, default=1000)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(); run(a.Re, a.seed, a.n_epochs, a.force)
