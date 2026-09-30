"""Width-matched RB SHAC (Ra=1e5 or 1e6, one seed).

Identical to the published phase3 RB SHAC recipe -- env_config and fans_config
loaded field-for-field from configs/rb_ra{ra}.json (copied from the published
run's history.json: horizon 30/50, segment 20/30, actor_lr 3e-4/1e-4, max_grad_norm
5/3, 1000 epochs, reset_every 10, IC pool 0..14) -- with ONE change: hidden
widths [64,64] for both actor and critic, equal to PPO (Markmann et al.).
Published SHAC used [96,96] at Ra=1e5 and [128,128] at Ra=1e6.  Ra=1e4 already
uses [64,64] and is not re-run.

  python scripts/train_wm_shac_rb.py --ra 1e5 --seed 42
"""
import os, sys, time, json, argparse
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src')); os.chdir(ROOT)
from experiments.rb_control.train_fans import train as train_fans

HIDDEN = [64, 64]                      # = PPO width (Markmann et al.)
TRAIN_POOL = list(range(15))           # IC pool 0..14 (test pool 100..104 disjoint)
FINAL_CKPT = 'fans_final_epoch_1000.npz'
INT_KEYS = {'Nx', 'Ny', 'n_probes_x', 'n_probes_y', 'n_segments', 'warmup_steps', 'baseline_steps',
            'episode_actions', 'action_interval', 'horizon', 'segment_length', 'n_epochs', 'reset_every',
            'save_every', 'snapshot_every', 'log_every', 'critic_update_epochs', 'seed'}


def out_dir(ra, seed):
    return f'experiments/rb_control/output_wm_shac_ra{ra}_s{seed}'


def _clean(cfg):
    return {k: (int(v) if k in INT_KEYS and isinstance(v, float) else v) for k, v in cfg.items()}


def run(ra, seed, n_epochs=1000, force=False):
    od = out_dir(ra, seed)
    if os.path.isfile(os.path.join(od, 'checkpoints', FINAL_CKPT)) and not force:
        print(f'[SKIP] wm_shac_ra{ra}_s{seed} -- final checkpoint exists', flush=True); return
    c = json.load(open(os.path.join(ROOT, 'configs', f'rb_ra{ra}.json')))
    env_cfg = _clean(c['env_config'])
    fans_cfg = _clean(c['fans_config'])
    fans_cfg.update({'seed': seed, 'n_epochs': n_epochs, 'output_dir': od,
                     'hidden_dims': HIDDEN, 'ic_pool_seeds': TRAIN_POOL})
    print(f"\n{'='*72}\nWM-SHAC RB  Ra={ra}  grid={env_cfg['Nx']}x{env_cfg['Ny']}  horizon={fans_cfg['horizon']}  "
          f"hidden={HIDDEN}  seed={seed}  n_epochs={n_epochs}\n{'='*72}", flush=True)
    t0 = time.time()
    train_fans(env_cfg, fans_cfg)
    print(f"WM-SHAC Ra={ra} s{seed} done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ra', required=True, choices=['1e5', '1e6'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--n_epochs', type=int, default=1000)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(); run(a.ra, a.seed, a.n_epochs, a.force)
