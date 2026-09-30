"""RB Ra=1e4 SHAC, 1000 epochs, one seed.

Byte-identical to the canonical `output_phase3_shac_ra1e4_*` runs: the config
below is copied field-for-field from that run's saved history.json
(env_config + fans_config), i.e. base ENV_CONFIG (Ra=1e4, 96x64) and base
FANS_CONFIG with the exact overrides the canonical launcher applied
(n_epochs=1000, reset_every=10, save_every=100, snapshot_every=25,
ic_pool_seeds=0..14).  Output dir matches the canonical name so it is a
drop-in.

  python scripts/train_rb1e4_shac1000.py --seed 42
"""
import os, sys, time, argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)  # output_dir is relative to repo root

from experiments.rb_control.config import ENV_CONFIG, FANS_CONFIG
from experiments.rb_control.train_fans import train as train_fans

TRAIN_POOL = list(range(15))   # IC pool seeds 0..14 (test pool 100..104 is disjoint)
FINAL_CKPT = 'fans_final_epoch_1000.npz'


def out_dir(seed):
    return f'experiments/rb_control/output_phase3_shac_ra1e4_s{seed}'


def run(seed, force=False):
    od = out_dir(seed)
    done = os.path.isfile(os.path.join(ROOT, od, 'checkpoints', FINAL_CKPT))
    if done and not force:
        print(f'[SKIP] rb_1e4_shac_s{seed} — {od}/checkpoints/{FINAL_CKPT} exists', flush=True)
        return

    env_cfg = {**ENV_CONFIG}                 # Ra=1e4, 96x64, dt=0.01, warmup 3000
    fans_cfg = {
        **FANS_CONFIG,                       # horizon 20, [64,64], lrs, gamma, max_grad_norm 5 ...
        'seed': seed,
        'n_epochs': 1000,                    # <- 1000 epochs (canonical)
        'reset_every': 10,
        'save_every': 100,
        'snapshot_every': 25,
        'ic_pool_seeds': TRAIN_POOL,
        'output_dir': od,
    }
    print(f"\n{'='*70}\nRB Ra=1e4 SHAC  seed={seed}  n_epochs=1000  grid=96x64\n{'='*70}", flush=True)
    t0 = time.time()
    train_fans(env_cfg, fans_cfg)
    print(f"RB 1e4 SHAC s{seed} done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--force', action='store_true', help='retrain even if final checkpoint exists')
    a = ap.parse_args()
    run(a.seed, a.force)
