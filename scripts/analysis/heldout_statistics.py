"""Held-out statistics with the matched uncontrolled denominator (Tables 2 and 4).

For each held-out IC the uncontrolled reference is the mean over the same
last-half window of an uncontrolled run from the same equilibrated state
(results/{rb,cyl}_heldout_matched_baselines.json).  Controlled values are the
per-IC last-half means stored by scripts/evaluate/rayleigh_benard_heldout.py and
scripts/evaluate/cylinder_heldout.py, so no policy is re-run.  Reductions are positive
for a decrease of Nu or Cd; per seed they are averaged over the five ICs.
Writes results/statistics/heldout_statistics.json and
results/figure_data/heldout_reductions_per_seed.json (input of Fig. 11b).
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json, numpy as np
from scipy import stats

SEEDS = ['42', '123', '456', '517', '1223']


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b); va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb); df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = stats.t.ppf(0.975, df); g = a.mean() - b.mean()
    return dict(gap=float(g), lo=float(g - t * se), hi=float(g + t * se), df=float(df))


def summ(r):
    r = np.asarray(r); return dict(seeds=r.tolist(), mean=float(r.mean()), sd1=float(r.std(ddof=1)))


out = {'rb': {}, 'cyl': {}}; fig = {}
# RB: Nu reductions in %
rbm = json.load(open('results/evaluation/rayleigh_benard_uncontrolled.json'))
rb = json.load(open('results/evaluation/rayleigh_benard_heldout.json'))
for ra in ('1e4', '1e5', '1e6'):
    bl = np.array([r['last_half'] for r in rbm[ra]])
    o = {'nu_unctrl_ic': bl.tolist(), 'nu_unctrl_mean': float(bl.mean())}; fig[ra] = {'nu_unctrl_ic': bl.tolist()}
    for a in ('shac', 'ppo'):
        r = [float(np.mean((bl - np.array(rb[ra][a][s]['nu_late_per_ic'])) / bl * 100)) for s in SEEDS]
        o[a] = summ(r); o[a]['cv'] = float(np.std(r, ddof=1) / abs(np.mean(r)))
        fig[ra][a] = {'seeds': [int(s) for s in SEEDS], 'reduction_per_seed': r}
    o['welch_shac_minus_ppo'] = welch(o['shac']['seeds'], o['ppo']['seeds'])
    out['rb'][ra] = o
# Ra=1e6 PPO re-trained with observation normalisation (Section 5.4): final checkpoints,
# evaluated against the matched references by scripts/evaluate/rayleigh_benard_heldout_obsnorm.py
on = json.load(open('results/evaluation/rayleigh_benard_heldout_obsnorm.json'))
b1 = np.array(out['rb']['1e6']['nu_unctrl_ic'])
r = [float(((b1 - np.array(v['nu_late_per_ic'])) / b1 * 100).mean()) for v in on['per_seed'].values()]
out['rb']['1e6']['ppo_obsnorm'] = summ(r)
# Cylinder: IBM drag reductions in %
cm = json.load(open('results/evaluation/cylinder_uncontrolled.json'))
cy = json.load(open('results/evaluation/cylinder_heldout.json'))
for Re in ('100', '200', '400', '500'):
    ics = sorted(cm[Re]); b1 = np.array([cm[Re][i]['last_half'] for i in ics])
    o = {'cd_unctrl_ic': b1.tolist(), 'cd_unctrl_mean': float(b1.mean())}
    for a in ('shac', 'ppo'):
        r = [float(((b1 - np.array(cy[Re][a][s]['cd_ctrl_per_ic'])) / b1 * 100).mean()) for s in SEEDS]
        o[a] = summ(r)
    o['welch_shac_minus_ppo'] = welch(o['shac']['seeds'], o['ppo']['seeds'])
    out['cyl'][Re] = o
print('obsnorm PPO 1e6', out['rb']['1e6']['ppo_obsnorm'])
json.dump(out, open('results/statistics/heldout_statistics.json', 'w'), indent=1)
os.makedirs('results/figure_data', exist_ok=True)
json.dump(fig, open('results/figure_data/heldout_reductions_per_seed.json', 'w'), indent=1)
for k, v in out.items():
    for c, o in v.items():
        w = o['welch_shac_minus_ppo']
        print(k, c, f"SHAC {o['shac']['mean']:.2f}±{o['shac']['sd1']:.2f}  PPO {o['ppo']['mean']:.2f}±{o['ppo']['sd1']:.2f}  gap {w['gap']:+.2f} [{w['lo']:+.2f},{w['hi']:+.2f}]",
              f"CV ppo {o['ppo'].get('cv', float('nan')):.2f} shac {o['shac'].get('cv', float('nan')):.3f}" if k == 'rb' else '',
              'PPO seeds', np.round(o['ppo']['seeds'], 1), 'SHAC seeds', np.round(o['shac']['seeds'], 1))
