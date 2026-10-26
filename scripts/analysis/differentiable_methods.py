"""Held-out comparison of SHAC with full-episode BPTT and GI-PPO (Section 4.4, Figs. 5 and 6).

Reads the per-IC held-out values written by scripts/evaluate/cylinder_heldout.py and
scripts/evaluate/rayleigh_benard_heldout.py (run them with --algo bptt gippo --seeds 42,
and for the Fig. 6 sweep with --ra 1e5 --seeds 42 --algo shac_ep120 shac_ep300 shac_ep480 shac_ep960
bptt_ep120 bptt_ep300 bptt_ep480 bptt_ep960 bptt_ep300_u120 bptt_ep480_u120 bptt_ep960_u120,
and for GI-PPO at SHAC's CFD budget with --algo gippo --seeds 42 --checkpoint epoch_0010 and
epoch_0020) and the matched uncontrolled references, as in scripts/analysis/heldout_statistics.py.
Reductions are positive for a decrease of Cd or Nu.  SHAC uses its five training seeds,
full-episode BPTT and GI-PPO training seed 42.  Reports, for every configuration, the
shortfall of each method from the best of the three, GI-PPO at its last checkpoint within
SHAC's CFD budget, and the training-episode-length runs (Fig. 6 at Ra = 1e5, with
--ra 1e4 --algo bptt_ep60 bptt_ep100 bptt_ep100_u333 bptt_ep200_u333 for Appendix G at Ra = 1e4).
Writes results/statistics/differentiable_methods.json.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json
import numpy as np
from diffrl_flow_control.configs import load_config

SEEDS = ['42', '123', '456', '517', '1223']
METHODS = ('shac', 'bptt', 'gippo')


def reductions(values, base):
    return float(((base - np.asarray(values)) / base * 100).mean())


def gippo_within_budget(problem, case):
    """Last saved GI-PPO checkpoint (every 10 epochs) whose CFD steps do not exceed SHAC's budget."""
    s = load_config(problem, f'shac_{case}')
    budget = s['n_epochs'] * s['horizon'] * s['action_interval']
    steps = json.load(open(f'runs/{problem}/gippo_{case}_s42/history.json'))['total_cfd_steps']
    within = [e for e in range(10, len(steps), 10) if steps[e - 1] <= budget]
    return (f'epoch_{within[-1]:04d}', steps[within[-1] - 1], budget) if within else None


def at_budget(problem, case, evaluation, field, base):
    w = gippo_within_budget(problem, case)
    if w is None or f'gippo@{w[0]}' not in evaluation:
        return None
    return {'checkpoint': w[0], 'cfd_steps': w[1], 'shac_cfd_steps': w[2],
            'reduction': reductions(evaluation[f'gippo@{w[0]}']['42'][field], base)}


out = {'configurations': {}, 'gippo_at_shac_budget': {}}
cy = json.load(open('results/evaluation/cylinder_heldout.json'))
cm = json.load(open('results/evaluation/cylinder_uncontrolled.json'))
for Re in ('100', '200', '400', '500'):
    base = np.array([cm[Re][i]['last_half'] for i in sorted(cm[Re])])
    o = {}
    for m in METHODS:
        seeds = SEEDS if m == 'shac' else ['42']
        r = [reductions(cy[Re][m][s]['cd_ctrl_per_ic'], base) for s in seeds]
        o[m] = {'seeds': seeds, 'reduction': float(np.mean(r)), 'per_seed': r}
    out['configurations'][f'cylinder_re{Re}'] = o
    out['gippo_at_shac_budget'][f'cylinder_re{Re}'] = at_budget('cylinder', f're{Re}', cy[Re], 'cd_ctrl_per_ic', base)
rb = json.load(open('results/evaluation/rayleigh_benard_heldout.json'))
rbm = json.load(open('results/evaluation/rayleigh_benard_uncontrolled.json'))
for ra in ('1e4', '1e5', '1e6'):
    base = np.array([r['last_half'] for r in rbm[ra]])
    o = {}
    for m in METHODS:
        seeds = SEEDS if m == 'shac' else ['42']
        r = [reductions(rb[ra][m][s]['nu_late_per_ic'], base) for s in seeds]
        o[m] = {'seeds': seeds, 'reduction': float(np.mean(r)), 'per_seed': r}
    out['configurations'][f'rayleigh_benard_ra{ra}'] = o
    out['gippo_at_shac_budget'][f'rayleigh_benard_ra{ra}'] = at_budget('rayleigh_benard', f'ra{ra}', rb[ra], 'nu_late_per_ic', base)

shortfall = {m: [] for m in METHODS}
for c, o in out['configurations'].items():
    best = max(o[m]['reduction'] for m in METHODS)
    for m in METHODS:
        o[m]['shortfall_pp'] = best - o[m]['reduction']
        shortfall[m].append(o[m]['shortfall_pp'])
    print(f"{c:22s} " + '  '.join(f"{m} {o[m]['reduction']:5.1f}" for m in METHODS))
out['max_shortfall_pp'] = {m: float(max(v)) for m, v in shortfall.items()}
print('largest shortfall from the best method (pp):', {m: round(v, 1) for m, v in out['max_shortfall_pp'].items()})
print('GI-PPO within SHAC\'s CFD budget:', {c: (v['checkpoint'], round(v['reduction'], 1)) if v else None
                                         for c, v in out['gippo_at_shac_budget'].items()})

for ra, default in (('1e5', '250'), ('1e4', '200')):
    # training-episode length, seed 42: Fig. 6 at Ra = 1e5 and Appendix G at Ra = 1e4
    base = np.array([r['last_half'] for r in rbm[ra]])
    o = out[f'episode_length_ra{ra}'] = {}
    for algo, d in sorted(rb.get(ra, {}).items()):
        if '_ep' in algo and '42' in d:
            o[algo] = reductions(d['42']['nu_late_per_ic'], base)
    if '42' in rb.get(ra, {}).get('bptt', {}):     # BPTT on the environment episode
        o['bptt_default'] = reductions(rb[ra]['bptt']['42']['nu_late_per_ic'], base)
    print(f'episode length, Ra = {ra} (seed 42; BPTT default episode {default} actions):',
          {k: round(v, 1) for k, v in o.items()})
os.makedirs('results/statistics', exist_ok=True)
json.dump(out, open('results/statistics/differentiable_methods.json', 'w'), indent=1)
