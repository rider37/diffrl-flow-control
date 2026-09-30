"""Across-IC coefficient of variation of the uncontrolled mean Nu (RB) and
drag (cylinder) over the five held-out ICs, in two windows: the early window
right after the fixed warm-up (first 1000/2000/5000 CFD steps for RB, 2000
steps for the cylinder) and the late evaluation window (last half of an
episode-length uncontrolled run).  Population SD over ICs.
Inputs: results/{rb,cyl}_heldout_matched_baselines.json.
Writes results/uncontrolled_cv_windows.json.
"""
import json, numpy as np

cv = lambda x: float(np.std(x) / np.mean(x))
rb = json.load(open('results/rb_heldout_matched_baselines.json'))
cy = json.load(open('results/cyl_heldout_matched_baselines.json'))
out = {'rb': {}, 'cyl': {}}
for ra, rs in rb.items():
    out['rb'][ra] = {'early': cv([r['bl_window'] for r in rs]), 'late': cv([r['last_half'] for r in rs])}
for Re, v in cy.items():
    out['cyl'][Re] = {'early': cv([x['bl2000'] for x in v.values()]), 'late': cv([x['last_half'] for x in v.values()])}
json.dump(out, open('results/uncontrolled_cv_windows.json', 'w'), indent=1)
for k, v in out.items():
    for c, d in v.items():
        print(k, c, f"early {d['early']:.2e}  late {d['late']:.2e}")
