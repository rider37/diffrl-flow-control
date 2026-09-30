"""Seed-by-seed cylinder force-spectrum statistics behind Section 5.1 (held-out IC 100,
400 actions, drag of Eq. 13, Welch spectra over the last 300 actions)."""
import json, numpy as np
from scipy.signal import welch
SEEDS = (42, 123, 456, 517, 1223); out = {}
for Re in (100, 200, 400, 500):
    out[Re] = {}
    for s in SEEDS:
        d = np.load(f'results/cyl_spectrum_series_re{Re}_s{s}.npz'); dt = float(d['dt'])
        def tail(k):
            x = d[k]; x = x[len(x) // 4:]; return x
        def spec(x):
            x = x - x.mean(); return welch(x, fs=1 / dt, nperseg=len(x) // 2)
        fb, Pb = spec(tail('uncontrolled_cd')); fl, Pl = spec(tail('uncontrolled_cl'))
        band = (fl >= 0.1) & (fl <= 0.6); St = fl[band][np.argmax(Pl[band])]
        j = np.argmin(np.abs(fb - 2 * St)); w = max(1, int(0.03 / (fb[1] - fb[0])))
        win = slice(max(0, j - w), j + w + 1); Pref = Pb[win].max()
        for a in ('shac', 'ppo'):
            f, P = spec(tail(f'{a}_cd')); m = (f >= 0.05) & (f <= 3.0); k = np.argmax(P[m])
            out[Re].setdefault(a, []).append(dict(
                seed=s, St_unctrl=float(St), P2St_rel=float(P[win].max() / Pref),
                Ppeak_rel=float(P[m][k] / Pref), fpeak_over_St=float(f[m][k] / St),
                cd_rms_rel=float(tail(f'{a}_cd').std() / tail('uncontrolled_cd').std()),
                cl_rms_rel=float(tail(f'{a}_cl').std() / tail('uncontrolled_cl').std()),
                cl_mean=float(tail(f'{a}_cl').mean()),
                cl_p99_rel=float(np.percentile(np.abs(tail(f'{a}_cl') - tail(f'{a}_cl').mean()), 99)
                                 / np.percentile(np.abs(tail('uncontrolled_cl') - tail('uncontrolled_cl').mean()), 99))))
json.dump(out, open('results/cyl_drag_spectrum_stats.json', 'w'), indent=1)
for Re, v in out.items():
    for a, r in v.items():
        print(Re, a, *[f"{k}=" + str([round(x[k], 3) for x in r]) for k in ('P2St_rel', 'Ppeak_rel', 'fpeak_over_St', 'cd_rms_rel', 'cl_rms_rel', 'cl_mean', 'cl_p99_rel')])
