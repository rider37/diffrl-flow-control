"""Time-to-target (Appendix D) recomputed with the width-matched SHAC runs.
Unsmoothed per-seed training-IC reduction interpolated onto a common 4000-point
grid of cumulative CFD steps (range covered by all five seeds, no extrapolation),
averaged over seeds; first grid point at which the mean reaches each target."""
import json, glob, numpy as np
SEEDS=[42,123,456,517,1223]; TARGETS=[5,10,15]
def curves(paths):
    xs,ys=[],[]
    for p in paths:
        h=json.load(open(p)); xs.append(np.array(h['total_cfd_steps'],float)); ys.append(np.array(h['reduction_pct'],float))
    lo=max(x.min() for x in xs); hi=min(x.max() for x in xs); xg=np.linspace(lo,hi,4000)
    Y=np.array([np.interp(xg,x,y) for x,y in zip(xs,ys)]); return xg, Y.mean(0), Y
def ttt(xg,m):
    out={}
    for t in TARGETS:
        i=np.argmax(m>=t); out[t]= xg[i]/1e6 if m[i]>=t else None
    return out
RB='experiments/rb_control'
cases={'1e4':dict(shac=[f'{RB}/output_phase3_shac_ra1e4_s{s}/history.json' for s in SEEDS], ppo=[f'{RB}/output_phase2_ppo_ra1e4_s{s}/history.json' for s in SEEDS]),
       '1e5':dict(shac=[f'{RB}/output_wm_shac_ra1e5_s{s}/history.json' for s in SEEDS], ppo=[f'{RB}/output_phase3_ppo_ra1e5_s{s}/history.json' for s in SEEDS]),
       '1e6':dict(shac=[f'{RB}/output_wm_shac_ra1e6_s{s}/history.json' for s in SEEDS], ppo=[f'{RB}/output_phase3_ppo_ra1e6_s{s}/history.json' for s in SEEDS])}
res={}
for ra,c in cases.items():
    for algo,paths in c.items():
        xg,m,Y=curves(paths); t=ttt(xg,m)
        res[f'{ra}_{algo}']=dict(ttt={str(k):v for k,v in t.items()}, budget=xg.max()/1e6, peak=float(m.max()), final50=float(m[-int(4000*0.05):].mean()))
        print(f"Ra={ra} {algo:8s} 5/10/15%: {', '.join('--' if v is None else f'{v:.3f}' for v in t.values())} M | budget {xg.max()/1e6:.1f} M | peak {m.max():.1f}% | last5% {m[-200:].mean():.1f}%")
json.dump(res,open('results/time_to_target_wm.json','w'),indent=1)
# cylinder: steps at which the SHAC five-seed training mean first exceeds PPO's final training-curve value
print('--- cylinder: SHAC mean vs PPO final training value ---')
CY='experiments/afc_cylinder'; V2='experiments/afc_cylinder'
for re,g in [(100,'768x144'),(200,'768x144'),(400,'1280x240'),(500,'1280x240')]:
    ppo=[f'{CY}/output_eps075_ppo_re{re}_{g}_s{s}/history.json' for s in SEEDS] if re!=500 else [f'experiments/afc_cylinder/output_eps075vel_ppo_re500_1280x240_s{s}/history.json' for s in SEEDS]
    def red(p):
        h=json.load(open(p)); return np.array(h['total_cfd_steps'],float), (1-np.array(h['cd_mean'])/h['cd_baseline'])*100
    pf=np.mean([red(p)[1][-10:].mean() for p in ppo])
    xs,ys=zip(*[red(f'{V2}/output_wm_shac_re{re}_{g}_s{s}/history.json') for s in SEEDS]); xg=np.linspace(max(x.min() for x in xs),min(x.max() for x in xs),4000); m=np.array([np.interp(xg,x,y) for x,y in zip(xs,ys)]).mean(0)
    i=np.argmax(m>=pf); print(f"Re={re}: PPO final training mean {pf:.1f}%  SHAC mean exceeds it at {xg[i]/1e6:.3f} M" if m[i]>=pf else f"Re={re}: PPO final {pf:.1f}% never exceeded (SHAC peak {m.max():.1f})")
    res.setdefault('cyl_shac_exceeds_ppo_final', {})[str(re)] = dict(ppo_final_training_pct=float(pf), shac_first_exceed_Msteps=(float(xg[i]/1e6) if m[i]>=pf else None))
json.dump(res,open('results/time_to_target_wm.json','w'),indent=1)
