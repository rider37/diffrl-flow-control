"""Re-measure the Section 5.1 cylinder claims with the width/reward-matched SHAC:
Cd and Cl spectra of uncontrolled / SHAC / PPO on the held-out IC 100 at each Re.
Long controlled rollout (400 actions = 100 convective time units) with the
evaluation protocol (EMA smoothing, evaluation interval).  Reports, per Re:
  * shedding frequency St (Cl PSD peak) uncontrolled/SHAC/PPO
  * Cd PSD peak at 2 St (relative to uncontrolled) for SHAC and PPO
  * Cl amplitude (RMS) relative to uncontrolled  -> does the vortex street persist?
  * Cl phase coherence: autocorrelation envelope at lag 5 periods
Writes results/cyl_spectrum_check.json.
"""
import os, sys, json, glob, time, numpy as np, jax, jax.numpy as jnp
PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); DIFF = PKG
sys.path.insert(0, PKG); sys.path.insert(0, os.path.join(PKG, 'src'))
from cases.cylinder_afc import make_env_config
from jax_cfd.envs.cylinder_flow_env import CylinderFlowEnv
from experiments.afc_cylinder.train_ppo import mlp_forward_relu
from scipy.signal import welch
RE_GRID = {100:(768,144), 200:(768,144), 400:(1280,240), 500:(1280,240)}
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
IC = 100; N_ACT = 400
def shac_ck(Re,nx,ny,s): return f'{PKG}/experiments/afc_cylinder/output_wm_shac_re{Re}_{nx}x{ny}_s{s}/checkpoints/fans_final_epoch_1000.npz'
def ppo_ck(Re,nx,ny,s):
    if Re==500: return f'{DIFF}/experiments/afc_cylinder/output_eps075vel_ppo_re500_1280x240_s{s}/checkpoints/final_params.npz'
    dd=f'{DIFF}/experiments/afc_cylinder/output_eps075_ppo_re{Re}_{nx}x{ny}_s{s}/checkpoints'
    return f'{dd}/final_params.npz' if os.path.exists(f'{dd}/final_params.npz') else sorted(glob.glob(f'{dd}/ckpt_*.npz'))[-1]
def load_actor(p):
    d=np.load(p); f=d.files
    if 'actor_0_W' in f: return [(jnp.asarray(d[f'actor_{i}_W']),jnp.asarray(d[f'actor_{i}_b'])) for i in range(99) if f'actor_{i}_W' in f]
    return [(jnp.asarray(d[f'actor_{i}_0']),jnp.asarray(d[f'actor_{i}_1'])) for i in range(99) if f'actor_{i}_0' in f]
def build_ic(env, seed):
    from jax_cfd.envs.cylinder_flow_env import FluidState
    rng=np.random.default_rng(seed); state,_=env.reset(); return state   # warm state; perturbation applied inside eval scripts via build_perturbed_ic
sys.path.insert(0, os.path.join(PKG,'scripts'))
from cyl_eval_common import build_perturbed_ic
out={}
for Re,(nx,ny) in RE_GRID.items():
    cfg=make_env_config(Re=Re,cells=(nx,ny),ibm_eps_factor=0.75); cfg['jet_method']='surface_bc'; cfg['soft_reset_prob']=0.0; cfg['obs_type']='velocity'
    env=CylinderFlowEnv(cfg); ai=int(env.action_interval); ema=float(env.ema_alpha); dt=env.dt
    t0=time.time(); st=build_perturbed_ic(env,IC)
    for _ in range(env.warmup_steps): st=env._step_jit(st,env.dt,0.0)
    warmed=st
    act={'shac':jax.jit(lambda a,o: jnp.tanh(mlp_forward_relu(a,o)[:1])),'ppo':jax.jit(lambda a,o: jnp.tanh(mlp_forward_relu(a,o)[:1]))}
    actors={'shac':load_actor(shac_ck(Re,nx,ny,SEED)),'ppo':load_actor(ppo_ck(Re,nx,ny,SEED))}
    series={}
    for name in ('uncontrolled','shac','ppo'):
        st=warmed; Q=0.0; cd=[]; cl=[]
        for k in range(N_ACT):
            a=0.0 if name=='uncontrolled' else float(act[name](actors[name],env._get_obs(st))[0])
            for _ in range(ai):
                Q=Q+ema*(a-Q); st=env._step_jit(st,env.dt,Q); cd.append(float(st.ibm_forces[0])/env.q_ref); cl.append(float(st.ibm_forces[1])/env.q_ref)
        series[name]=(np.array(cd),np.array(cl))
    np.savez(f'{DIFF}/results/cyl_spectrum_series_re{Re}_s{SEED}.npz', dt=dt, **{f'{k}_cd':v[0] for k,v in series.items()}, **{f'{k}_cl':v[1] for k,v in series.items()})
    res={}
    n=N_ACT*ai; skip=n//4
    for name,(cd,cl) in series.items():
        cdx=cd[skip:]-cd[skip:].mean(); clx=cl[skip:]-cl[skip:].mean()
        f,Pcl=welch(clx,fs=1/dt,nperseg=len(clx)//2); f2,Pcd=welch(cdx,fs=1/dt,nperseg=len(cdx)//2)
        band=(f>=0.1)&(f<=0.6); i=np.where(band)[0][np.argmax(Pcl[band])]; St=f[i]
        St_ref=res['uncontrolled']['St'] if 'uncontrolled' in res else St
        j=np.argmin(np.abs(f2-2*St_ref)); w=max(1,int(0.03/(f2[1]-f2[0]))); Pcd_peak=Pcd[max(0,j-w):j+w+1].max()
        # phase coherence: normalised Cl autocorrelation envelope at lag = 5 shedding periods
        T=int(round(1/St_ref/dt)); ac=np.correlate(clx,clx,'full')[len(clx)-1:]; ac/=ac[0]; lag=5*T; env5=np.abs(ac[max(0,lag-T//2):lag+T//2]).max() if lag<len(ac) else np.nan
        res[name]=dict(St=float(St),cl_rms=float(clx.std()),cd_mean=float(cd[skip:].mean()),Pcl_peak=float(Pcl[i]),Pcd_peak_2St=float(Pcd_peak),ac_env_5T=float(env5))
    b=res['uncontrolled']
    for name in ('shac','ppo'):
        r=res[name]; r['Pcd_peak_rel']=r['Pcd_peak_2St']/b['Pcd_peak_2St']; r['cl_rms_rel']=r['cl_rms']/b['cl_rms']; r['drag_red_pct']=(1-r['cd_mean']/b['cd_mean'])*100
    out[Re]=res
    print(f"Re={Re} [{(time.time()-t0)/60:.1f} min]  St bl/SHAC/PPO={b['St']:.3f}/{res['shac']['St']:.3f}/{res['ppo']['St']:.3f} | Cd-peak@2St rel SHAC {res['shac']['Pcd_peak_rel']:.2f} PPO {res['ppo']['Pcd_peak_rel']:.2f} | Cl-RMS rel SHAC {res['shac']['cl_rms_rel']:.2f} PPO {res['ppo']['cl_rms_rel']:.2f} | AC env(5T) bl {b['ac_env_5T']:.2f} SHAC {res['shac']['ac_env_5T']:.2f} PPO {res['ppo']['ac_env_5T']:.2f} | drag red SHAC {res['shac']['drag_red_pct']:.1f} PPO {res['ppo']['drag_red_pct']:.1f}", flush=True)
os.makedirs(f'{DIFF}/results',exist_ok=True); json.dump(out,open(f'{DIFF}/results/cyl_spectrum_check_s{SEED}.json','w'),indent=1)
