"""Measure SHAC and PPO action time series a(t) for ALL 5 seeds at each Re, so a
REPRESENTATIVE (median-RMS) rollout can be chosen per algorithm for the overlay.
Same env (velocity, eps=0.75), warm IC 100 once per Re, relu forward.

Writes: action_both_vs_re.npz  (all-seed a(t) + per-seed mean|a|/RMS for both algos)
"""
import os, sys, glob
import numpy as np
import jax, jax.numpy as jnp

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); DIFF = PKG
sys.path.insert(0, PKG); sys.path.insert(0, os.path.join(PKG, 'src'))
from cases.cylinder_afc import make_env_config
from jax_cfd.envs.cylinder_flow_env import CylinderFlowEnv, FluidState
from experiments.afc_cylinder.train_ppo import mlp_forward_relu

RE_GRID = {100:(768,144), 200:(768,144), 400:(1280,240), 500:(1280,240)}
SEEDS = [42,123,456,517,1223]
IC, EP = 100, 80


def build_ic(env, seed):
    rng = np.random.default_rng(seed); cfg = env.cfg; grid = env.grid; U = cfg['U_inf']
    Um = cfg.get('Um',1.5); Ly = grid.domain_size[1]; dy = Ly/grid.shape[1]
    y = jnp.arange(grid.shape[1])*dy+dy/2
    u = jnp.broadcast_to((Um*4.0*y*(Ly-y)/Ly**2)[None,:], grid.face_shape(0))
    v = jnp.zeros(grid.face_shape(1))
    xv,yv = grid.face_centers(1); Xv,Yv = jnp.meshgrid(xv,yv,indexing='ij')
    px = cfg['cx']+rng.uniform(1.0,3.0)*cfg['R']; py = cfg['cy']+rng.uniform(-1.0,1.0)*cfg['R']
    r2 = ((Xv-px)/cfg['R'])**2+((Yv-py)/cfg['R'])**2
    v = v+rng.choice([-1.0,1.0])*rng.uniform(0.03,0.08)*U*jnp.exp(-r2/0.5)
    return FluidState(velocity=(u,v), pressure=jnp.zeros(grid.cell_shape()), scalars={}, ibm_forces=None)


def load_actor(p):
    d = np.load(p); f = d.files
    if any(x.endswith('_W') for x in f):
        return [(jnp.asarray(d[f'actor_{i}_W']), jnp.asarray(d[f'actor_{i}_b'])) for i in range(99) if f'actor_{i}_W' in f]
    return [(jnp.asarray(d[f'actor_{i}_0']), jnp.asarray(d[f'actor_{i}_1'])) for i in range(99) if f'actor_{i}_0' in f]


def shac_ck(Re, nx, ny, s):
    return f'{PKG}/experiments/afc_cylinder/output_wm_shac_re{Re}_{nx}x{ny}_s{s}/checkpoints/fans_final_epoch_1000.npz'
def ppo_ck(Re, nx, ny, s):
    if Re == 500:
        return f'{DIFF}/experiments/afc_cylinder/output_eps075vel_ppo_re500_1280x240_s{s}/checkpoints/final_params.npz'
    dd = f'{DIFF}/experiments/afc_cylinder/output_eps075_ppo_re{Re}_{nx}x{ny}_s{s}/checkpoints'
    if os.path.exists(f'{dd}/final_params.npz'): return f'{dd}/final_params.npz'
    g = sorted(glob.glob(f'{dd}/ckpt_*.npz')); return g[-1] if g else None


act_fn = jax.jit(lambda a,o: jnp.tanh(mlp_forward_relu(a,o)[:1]))
save = {}

for Re,(nx,ny) in RE_GRID.items():
    cfg = make_env_config(Re=Re, cells=(nx,ny), ibm_eps_factor=0.75)
    cfg['jet_method']='surface_bc'; cfg['soft_reset_prob']=0.0; cfg['obs_type']='velocity'
    env = CylinderFlowEnv(cfg); ai=int(env.action_interval); ema=float(env.ema_alpha)
    s0 = build_ic(env, IC)
    for _ in range(env.warmup_steps):
        s0 = env._step_jit(s0, env.dt, 0.0)
    warmed = s0
    for algo, ckf in (('shac', shac_ck), ('ppo', ppo_ck)):
        A = np.full((len(SEEDS), EP), np.nan)
        for j, s in enumerate(SEEDS):
            ck = ckf(Re, nx, ny, s)
            if ck is None or not os.path.exists(ck):
                print(f"  {algo} Re{Re} s{s}: MISSING", flush=True); continue
            actor = load_actor(ck); st=warmed; Q=0.0
            for k in range(EP):
                a = float(act_fn(actor, env._get_obs(st))[0]); A[j,k]=a
                for _ in range(ai):
                    Q=Q+ema*(a-Q); st=env._step_jit(st, env.dt, Q)
        save[f'{algo}_re{Re}'] = A
        late = A[:, EP//2:]
        rms = np.nanstd(late, axis=1); ma = np.nanmean(np.abs(late), axis=1)
        med = int(np.argsort(rms)[len(rms)//2])
        print(f"{algo} Re={Re}: median-RMS seed idx={med} (s{SEEDS[med]}) rms={rms[med]:.3f}  |a|={ma[med]:.3f}", flush=True)

np.savez(f'{PKG}/results/figure_data/action_both_vs_re.npz', seeds=np.array(SEEDS), Re=np.array(list(RE_GRID)), **save)
print("\nsaved action_both_vs_re.npz", flush=True)
