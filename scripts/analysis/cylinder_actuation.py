"""Data of Fig. 5: SHAC and PPO action time series a(t) of all five training seeds
at each Re on the held-out IC 100, from which the representative (median-RMS)
rollout of each algorithm is plotted, and the per-seed mean |a| and RMS.
Writes results/figure_data/fig05_cylinder_actuation.npz.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import glob
import numpy as np
import jax, jax.numpy as jnp

from diffrl_flow_control.configs import load_config
from diffrl_flow_control.cylinder.env import CylinderFlowEnv, FluidState
from diffrl_flow_control.cylinder.ppo import mlp_forward_relu

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
    return f'{ROOT}/runs/cylinder/shac_re{Re}_s{s}/checkpoints/final.npz'
def ppo_ck(Re, nx, ny, s):
    return f'{ROOT}/runs/cylinder/ppo_re{Re}_s{s}/checkpoints/final.npz'


act_fn = jax.jit(lambda a,o: jnp.tanh(mlp_forward_relu(a,o)[:1]))
save = {}

for Re,(nx,ny) in RE_GRID.items():
    cfg = load_config('cylinder', f'env_re{Re}')
    cfg['soft_reset_prob']=0.0; cfg['obs_type']='velocity'
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

np.savez(f'{ROOT}/results/figure_data/fig05_cylinder_actuation.npz', seeds=np.array(SEEDS), Re=np.array(list(RE_GRID)), **save)
print("\nsaved fig05_cylinder_actuation.npz", flush=True)
