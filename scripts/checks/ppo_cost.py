"""Peak GPU memory and time of one PPO update (Supplementary Material).

Runs one PPO update of one configuration of the article into a scratch directory
and records the JAX peak memory and the update time.  One configuration per
process:  python scripts/checks/ppo_cost.py cylinder re100
(cylinder re100|re400, rayleigh_benard ra1e4|ra1e5|ra1e6)
Writes results/checks/ppo_cost_<problem>_<case>.json
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json, time, tempfile
import jax
from diffrl_flow_control.configs import load_config
problem, tag = sys.argv[1], sys.argv[2]
case = f'{problem}_{tag}'
scratch = tempfile.mkdtemp(prefix=f'ppo_cost_{case}_', dir=None)
if problem == 'cylinder':
    from diffrl_flow_control.cylinder.ppo import train
else:
    from diffrl_flow_control.rayleigh_benard.ppo import train
env_cfg, ppo_cfg = load_config(problem, f'env_{tag}'), load_config(problem, f'ppo_{tag}')
run = f'configs/{problem}/ppo_{tag}.json'
ppo_cfg.update(n_updates=1, output_dir=scratch, save_every=10**6, snapshot_every=10**6, seed=42)
t0 = time.time(); train(env_cfg, ppo_cfg); wall = time.time() - t0
st = jax.devices()[0].memory_stats()
hh = json.load(open(scratch + '/history.json'))
res = dict(case=case, device=jax.devices()[0].device_kind, source=run, peak_GB=st.get('peak_bytes_in_use', -1) / 1e9,
           update_time=hh.get('time'), wall_total=wall)
json.dump(res, open(f'{ROOT}/results/checks/ppo_cost_{case}.json', 'w'), indent=1)
print(json.dumps(res))
