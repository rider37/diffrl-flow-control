"""Peak GPU memory and time per update of SHAC training (Supplementary Material).

Runs three SHAC epochs of one configuration of the article into a scratch
directory and records the JAX peak memory, the JIT compile time and the time per
epoch.  One configuration per process (clean peak counter):

    XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/checks/shac_cost.py cylinder re100
    (cylinder re100|re200|re400|re500, rayleigh_benard ra1e4|ra1e5|ra1e6)
Writes results/checks/shac_cost_<problem>_<case>.json
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
env_cfg, shac_cfg = load_config(problem, f'env_{tag}'), load_config(problem, f'shac_{tag}')
run = f'runs/{problem}/shac_{tag}_s42'          # production run (training time per epoch)
h = json.load(open(run + '/history.json'))
out = tempfile.mkdtemp(prefix=f'shac_cost_{case}_', dir=None)
shac_cfg.update({'seed': 42, 'n_epochs': 3, 'output_dir': out, 'save_every': 10**6, 'snapshot_every': 10**6})
if problem == 'cylinder':
    from diffrl_flow_control.cylinder.train_shac import train
else:
    from diffrl_flow_control.rayleigh_benard.train_shac import train
t0 = time.time(); train(env_cfg, shac_cfg); wall = time.time() - t0
hh = json.load(open(out + '/history.json'))
st = jax.devices()[0].memory_stats()
res = dict(case=case, device=jax.devices()[0].device_kind, source_run=run,
           grid=[env_cfg.get('Nx'), env_cfg.get('Ny')], horizon=shac_cfg.get('horizon'),
           segment_length=shac_cfg.get('segment_length'),
           peak_bytes_in_use=int(st.get('peak_bytes_in_use', -1)),
           peak_GB=st.get('peak_bytes_in_use', -1) / 1e9,
           jit_compile_time=hh.get('jit_compile_time'), epoch_times=hh.get('time'), wall_total=wall,
           production_jit_compile_time=h.get('jit_compile_time'),
           production_median_epoch_time=sorted(h['time'])[len(h['time']) // 2])
json.dump(res, open(f'{ROOT}/results/checks/shac_cost_{case}.json', 'w'), indent=1)
print(json.dumps({k: res[k] for k in ('case', 'device', 'peak_GB', 'jit_compile_time', 'epoch_times', 'production_median_epoch_time')}))
