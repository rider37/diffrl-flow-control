"""Peak GPU memory and per-update time of the production SHAC training step.

For one production configuration, loads env_config and fans_config from the
final seed-42 run's history.json, runs train() for 3 epochs into a scratch
directory, and records jax peak_bytes_in_use, JIT compile time and per-epoch
update time.  Run one configuration per process (clean peak counter):

  XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/verification/v03_shac_memory_time.py cyl100
Writes results/verification/v03_shac_memory_time_<case>.json
"""
import os, sys, json, time, tempfile
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); V2 = R
sys.path[:0] = [V2, V2 + '/src', V2 + '/scripts']
os.chdir(V2)
import jax
from train_wm_shac_cyl import _clean
CASES = {
    'cyl100': ('cyl', R + '/experiments/afc_cylinder/output_wm_shac_re100_768x144_s42'),
    'cyl200': ('cyl', R + '/experiments/afc_cylinder/output_wm_shac_re200_768x144_s42'),
    'cyl400': ('cyl', R + '/experiments/afc_cylinder/output_wm_shac_re400_1280x240_s42'),
    'cyl500': ('cyl', R + '/experiments/afc_cylinder/output_wm_shac_re500_1280x240_s42'),
    'rb1e4': ('rb', R + '/experiments/rb_control/output_phase3_shac_ra1e4_s42'),
    'rb1e5': ('rb', R + '/experiments/rb_control/output_wm_shac_ra1e5_s42'),
    'rb1e6': ('rb', R + '/experiments/rb_control/output_wm_shac_ra1e6_s42'),
}
case = sys.argv[1]; kind, run = CASES[case]
h = json.load(open(run + '/history.json'))
import ast
def _lists(c):
    return {k: (ast.literal_eval(v) if isinstance(v, str) and v.startswith('[') else v) for k, v in c.items()}
env_cfg, fans_cfg = _clean(_lists(h['env_config'])), _clean(_lists(h['fans_config']))
out = tempfile.mkdtemp(prefix=f'v03_{case}_', dir=None)
fans_cfg.update({'n_epochs': 3, 'output_dir': out, 'save_every': 10**6, 'snapshot_every': 10**6})
if kind == 'cyl':
    from experiments.afc_cylinder.train_fans import train
else:
    from experiments.rb_control.train_fans import train
t0 = time.time(); train(env_cfg, fans_cfg); wall = time.time() - t0
hh = json.load(open(out + '/history.json'))
st = jax.devices()[0].memory_stats()
res = dict(case=case, device=jax.devices()[0].device_kind, source_run=run,
           grid=[env_cfg.get('Nx'), env_cfg.get('Ny')], horizon=fans_cfg.get('horizon'),
           segment_length=fans_cfg.get('segment_length'),
           peak_bytes_in_use=int(st.get('peak_bytes_in_use', -1)),
           peak_GB=st.get('peak_bytes_in_use', -1) / 1e9,
           jit_compile_time=hh.get('jit_compile_time'), epoch_times=hh.get('time'), wall_total=wall,
           production_jit_compile_time=h.get('jit_compile_time'),
           production_median_epoch_time=sorted(h['time'])[len(h['time']) // 2])
json.dump(res, open(f'{R}/results/verification/v03_shac_memory_time_{case}.json', 'w'), indent=1)
print(json.dumps({k: res[k] for k in ('case', 'device', 'peak_GB', 'jit_compile_time', 'epoch_times', 'production_median_epoch_time')}))
