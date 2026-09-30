"""Peak GPU memory and per-update time of one production PPO update (companion of v03).

Cylinder: env_config/ppo_config from the final seed-42 run history; RB: PPO_V2_CONFIG with the
production episode length (the recipe of scripts/rb_ppo_expand.py).  n_updates is set to 1.
Writes results/verification/v03b_ppo_memory_time_<case>.json
"""
import os, sys, json, time, tempfile, ast
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); V2 = R
sys.path[:0] = [V2, V2 + '/src', V2 + '/scripts']
os.chdir(V2)
import jax
from train_wm_shac_cyl import _clean
case = sys.argv[1]
scratch = tempfile.mkdtemp(prefix=f'v03b_{case}_', dir=None)
if case.startswith('cyl'):
    Re = int(case[3:]); g = {100: '768x144', 200: '768x144', 400: '1280x240', 500: '1280x240'}[Re]
    run = (f'{R}/experiments/afc_cylinder/output_eps075_ppo_re{Re}_{g}_s42' if Re != 500
           else f'{R}/experiments/afc_cylinder/output_eps075vel_ppo_re500_{g}_s42')
    h = json.load(open(run + '/history.json'))
    lst = lambda c: {k: (ast.literal_eval(v) if isinstance(v, str) and v.startswith('[') else v) for k, v in c.items()}
    env_cfg, ppo_cfg = _clean(lst(h['env_config'])), _clean(lst(h['ppo_config']))
    from experiments.afc_cylinder.train_ppo import train
else:
    from experiments.rb_control.config import ENV_CONFIG, ENV_CONFIG_RA1E5, ENV_CONFIG_RA1E6, PPO_V2_CONFIG
    ra = case[2:]; env_cfg = dict({'1e4': ENV_CONFIG, '1e5': ENV_CONFIG_RA1E5, '1e6': ENV_CONFIG_RA1E6}[ra])
    ppo_cfg = dict(PPO_V2_CONFIG, episode_length=env_cfg['episode_actions'], ic_pool_seeds=list(range(15)))
    run = 'PPO_V2_CONFIG (scripts/rb_ppo_expand.py recipe)'
    from experiments.rb_control.train_ppo import train
ppo_cfg.update(n_updates=1, output_dir=scratch, save_every=10**6, snapshot_every=10**6, seed=42)
t0 = time.time(); train(env_cfg, ppo_cfg); wall = time.time() - t0
st = jax.devices()[0].memory_stats()
hh = json.load(open(scratch + '/history.json'))
res = dict(case=case, device=jax.devices()[0].device_kind, source=run, peak_GB=st.get('peak_bytes_in_use', -1) / 1e9,
           update_time=hh.get('time'), wall_total=wall)
json.dump(res, open(f'{R}/results/verification/v03b_ppo_memory_time_{case}.json', 'w'), indent=1)
print(json.dumps(res))
