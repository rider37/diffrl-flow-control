"""Configurations of the runs in the article (configs/<problem>/<name>.json).

The files hold the environment and algorithm settings stored in the training
histories of the final runs; ``seed`` and ``output_dir`` are set per run.
"""
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_config(problem: str, name: str) -> dict:
    """Load configs/<problem>/<name>.json, e.g. load_config('cylinder', 'env_re100')."""
    with open(os.path.join(REPO_ROOT, 'configs', problem, name + '.json')) as f:
        return json.load(f)


def run_dir(problem: str, name: str, seed: int) -> str:
    """Directory of one training run, e.g. runs/cylinder/shac_re100_s42."""
    return os.path.join('runs', problem, f'{name}_s{seed}')
