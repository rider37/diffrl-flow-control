"""
2D Cylinder AFC Benchmark — Rabault et al. (2019, JFM)

Usage:
    # Standalone CFD simulation (baseline, no jets)
    from cases.cylinder_afc import CONFIG
    from jax_cfd.runner import run
    run(CONFIG)

    # RL environment
    from cases.cylinder_afc import CONFIG, AFC_CONFIG
    from jax_cfd.envs import CylinderFlowEnv
    env = CylinderFlowEnv(make_env_config(Re=100))

    # Change Re
    from cases.cylinder_afc import make_config
    config_re400 = make_config(Re=400)
"""

from .domain import DOMAIN
from .physics import PHYSICS
from .control import CONTROL
from .afc import AFC

# Standard CFD config (for jax_cfd.runner — baseline simulation, no jets)
CONFIG = {**DOMAIN, **PHYSICS, **CONTROL}

# AFC-specific config (jets, probes, RL timing, reward)
AFC_CONFIG = AFC


def make_config(Re=100, cells=None):
    """Create CFD config with optional Re and grid override.

    Parameters
    ----------
    Re : int
        Reynolds number (100, 200, 400, 1000, ...).
    cells : tuple of int, optional
        Grid resolution (Nx, Ny). Default: (512, 96).
        Suggested: Re<=200 → (512, 96), Re=400 → (768, 144),
                   Re>=1000 → (1024, 192).
    """
    cfg = {**CONFIG, 'Re': Re}
    if cells is not None:
        cfg['domain'] = {**cfg['domain'], 'cells': list(cells)}
    # Update output dir per Re
    cfg['output'] = {**cfg['output'], 'dir': f'cases/cylinder_afc/output_re{Re}'}
    return cfg


def make_env_config(Re=100, cells=None, ibm_eps_factor=None):
    """Create RL environment config merging domain + physics + AFC.

    Returns dict compatible with CylinderFlowEnv(config=...).

    ibm_eps_factor : float or None
        IBM sigmoid half-width in units of dx.  None -> solver default
        (0.5*dx, original behavior).  0.75 -> benchmark-calibrated interface
        (Cd matches Schaefer-Turek at the converged grid; eps sweep 2026-06-05).
    """
    body = DOMAIN['bodies'][0]
    dom = DOMAIN['domain']
    _cells = list(cells) if cells else dom['cells']

    # Auto-scale dt to maintain CFL < 0.25 when grid is refined
    base_cells = dom['cells']  # [512, 96]
    base_dt = AFC['timing']['dt']  # 0.005
    grid_ratio = min(base_cells[0] / _cells[0], base_cells[1] / _cells[1])
    dt = base_dt * grid_ratio  # finer grid → smaller dt

    return {
        # Domain
        'Re': Re,
        'Lx': dom['size'][0],
        'Ly': dom['size'][1],
        'Nx': _cells[0],
        'Ny': _cells[1],
        'R': body['radius'],
        'cx': body['center'][0],
        'cy': body['center'][1],
        'U_inf': PHYSICS['reference']['U'],
        'n_ibm_pts': body['n_points'],

        # Inlet / wall (Rabault 2019: parabolic channel flow)
        'inlet_type': 'parabolic',
        'Um': 1.5,
        'wall_type': 'wall_no_slip',

        # Jets
        'n_jets': AFC['jets']['n_jets'],
        'jet_thetas': AFC['jets']['thetas'],
        'jet_width': AFC['jets']['width'],
        'Q_max': AFC['jets']['Q_max'],

        # RL timing (steps scale inversely with dt to keep same physical time)
        'dt': dt,
        'action_interval': int(AFC['timing']['action_interval'] / grid_ratio),
        'episode_actions': AFC['timing']['episode_length'],
        'warmup_steps': int(AFC['timing']['warmup_steps'] / grid_ratio),
        'baseline_steps': int(AFC['timing']['baseline_steps'] / grid_ratio),
        'ema_alpha': AFC['timing'].get('ema_alpha', 1.0),

        # IBM interface calibration (None -> 0.5*dx default; 0.75 -> benchmark)
        'ibm_eps_factor': ibm_eps_factor,

        # Probes (Rabault 2019: 151 pressure probes)
        'probe_layout': AFC['probes'].get('layout', 'grid'),
        'obs_type': AFC['probes'].get('obs_type', 'velocity'),

        # Reward
        'cl_weight': AFC['reward'].get('cl_weight', 0.2),
    }
