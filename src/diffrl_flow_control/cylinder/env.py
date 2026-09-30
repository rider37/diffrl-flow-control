"""Cylinder wake with two synthetic jets (Section 4.1).

Channel flow past a circular cylinder (Schafer--Turek geometry) with a parabolic
inlet, a zero-gradient outlet and no-slip channel walls.  The cylinder is a
ghost-cell immersed boundary; the two zero-net-mass-flux jets at 90 and 270
degrees are imposed as a Gaussian surface-velocity profile on the immersed
boundary.  The observation is the velocity at the 151 probes of Rabault et al.
(2019); the action is the jet amplitude in [-1, 1], applied through a
first-order filter within each action interval.
"""
import jax
import jax.numpy as jnp
import numpy as np

from diffrl_flow_control.solver.actuators import surface_constrained_params
from diffrl_flow_control.solver.boundaries import apply_boundaries, pad_velocity_for_diffusion
from diffrl_flow_control.solver.geometry import cylinder_coords_diff
from diffrl_flow_control.solver.grid import MACGrid
from diffrl_flow_control.solver.ibm import make_ghost_cell_ibm_fn
from diffrl_flow_control.solver.schemes import make_muscl_advection, solve_pressure_fft
from diffrl_flow_control.solver.state import FluidState
from diffrl_flow_control.solver.step import create_step_fn
from diffrl_flow_control.solver.viscosity import laminar_viscosity


class CylinderFlowEnv:
    """Cylinder wake with a zero-net-mass-flux jet pair.

    Observation: u and v at the 151 probes (302 values).
    Action: jet amplitude a in [-1, 1]; the jet velocity is a * Q_max.
    Reward: -(Cd - Cd_baseline) - cl_weight * |mean(Cl)| over the action interval.
    """

    def __init__(self, config):
        cfg = dict(config)
        self.cfg = cfg
        # fixed in this environment; the keys are kept in the configs for reference
        assert cfg.get('jet_method', 'surface_bc') == 'surface_bc'
        assert cfg.get('n_jets', 2) == 2
        assert cfg.get('wall_type', 'wall_no_slip') == 'wall_no_slip'

        Nx, Ny = cfg['Nx'], cfg['Ny']
        Lx, Ly = cfg['Lx'], cfg['Ly']
        Re = cfg['Re']
        R = cfg['R']
        cx, cy = cfg['cx'], cfg['cy']

        self.dt = cfg['dt']
        self.action_interval = cfg['action_interval']
        self.episode_actions = cfg['episode_actions']
        self.warmup_steps = cfg['warmup_steps']
        self.baseline_steps = cfg['baseline_steps']
        self.Q_max = cfg['Q_max']
        self.cl_weight = cfg['cl_weight']

        # first-order action filter (per CFD step)
        self.ema_alpha = cfg['ema_alpha']
        self.Q_actual = jnp.float32(0.0)

        self.grid = MACGrid((Nx, Ny), (Lx, Ly))
        nu = cfg['U_inf'] * (2 * R) / Re   # nu = U D / Re
        self.nu = nu

        self.jet_width = cfg['jet_width']
        jet_thetas = jnp.array(cfg['jet_thetas'])
        jet_center = jnp.array([cx, cy])
        positions, directions, _ = surface_constrained_params(jet_thetas, jet_center, R)
        self.jet_positions = positions
        self.jet_directions = directions

        # ghost-cell immersed boundary with interface half-width eps = ibm_eps_factor * dx
        coords, ds = cylinder_coords_diff(
            jnp.float32(R), (jnp.float32(cx), jnp.float32(cy)), cfg['n_ibm_pts'],
        )
        _ibm_eps = float(cfg['ibm_eps_factor']) * (cfg['Lx'] / cfg['Nx'])
        self._jet_surface_basis = self._precompute_jet_surface_basis()
        self.ibm_fn = make_ghost_cell_ibm_fn(
            self.grid, coords, ds, eps=_ibm_eps,
            jet_surface_basis=self._jet_surface_basis,
        )

        bc_cfg = {
            'x_min': {'type': 'inlet_parabolic', 'Um': cfg['Um']},
            'x_max': {'type': 'outlet_zero_gradient'},
            'y_min': {'type': 'wall_no_slip'},
            'y_max': {'type': 'wall_no_slip'},
        }
        self.bc_cfg = bc_cfg

        def apply_bc(vel, _cfg, g):
            return apply_boundaries(vel, bc_cfg, g)

        def pad_diff(vel, g):
            return pad_velocity_for_diffusion(vel, bc_cfg, g)

        self.step_fn = create_step_fn(
            self.grid, nu, apply_bc, pad_diff,
            laminar_viscosity, make_muscl_advection(()), solve_pressure_fft,
            ibm_fn=self.ibm_fn,
        )

        self._step_jit = jax.jit(self.step_fn)
        self._step_with_jets_jit = jax.jit(self._step_with_jets_pure)
        self._get_obs_jit = jax.jit(self._get_obs)

        self._setup_probes(cfg)

        q_ref = 0.5 * cfg['U_inf'] ** 2 * (2 * R)   # 0.5 rho U^2 D
        self.q_ref = q_ref

        self.cd_baseline = None
        self._current_state = None
        self._warmup_state = None  # cached equilibrated state (deterministic)
        self.obs_size = 2 * self.n_probes
        self.action_size = 1

    def _precompute_jet_surface_basis(self):
        """Surface-velocity basis of the jet pair for the immersed boundary.

        Jet 0 (top, 90 deg) blows outward and jet 1 (bottom, 270 deg) sucks for
        a > 0: basis = Q_max * (G_top n_top - G_bot n_bot) with Gaussian
        profiles G of width ``jet_width``.  Returns {'u': (Nx+1, Ny), 'v': (Nx, Ny+1)}.
        """
        grid = self.grid
        cx, cy = self.cfg['cx'], self.cfg['cy']
        R = self.cfg['R']
        sigma = self.jet_width
        Q_max = self.Q_max
        thetas = self.cfg['jet_thetas']

        xu, yu = grid.face_centers(0)
        Xu, Yu = jnp.meshgrid(xu, yu, indexing='ij')

        xv, yv = grid.face_centers(1)
        Xv, Yv = jnp.meshgrid(xv, yv, indexing='ij')

        basis_u = jnp.zeros_like(Xu)
        basis_v = jnp.zeros_like(Xv)

        for i, theta_j in enumerate(thetas):
            x_j = cx + R * np.cos(theta_j)
            y_j = cy + R * np.sin(theta_j)
            nx_j = np.cos(theta_j)
            ny_j = np.sin(theta_j)

            G_u = jnp.exp(-((Xu - x_j)**2 + (Yu - y_j)**2) / (2 * sigma**2))
            G_v = jnp.exp(-((Xv - x_j)**2 + (Yv - y_j)**2) / (2 * sigma**2))

            sign = 1.0 if i == 0 else -1.0
            basis_u = basis_u + sign * Q_max * nx_j * G_u
            basis_v = basis_v + sign * Q_max * ny_j * G_v

        return {'u': basis_u, 'v': basis_v}

    def _setup_probes(self, cfg):
        """The 151 probes of Rabault et al. (2019): a 9 x 7 wake grid, a 4 x 4
        upstream grid and two rings of 36 probes around the cylinder."""
        cx, cy = cfg['cx'], cfg['cy']
        R = cfg['R']
        probes_x, probes_y = [], []

        # wake grid: 0.75D to 4.5D downstream, +-1.5D cross-stream
        wake_x = np.linspace(cx + 0.75, cx + 4.5, 9)
        wake_y = np.linspace(cy - 1.5, cy + 1.5, 7)
        for xi in wake_x:
            for yi in wake_y:
                probes_x.append(xi)
                probes_y.append(yi)

        # upstream grid: -0.25D to 0.5D, 1.0D to 1.5D above and below the axis
        up_x = np.linspace(cx - 0.25, cx + 0.5, 4)
        up_y = np.concatenate([
            np.linspace(cy - 1.5, cy - 1.0, 2),
            np.linspace(cy + 1.0, cy + 1.5, 2),
        ])
        for xi in up_x:
            for yi in up_y:
                probes_x.append(xi)
                probes_y.append(yi)

        # rings at r = 0.7D and 1.0D from the centre
        for r_ring in [R + 0.2, R + 0.5]:
            for k in range(36):
                theta = 2.0 * np.pi * k / 36
                probes_x.append(cx + r_ring * np.cos(theta))
                probes_y.append(cy + r_ring * np.sin(theta))

        self.probe_x = np.array(probes_x)
        self.probe_y = np.array(probes_y)
        self.n_probes = len(self.probe_x)

        # fractional indices of the probes on the u- and v-face grids
        dx, dy = self.grid.spacings
        xu, yu = self.grid.face_centers(0)
        self.probe_ix_u = jnp.array((self.probe_x - float(xu[0])) / dx)
        self.probe_iy_u = jnp.array((self.probe_y - float(yu[0])) / dy)
        xv, yv = self.grid.face_centers(1)
        self.probe_ix_v = jnp.array((self.probe_x - float(xv[0])) / dx)
        self.probe_iy_v = jnp.array((self.probe_y - float(yv[0])) / dy)

    def _get_obs(self, state):
        """u and v at the probes (bilinear interpolation)."""
        u, v = state.velocity
        u_probes = jax.scipy.ndimage.map_coordinates(
            u, [self.probe_ix_u, self.probe_iy_u], order=1, mode='nearest',
        )
        v_probes = jax.scipy.ndimage.map_coordinates(
            v, [self.probe_ix_v, self.probe_iy_v], order=1, mode='nearest',
        )
        return jnp.concatenate([u_probes, v_probes])

    def _initial_state(self, key=None):
        """Parabolic inflow with a localised cross-flow perturbation behind the
        cylinder that triggers vortex shedding."""
        grid = self.grid
        U_inf = self.cfg['U_inf']

        Um = self.cfg['Um']
        Ly = grid.domain_size[1]
        dy = Ly / grid.shape[1]
        y = jnp.arange(grid.shape[1]) * dy + dy / 2   # u-faces lie at the cell-centre heights
        u_profile = Um * 4.0 * y * (Ly - y) / Ly**2
        u = jnp.broadcast_to(u_profile[None, :], grid.face_shape(0))
        v = jnp.zeros(grid.face_shape(1))

        cx, cy = self.cfg['cx'], self.cfg['cy']
        R = self.cfg['R']
        xv, yv = grid.face_centers(1)
        Xv, Yv = jnp.meshgrid(xv, yv, indexing='ij')
        r2 = ((Xv - (cx + 2*R)) / R)**2 + ((Yv - (cy + 0.5*R)) / R)**2
        v_perturb = 0.05 * U_inf * jnp.exp(-r2 / 0.5)
        v = v + v_perturb

        p = jnp.zeros(grid.cell_shape())
        return FluidState(velocity=(u, v), pressure=p, scalars={}, ibm_forces=None)

    def _step_with_jets_pure(self, state, action, dt):
        """One CFD step; the jet amplitude is passed to the immersed boundary."""
        return self.step_fn(state, dt, action)

    def reset(self, key=None, soft_prob=0.0):
        """Return (state, obs) of the equilibrated uncontrolled flow.

        The first call runs ``warmup_steps`` uncontrolled steps and measures the
        baseline drag over ``baseline_steps`` further steps; later calls reuse the
        cached state.  With probability ``soft_prob`` the current state is kept
        and only the jet filter is reset (Rabault et al. 2019).
        """
        if self._current_state is not None and soft_prob > 0:
            if np.random.random() < soft_prob:
                self.Q_actual = jnp.float32(0.0)
                obs = self._get_obs_jit(self._current_state)
                return self._current_state, obs

        if self._warmup_state is None:
            print(f"[CylinderFlowEnv] Hard reset: {self.warmup_steps} warmup steps...")
            state = self._initial_state()
            for i in range(self.warmup_steps):
                state = self._step_jit(state, self.dt, 0.0)
                if (i + 1) % 1000 == 0:
                    cd = float(state.ibm_forces[0]) / self.q_ref if state.ibm_forces is not None else 0.0
                    print(f"  warmup step {i+1}/{self.warmup_steps}, Cd={cd:.4f}")

            print(f"[CylinderFlowEnv] Measuring baseline Cd ({self.baseline_steps} steps)...")
            cd_sum = 0.0
            for i in range(self.baseline_steps):
                state = self._step_jit(state, self.dt, 0.0)
                if state.ibm_forces is not None:
                    cd_sum += float(state.ibm_forces[0]) / self.q_ref
            self.cd_baseline = cd_sum / self.baseline_steps
            print(f"  Cd_baseline = {self.cd_baseline:.4f}")
            self._warmup_state = state
        else:
            print("[CylinderFlowEnv] Hard reset (cached)")

        state = self._warmup_state
        self._current_state = state
        self.Q_actual = jnp.float32(0.0)
        obs = self._get_obs_jit(state)
        return state, obs

    def step_no_scan(self, state, action):
        """Advance one action interval with the filtered jet amplitude; return
        (state, obs, reward, info)."""
        dt = self.dt
        Q_smooth = self.Q_actual
        cds, cls = [], []
        for _ in range(self.action_interval):
            Q_smooth = Q_smooth + self.ema_alpha * (action - Q_smooth)
            state = self._step_with_jets_jit(state, float(Q_smooth), dt)
            if state.ibm_forces is not None:
                cds.append(float(state.ibm_forces[0]) / self.q_ref)
                cls.append(float(state.ibm_forces[1]) / self.q_ref)
        self.Q_actual = Q_smooth

        cd_mean = np.mean(cds) if cds else 0.0
        cl_mean_signed = np.mean(cls) if cls else 0.0
        reward = -(cd_mean - self.cd_baseline) - self.cl_weight * abs(cl_mean_signed)
        obs = self._get_obs_jit(state)
        self._current_state = state
        info = {'cd': cd_mean, 'cl': cl_mean_signed}
        return state, obs, float(reward), info
