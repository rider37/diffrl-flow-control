"""
2D cylinder flow environment with ZNMF synthetic jets for RL-based drag reduction.

Benchmark: Rabault et al. (2019) — Re=100 cylinder with 2 jets at 90/270 deg.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.state import FluidState
from jax_cfd.core.geometry import cylinder_coords_diff
from jax_cfd.core.ghost_cell_ibm import make_ghost_cell_ibm_fn
from jax_cfd.core.solver import create_step_fn
from jax_cfd.core.boundaries import apply_boundaries, pad_velocity_for_diffusion
from jax_cfd.core.schemes import get_pressure_solver, make_advection_scheme
from jax_cfd.core.turbulence import get_turbulence_model
from jax_cfd.core.actuators import surface_constrained_params


# ── Default config ──────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Domain
    'Re': 100,
    'Lx': 22.0, 'Ly': 10.0,
    'Nx': 352, 'Ny': 160,
    'R': 0.5, 'cx': 5.0, 'cy': 5.0,
    'U_inf': 1.0,
    'n_ibm_pts': 128,

    # Jets
    'n_jets': 2,
    'jet_thetas': [np.pi / 2, 3 * np.pi / 2],
    'jet_width': 0.15,
    'Q_max': 0.5,

    # RL timing
    'dt': 0.005,
    'action_interval': 50,
    'episode_actions': 80,
    'warmup_steps': 5000,
    'baseline_steps': 2000,

    # Probes
    'probe_layout': 'grid',
    'obs_type': 'velocity',
    'n_probes_x': 8,
    'n_probes_y': 8,
    'probe_x_range': [1.0, 6.0],   # relative to cx
    'probe_y_range': [-2.0, 2.0],  # relative to cy
}


class CylinderFlowEnv:
    """2D cylinder with ZNMF synthetic jets for RL-based drag reduction.

    Observation: pressure or velocity at probe locations
    Action: scalar in [-1, 1] — symmetric jet amplitude (ZNMF)
    Reward: -(Cd - Cd_baseline) - cl_weight * |mean(Cl)|
    """

    def __init__(self, config=None):
        cfg = {**DEFAULT_CONFIG}
        if config is not None:
            cfg.update(config)
        self.cfg = cfg

        Nx, Ny = cfg['Nx'], cfg['Ny']
        Lx, Ly = cfg['Lx'], cfg['Ly']
        Re = cfg['Re']
        R = cfg['R']
        cx, cy = cfg['cx'], cfg['cy']

        self.dt = cfg['dt']
        self.action_interval = cfg['action_interval']
        self.episode_actions = cfg.get('episode_actions', 80)
        self.warmup_steps = cfg['warmup_steps']
        self.baseline_steps = cfg['baseline_steps']
        self.Q_max = cfg['Q_max']
        self.cl_weight = cfg.get('cl_weight', 0.2)

        # Action smoothing (Rabault 2019: EMA α=0.1)
        self.ema_alpha = cfg.get('ema_alpha', 1.0)  # 1.0 = no smoothing
        self.Q_actual = jnp.float32(0.0)

        # ── Grid ──
        self.grid = MACGrid((Nx, Ny), (Lx, Ly))
        nu = cfg['U_inf'] * (2 * R) / Re   # nu = U*D/Re
        self.nu = nu

        # ── Jet method & geometry ──
        jet_method = cfg.get('jet_method', 'body_force')
        self.jet_method = jet_method
        self.jet_width = cfg['jet_width']

        jet_thetas = jnp.array(cfg['jet_thetas'])
        jet_center = jnp.array([cx, cy])
        positions, directions, _ = surface_constrained_params(
            jet_thetas, jet_center, R,
        )
        self.jet_positions = positions
        self.jet_directions = directions

        # ── IBM (ghost-cell) ──
        coords, ds = cylinder_coords_diff(
            jnp.float32(R), (jnp.float32(cx), jnp.float32(cy)),
            cfg['n_ibm_pts'],
        )
        # IBM interface half-width.  factor None -> eps=None -> the builder's
        # built-in 0.5*dx (original behavior, byte-for-byte unchanged).  Set
        # cfg['ibm_eps_factor']=0.75 for the benchmark-calibrated interface
        # (Cd_T5/Cd_IBM both match Schaefer-Turek; eps sweep 2026-06-05).
        _eps_factor = cfg.get('ibm_eps_factor', None)
        _ibm_eps = (None if _eps_factor is None
                    else float(_eps_factor) * (cfg['Lx'] / cfg['Nx']))
        if jet_method == 'surface_bc':
            self._jet_surface_basis = self._precompute_jet_surface_basis()
            self.ibm_fn = make_ghost_cell_ibm_fn(
                self.grid, coords, ds, eps=_ibm_eps,
                jet_surface_basis=self._jet_surface_basis,
            )
        else:
            self.ibm_fn = make_ghost_cell_ibm_fn(self.grid, coords, ds,
                                                 eps=_ibm_eps)
            self._jet_force_basis = self._precompute_jet_basis()

        # ── Boundary conditions ──
        inlet_type = cfg.get('inlet_type', 'uniform')
        if inlet_type == 'parabolic':
            inlet_bc = {'type': 'inlet_parabolic', 'Um': cfg.get('Um', 1.5)}
        else:
            inlet_bc = {'type': 'inlet', 'velocity': [cfg['U_inf'], 0.0]}
        bc_cfg = {
            'x_min': inlet_bc,
            'x_max': {'type': 'outlet_zero_gradient'},
            'y_min': {'type': cfg.get('wall_type', 'wall_no_slip')},
            'y_max': {'type': cfg.get('wall_type', 'wall_no_slip')},
        }
        self.bc_cfg = bc_cfg

        def apply_bc(vel, _cfg, g, inlet_vel=None):
            return apply_boundaries(vel, bc_cfg, g, inlet_vel=inlet_vel)

        def pad_diff(vel, g, inlet_vel=None):
            return pad_velocity_for_diffusion(vel, bc_cfg, g, inlet_vel=inlet_vel)

        # ── Solver components ──
        turb_fn = get_turbulence_model('laminar')
        advect_fn = make_advection_scheme('muscl', ())
        solve_p = get_pressure_solver('fft')

        # ── Step function ──
        self.step_fn = create_step_fn(
            self.grid, nu, apply_bc, pad_diff,
            turb_fn, advect_fn, solve_p,
            ibm_fn=self.ibm_fn,
            time_scheme='rk3',
        )

        # ── JIT-compile functions ──
        self._step_jit = jax.jit(self.step_fn)
        self._step_with_jets_jit = jax.jit(self._step_with_jets_pure)
        self._get_obs_jit = jax.jit(self._get_obs)

        # ── Probes ──
        self._setup_probes(cfg)

        # ── Reference quantities ──
        q_ref = 0.5 * cfg['U_inf'] ** 2 * (2 * R)   # 0.5 * rho * U^2 * D
        self.q_ref = q_ref

        # ── State tracking ──
        self.cd_baseline = None
        self._current_state = None
        self._warmup_state = None  # cached warmup result (deterministic)
        self.obs_size = self.n_probes if cfg.get('obs_type') == 'pressure' \
            else 2 * self.n_probes
        self.action_size = 1

    def _precompute_jet_surface_basis(self):
        """Pre-compute surface velocity basis for IBM jet injection.

        For ZNMF pair at theta=pi/2 and 3pi/2:
        - action > 0: top blows outward (+y), bottom sucks inward (+y)
        - Combined basis_v = Q_max * (G_top + G_bot)
        - Combined basis_u ~ 0 (jets at 90/270 have no x-component)

        Returns dict {'u': array(Nx+1,Ny), 'v': array(Nx,Ny+1)}.
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

            # ZNMF: jet 0 (top) = +Q, jet 1 (bottom) = -Q
            sign = 1.0 if i == 0 else -1.0
            basis_u = basis_u + sign * Q_max * nx_j * G_u
            basis_v = basis_v + sign * Q_max * ny_j * G_v

        return {'u': basis_u, 'v': basis_v}

    def _precompute_jet_basis(self):
        """Pre-compute force field for unit amplitude per jet.

        Forces are masked to the fluid region (outside cylinder) so that
        no momentum is wasted inside the IBM zone.  Without this masking,
        ~50 % of the Gaussian force falls inside the cylinder and is
        immediately zeroed by IBM, halving the effective jet strength.

        Returns dict with 'Fx0', 'Fy0', 'Fx1', 'Fy1'.
        """
        grid = self.grid
        width = self.jet_width
        cx, cy = self.cfg['cx'], self.cfg['cy']
        R = self.cfg['R']

        from jax_cfd.core.actuators import jet_actuator_force_mac

        # Fluid-region masks (1 outside cylinder, 0 inside)
        xu, yu = grid.face_centers(0)
        Xu, Yu = jnp.meshgrid(xu, yu, indexing='ij')
        mask_u = ((Xu - cx) ** 2 + (Yu - cy) ** 2 > R ** 2).astype(jnp.float32)

        xv, yv = grid.face_centers(1)
        Xv, Yv = jnp.meshgrid(xv, yv, indexing='ij')
        mask_v = ((Xv - cx) ** 2 + (Yv - cy) ** 2 > R ** 2).astype(jnp.float32)

        # Jet 0 at unit amplitude, masked to fluid region
        Fx0, Fy0 = jet_actuator_force_mac(
            grid, self.jet_positions[0], self.jet_directions[0], 1.0, width,
        )
        Fx0, Fy0 = Fx0 * mask_u, Fy0 * mask_v

        # Jet 1 at unit amplitude, masked to fluid region
        Fx1, Fy1 = jet_actuator_force_mac(
            grid, self.jet_positions[1], self.jet_directions[1], 1.0, width,
        )
        Fx1, Fy1 = Fx1 * mask_u, Fy1 * mask_v

        return {'Fx0': Fx0, 'Fy0': Fy0, 'Fx1': Fx1, 'Fy1': Fy1}

    def _setup_probes(self, cfg):
        """Set up probe locations.

        Supports two layouts:
        - 'rabault2019': 151 pressure probes in 3 groups (Rabault et al. 2019)
        - 'grid': regular NxM grid of velocity probes in wake
        """
        cx, cy = cfg['cx'], cfg['cy']
        R = cfg['R']
        layout = cfg.get('probe_layout', 'grid')

        if layout == 'rabault2019':
            probes_x, probes_y = [], []

            # Group 1: Wake grid (9×7 = 63 probes)
            # x: 0.75D to 4.5D downstream, y: ±1.5D cross-stream
            wake_x = np.linspace(cx + 0.75, cx + 4.5, 9)
            wake_y = np.linspace(cy - 1.5, cy + 1.5, 7)
            for xi in wake_x:
                for yi in wake_y:
                    probes_x.append(xi)
                    probes_y.append(yi)

            # Group 2: Upstream grid (4×4 = 16 probes)
            # x: -0.25D to 0.5D, y: ±1.0D to ±1.5D (outside near-cylinder)
            up_x = np.linspace(cx - 0.25, cx + 0.5, 4)
            up_y = np.concatenate([
                np.linspace(cy - 1.5, cy - 1.0, 2),
                np.linspace(cy + 1.0, cy + 1.5, 2),
            ])
            for xi in up_x:
                for yi in up_y:
                    probes_x.append(xi)
                    probes_y.append(yi)

            # Group 3: Cylinder rings (2 rings × 36 angles = 72 probes)
            for r_ring in [R + 0.2, R + 0.5]:  # r = 0.7D and 1.0D from center
                for k in range(36):
                    theta = 2.0 * np.pi * k / 36
                    probes_x.append(cx + r_ring * np.cos(theta))
                    probes_y.append(cy + r_ring * np.sin(theta))

            self.probe_x = np.array(probes_x)
            self.probe_y = np.array(probes_y)
        else:
            # Legacy: regular grid in wake
            px_lo, px_hi = cfg.get('probe_x_range', [1.0, 6.0])
            py_lo, py_hi = cfg.get('probe_y_range', [-2.0, 2.0])
            npx = cfg.get('n_probes_x', 8)
            npy = cfg.get('n_probes_y', 8)
            probe_x = np.linspace(cx + px_lo, cx + px_hi, npx)
            probe_y = np.linspace(cy + py_lo, cy + py_hi, npy)
            px_grid, py_grid = np.meshgrid(probe_x, probe_y, indexing='ij')
            self.probe_x = px_grid.ravel()
            self.probe_y = py_grid.ravel()

        self.n_probes = len(self.probe_x)
        self.obs_type = cfg.get('obs_type', 'velocity')

        # Pre-compute fractional indices for cell-centered grid (pressure)
        dx, dy = self.grid.spacings
        xc, yc = self.grid.cell_centers()
        self.probe_ix_p = jnp.array((self.probe_x - float(xc[0])) / dx)
        self.probe_iy_p = jnp.array((self.probe_y - float(yc[0])) / dy)

        # Pre-compute fractional indices for face grids (velocity)
        xu, yu = self.grid.face_centers(0)
        self.probe_ix_u = jnp.array((self.probe_x - float(xu[0])) / dx)
        self.probe_iy_u = jnp.array((self.probe_y - float(yu[0])) / dy)
        xv, yv = self.grid.face_centers(1)
        self.probe_ix_v = jnp.array((self.probe_x - float(xv[0])) / dx)
        self.probe_iy_v = jnp.array((self.probe_y - float(yv[0])) / dy)

    def _get_obs(self, state):
        """Extract observations at probe points via bilinear interpolation.

        Returns pressure (Rabault 2019) or velocity probes depending on obs_type.
        """
        if self.obs_type == 'pressure':
            p_probes = jax.scipy.ndimage.map_coordinates(
                state.pressure, [self.probe_ix_p, self.probe_iy_p],
                order=1, mode='nearest',
            )
            return p_probes  # shape (n_probes,)
        else:
            u, v = state.velocity
            u_probes = jax.scipy.ndimage.map_coordinates(
                u, [self.probe_ix_u, self.probe_iy_u], order=1, mode='nearest',
            )
            v_probes = jax.scipy.ndimage.map_coordinates(
                v, [self.probe_ix_v, self.probe_iy_v], order=1, mode='nearest',
            )
            return jnp.concatenate([u_probes, v_probes])

    def _initial_state(self, key=None):
        """Create initial flow with asymmetric perturbation to trigger shedding.

        Supports uniform or parabolic (Poiseuille) inlet profile.
        A localized cross-flow perturbation near the cylinder breaks symmetry,
        causing vortex shedding to develop ~3x faster than from quiescent start.
        """
        grid = self.grid
        U_inf = self.cfg['U_inf']

        if self.cfg.get('inlet_type') == 'parabolic':
            Um = self.cfg.get('Um', 1.5)
            Ly = grid.domain_size[1]
            # u-face y-positions are at cell centers
            dy = Ly / grid.shape[1]
            y = jnp.arange(grid.shape[1]) * dy + dy / 2
            u_profile = Um * 4.0 * y * (Ly - y) / Ly**2
            u = jnp.broadcast_to(u_profile[None, :], grid.face_shape(0))
        else:
            u = jnp.ones(grid.face_shape(0)) * U_inf
        v = jnp.zeros(grid.face_shape(1))

        # Add small asymmetric v-perturbation behind cylinder to trigger shedding
        cx, cy = self.cfg['cx'], self.cfg['cy']
        R = self.cfg['R']
        xv, yv = grid.face_centers(1)
        Xv, Yv = jnp.meshgrid(xv, yv, indexing='ij')
        # Localized perturbation at (cx+2R, cy+0.5R) — asymmetric
        r2 = ((Xv - (cx + 2*R)) / R)**2 + ((Yv - (cy + 0.5*R)) / R)**2
        v_perturb = 0.05 * U_inf * jnp.exp(-r2 / 0.5)
        v = v + v_perturb

        p = jnp.zeros(grid.cell_shape())
        return FluidState(velocity=(u, v), pressure=p, scalars={}, ibm_forces=None)

    def _compute_jet_force(self, action):
        """Compute MAC-staggered jet body force from scalar action.

        Uses pre-computed jet force basis for efficiency.
        action : scalar in [-1, 1]
        Returns (Fx, Fy) force arrays on MAC grid.
        """
        Q_top = action * self.Q_max
        Q_bottom = -Q_top   # ZNMF
        b = self._jet_force_basis
        Fx = Q_top * b['Fx0'] + Q_bottom * b['Fx1']
        Fy = Q_top * b['Fy0'] + Q_bottom * b['Fy1']
        return Fx, Fy

    def _step_with_jets_pure(self, state, action, dt):
        """Single CFD step + jet forcing (pure function for JIT)."""
        if self.jet_method == 'surface_bc':
            # Action passed as 'extra' → IBM applies jet surface velocity
            return self.step_fn(state, dt, action)
        else:
            # Original: external body-force impulse after step
            state_new = self.step_fn(state, dt, 0.0)
            Fx, Fy = self._compute_jet_force(action)
            u_new = state_new.velocity[0] + Fx * dt
            v_new = state_new.velocity[1] + Fy * dt
            return FluidState(
                velocity=(u_new, v_new),
                pressure=state_new.pressure,
                scalars=state_new.scalars,
                ibm_forces=state_new.ibm_forces,
            )

    def reset(self, key=None, soft_prob=0.0):
        """Initialize flow with warmup to developed vortex shedding.

        Parameters
        ----------
        key : PRNGKey, optional
        soft_prob : float
            Probability of soft reset (Rabault 2019: 0.8).
            0.0 = always hard reset (default).

        Returns (state, obs).
        """
        # Soft reset: keep current flow, just reset jet action
        if self._current_state is not None and soft_prob > 0:
            if np.random.random() < soft_prob:
                self.Q_actual = jnp.float32(0.0)
                obs = self._get_obs_jit(self._current_state)
                return self._current_state, obs

        # Hard reset: use cached warmup state (deterministic → identical every time)
        if self._warmup_state is None:
            print(f"[CylinderFlowEnv] Hard reset: {self.warmup_steps} warmup steps...")
            state = self._initial_state()
            for i in range(self.warmup_steps):
                state = self._step_jit(state, self.dt, 0.0)
                if (i + 1) % 1000 == 0:
                    cd = float(state.ibm_forces[0]) / self.q_ref if state.ibm_forces is not None else 0.0
                    print(f"  warmup step {i+1}/{self.warmup_steps}, Cd={cd:.4f}")

            # Measure baseline Cd
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

    def step(self, state, action):
        """Apply action for action_interval steps (uses lax.scan).

        Parameters
        ----------
        state : FluidState
        action : scalar in [-1, 1]

        Returns
        -------
        state_new, obs, reward, info
        """
        dt = self.dt
        q_ref = self.q_ref
        ema_alpha = self.ema_alpha

        def sub_step(carry, _):
            s, Q_smooth = carry
            Q_smooth = Q_smooth + ema_alpha * (action - Q_smooth)
            s_new = self._step_with_jets_pure(s, Q_smooth, dt)
            cd = s_new.ibm_forces[0] / q_ref
            cl = s_new.ibm_forces[1] / q_ref
            return (s_new, Q_smooth), (cd, cl)

        (state_new, Q_final), (cds, cls) = jax.lax.scan(
            sub_step, (state, self.Q_actual), None, length=self.action_interval,
        )
        self.Q_actual = Q_final

        cd_mean = jnp.mean(cds)
        cl_mean_signed = jnp.mean(cls)
        reward = -(cd_mean - self.cd_baseline) - self.cl_weight * jnp.abs(cl_mean_signed)

        obs = self._get_obs(state_new)
        self._current_state = state_new
        info = {'cd': cd_mean, 'cl': cl_mean_signed}
        return state_new, obs, reward, info

    def step_no_scan(self, state, action):
        """Python-loop version with JIT-compiled inner step + EMA smoothing."""
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
