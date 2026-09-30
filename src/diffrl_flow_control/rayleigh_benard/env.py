"""Rayleigh--Benard convection with a segmented heated bottom wall (Section 4.2).

Two-dimensional Boussinesq convection in a periodic cell of aspect ratio pi,
following Markmann et al. (2025).  The bottom wall is divided into 12 heater
segments whose temperatures are the control; their mean is held at T_hot so the
Rayleigh number is unchanged.  The observation is T, u and v on a 48 x 8 probe
grid; the reward is 1 - Nu / Nu_base.

Initial conditions: ``reset(ic_seed=s)`` uses the perturbed conduction profile
of IC seed ``s`` (training pool 0-14, held-out ICs 100-104); ``reset()`` without
a seed uses a fixed reference perturbation (modes k = 1-3), which the gradient
diagnostics use.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from diffrl_flow_control.solver.advection import advect_scalar_muscl
from diffrl_flow_control.solver.boundaries import (
    apply_boundaries, pad_scalar_for_diffusion, pad_velocity_for_diffusion,
)
from diffrl_flow_control.solver.grid import MACGrid
from diffrl_flow_control.solver.schemes import make_muscl_advection, make_pressure_solver_fft_periodic
from diffrl_flow_control.solver.state import FluidState
from diffrl_flow_control.solver.step import create_step_fn
from diffrl_flow_control.solver.viscosity import laminar_viscosity


class RBControlEnv:
    """2D Rayleigh-Benard convection with segmented bottom wall heating.

    Observation: T, u, v at probe grid (48x8 = 384 probes, 3 channels = 1152)
    Action: n_segments-dim continuous [-1, 1], mapped to temperature perturbations
    Reward: R = 1 - Nu(t) / Nu_base
    """

    def __init__(self, config):
        cfg = dict(config)
        self.cfg = cfg

        Nx, Ny = cfg['Nx'], cfg['Ny']
        Lx, Ly = cfg['Lx'], cfg['Ly']
        Ra = float(cfg['Ra'])
        Pr = float(cfg['Pr'])

        self.dt = cfg['dt']
        self.action_interval = cfg['action_interval']
        self.episode_actions = cfg['episode_actions']
        self.warmup_steps = cfg['warmup_steps']
        self.baseline_steps = cfg['baseline_steps']

        # Temperature
        self.T_hot = float(cfg['T_hot'])
        self.T_cold = float(cfg['T_cold'])
        self.T_ref = 0.5 * (self.T_hot + self.T_cold)
        self.perturb_amp = float(cfg['perturb_amp'])

        # Segments
        self.n_segments = cfg['n_segments']
        self.amplitude = float(cfg['amplitude'])
        seg_width = Nx // self.n_segments
        self.seg_width = seg_width
        # Map cell index → segment index
        self.segment_map = jnp.array(
            np.clip(np.arange(Nx) // seg_width, 0, self.n_segments - 1)
        )

        # ── Grid ──
        self.grid = MACGrid((Nx, Ny), (Lx, Ly))

        # ── Physics ──
        Re = np.sqrt(Ra / Pr)
        nu = 1.0 / Re
        self.nu = nu
        self.Ra = Ra
        self.Pr = Pr

        # ── Boundary conditions ──
        bc_cfg = {
            'x_min': {'type': 'periodic'},
            'x_max': {'type': 'periodic'},
            'y_min': {'type': 'wall_no_slip', 'T': self.T_hot},
            'y_max': {'type': 'wall_no_slip', 'T': self.T_cold},
        }
        self.bc_cfg = bc_cfg

        def apply_bc(vel, _cfg, g):
            return apply_boundaries(vel, bc_cfg, g)

        def pad_diff(vel, g):
            return pad_velocity_for_diffusion(vel, bc_cfg, g)

        # ── Solver components ──
        periodic_axes = (0,)
        advect_fn = make_muscl_advection(periodic_axes)
        solve_p = make_pressure_solver_fft_periodic(periodic_axes)
        _advect_scalar_fn = partial(advect_scalar_muscl, periodic_axes=periodic_axes)

        # Buoyancy: free-fall non-dim → gravity_vector=(0,1), beta=Pr
        gravity_vector = (0.0, 1.0)
        beta = Pr

        # Custom pad_scalar_fn with segment-wise BC support
        def pad_scalar_rb(s, name, g, extra=None):
            return self._pad_scalar_rb(s, name, g, extra)

        # ── Step function ──
        self.step_fn = create_step_fn(
            self.grid, nu, apply_bc, pad_diff,
            laminar_viscosity, advect_fn, solve_p,
            ibm_fn=None,
            solve_scalars=True,
            solve_buoyancy=True,
            gravity_vector=gravity_vector,
            beta=beta,
            T_ref=self.T_ref,
            advect_scalar_fn=_advect_scalar_fn,
            pad_scalar_fn=pad_scalar_rb,
            scalar_params={'T': {'Pr': Pr}},
        )

        # ── JIT-compile ──
        self._step_jit = jax.jit(self.step_fn)
        self._get_obs_jit = jax.jit(self._get_obs)
        self._compute_nu_jit = jax.jit(self.compute_nu)

        # ── Probes ──
        self._setup_probes(cfg)

        # ── State tracking ──
        self.nu_baseline = None
        self._current_state = None
        self._warmup_state = None
        # Multi-IC pool: {ic_seed → (warmup_state, nu_bl_mean, nu_bl_max)}
        self._warmup_state_pool = {}
        self.obs_size = 3 * self.n_probes  # T + u + v
        self.action_size = self.n_segments

    def _pad_scalar_rb(self, s, name, g, extra=None):
        """Custom scalar padding with segment-wise Dirichlet BC at bottom wall.

        When extra is not None and name == 'T':
            extra is (n_segments,) array of per-segment wall temperatures.
            The y_min ghost cells are overwritten with segment-wise values.
        Otherwise:
            Falls back to standard uniform padding (T_hot at bottom).
        """
        # Standard padding first (handles periodic x, uniform T at walls)
        padded = pad_scalar_for_diffusion(s, name, self.bc_cfg, g)

        if name != 'T' or extra is None:
            return padded

        # Overwrite y_min ghost row with segment-wise Dirichlet
        # padded shape: (Nx+2, Ny+2), interior cells at [1:-1, 1:-1]
        # y_min ghost: padded[:, 0], y_min interior: padded[:, 1]
        interior_bot = padded[1:-1, 1]  # (Nx,)
        T_wall_per_cell = extra[self.segment_map]  # (Nx,)
        ghost_bot = 2.0 * T_wall_per_cell - interior_bot
        padded = padded.at[1:-1, 0].set(ghost_bot)
        # Update x-periodic ghost cells at y_min row
        padded = padded.at[0, 0].set(padded[-2, 0])
        padded = padded.at[-1, 0].set(padded[1, 0])
        return padded

    def _setup_probes(self, cfg):
        """Uniform probe grid for T, u, v observation (Markmann 2025: 48x8)."""
        Lx, Ly = self.grid.domain_size
        npx = cfg['n_probes_x']
        npy = cfg['n_probes_y']

        # Probe locations (cell-centered spacing, avoid walls)
        px = np.linspace(Lx / (2 * npx), Lx - Lx / (2 * npx), npx)
        py = np.linspace(Ly / (2 * npy), Ly - Ly / (2 * npy), npy)
        PX, PY = np.meshgrid(px, py, indexing='ij')
        self.probe_x = PX.ravel()
        self.probe_y = PY.ravel()
        self.n_probes = len(self.probe_x)

        dx, dy = self.grid.spacings
        # Cell-centered fractional indices (for T)
        xc, yc = self.grid.cell_centers()
        self.probe_ix_c = jnp.array((self.probe_x - float(xc[0])) / dx)
        self.probe_iy_c = jnp.array((self.probe_y - float(yc[0])) / dy)
        # u-face fractional indices
        xu, yu = self.grid.face_centers(0)
        self.probe_ix_u = jnp.array((self.probe_x - float(xu[0])) / dx)
        self.probe_iy_u = jnp.array((self.probe_y - float(yu[0])) / dy)
        # v-face fractional indices
        xv, yv = self.grid.face_centers(1)
        self.probe_ix_v = jnp.array((self.probe_x - float(xv[0])) / dx)
        self.probe_iy_v = jnp.array((self.probe_y - float(yv[0])) / dy)

    def _get_obs(self, state):
        """Extract T, u, v at probe points via bilinear interpolation."""
        T = state.scalars['T']
        u, v = state.velocity
        T_probes = jax.scipy.ndimage.map_coordinates(
            T, [self.probe_ix_c, self.probe_iy_c], order=1, mode='nearest')
        u_probes = jax.scipy.ndimage.map_coordinates(
            u, [self.probe_ix_u, self.probe_iy_u], order=1, mode='nearest')
        v_probes = jax.scipy.ndimage.map_coordinates(
            v, [self.probe_ix_v, self.probe_iy_v], order=1, mode='nearest')
        return jnp.concatenate([T_probes, u_probes, v_probes])

    def compute_nu(self, state, T_wall_segments=None):
        """Wall-gradient Nusselt number (differentiable).

        Nu = 0.5 * (Nu_bot + Nu_top)

        When T_wall_segments is provided, the bottom-wall gradient uses
        the actual per-segment wall temperature instead of uniform T_hot.
        This is essential during active control where wall temperatures
        vary spatially.

        Parameters
        ----------
        state : FluidState
        T_wall_segments : (n_segments,) array, optional
            Actual bottom-wall temperatures per segment (from map_action).
            If None, uses uniform T_hot (baseline / uncontrolled).
        """
        T = state.scalars['T']
        dy = self.grid.spacings[1]
        Ly = self.grid.domain_size[1]
        dT = self.T_hot - self.T_cold

        if T_wall_segments is not None:
            # Per-cell wall temperature from segment map
            T_wall_bot = T_wall_segments[self.segment_map]  # (Nx,)
        else:
            T_wall_bot = self.T_hot

        Nu_bot = jnp.mean(2.0 * Ly * (T_wall_bot - T[:, 0]) / (dy * dT))
        Nu_top = jnp.mean(2.0 * Ly * (T[:, -1] - self.T_cold) / (dy * dT))
        return 0.5 * (Nu_bot + Nu_top)

    def map_action(self, action):
        """Map [-1,1]^n_segments to wall temperatures with zero-mean constraint.

        Markmann 2025, Eq. 3:
        1. Center: subtract mean → zero-mean perturbation
        2. Scale by amplitude / max(|centered|), clamped
        3. Add to T_hot
        """
        a_centered = action - jnp.mean(action)
        a_max = jnp.max(jnp.abs(a_centered))
        # Normalize only if max exceeds amplitude
        a_scaled = jnp.where(
            a_max > self.amplitude,
            a_centered * self.amplitude / jnp.maximum(a_max, 1e-8),
            a_centered,
        )
        return self.T_hot + a_scaled

    def _initial_state(self, key=None, ic_seed=None):
        """Linear T profile + perturbation, zero velocity.

        ic_seed : int, optional
            Random perturbation of modes k = 1-6 (amplitude factor in [0.5, 1.5],
            random phase).  Without a seed, fixed modes k = 1-3 (reference IC).
        """
        grid = self.grid
        Nx, Ny = grid.shape
        Lx, Ly = grid.domain_size

        u = jnp.zeros(grid.face_shape(0))
        v = jnp.zeros(grid.face_shape(1))
        p = jnp.zeros(grid.cell_shape())

        # Linear temperature: T_hot at y=0, T_cold at y=Ly
        xc, yc = grid.cell_centers()
        T_linear = self.T_hot + (self.T_cold - self.T_hot) * yc / Ly  # (Ny,)
        T = jnp.broadcast_to(T_linear[None, :], (Nx, Ny)).copy()

        amp = self.perturb_amp
        XC, YC = jnp.meshgrid(xc, yc, indexing='ij')
        if ic_seed is None:
            # reference IC: fixed modes k = 1-3
            for k in range(1, 4):
                T = T + amp * jnp.cos(2.0 * jnp.pi * k * XC / Lx) * jnp.sin(jnp.pi * YC / Ly)
        else:
            # seeded IC: random modes k = 1-6 (same as heldout.build_ic)
            rng = np.random.default_rng(int(ic_seed))
            T_np = np.asarray(T)
            XC_np = np.asarray(XC)
            YC_np = np.asarray(YC)
            for k in range(1, 7):
                a_k = rng.uniform(0.5, 1.5) * amp
                phi_k = rng.uniform(0.0, 2.0 * np.pi)
                T_np = T_np + a_k * np.cos(
                    2.0 * np.pi * k * XC_np / Lx + phi_k
                ) * np.sin(np.pi * YC_np / Ly)
            T = jnp.asarray(T_np)

        scalars = {'T': T}
        return FluidState(velocity=(u, v), pressure=p, scalars=scalars, ibm_forces=None)

    def preload_ic_pool(self, seeds, verbose=True):
        """Build, warmup, and cache a pool of initial conditions.

        After this call, ``self._warmup_state_pool[seed]`` contains the
        post-warmup state for each seed, and baseline Nu statistics are
        computed per-seed.  Also sets ``self.nu_baseline_pool_mean``
        (mean across pool) for reward normalization.

        Parameters
        ----------
        seeds : iterable of int
            IC seeds to preload.  Matches ``_initial_state(ic_seed=seed)``.
        verbose : bool
            Print progress messages.
        """
        pool_nu_means = []
        pool_nu_maxes = []
        for s in seeds:
            if s in self._warmup_state_pool:
                continue
            if verbose:
                print(f"[RBControlEnv] Preloading IC seed={s} "
                      f"(warmup={self.warmup_steps}, baseline={self.baseline_steps})")
            state = self._initial_state(ic_seed=s)
            for i in range(self.warmup_steps):
                state = self._step_jit(state, self.dt, None)
            nu_sum = 0.0
            nu_max = 0.0
            for i in range(self.baseline_steps):
                state = self._step_jit(state, self.dt, None)
                nu_val = float(self._compute_nu_jit(state))
                nu_sum += nu_val
                nu_max = max(nu_max, nu_val)
            nu_mean = nu_sum / self.baseline_steps
            self._warmup_state_pool[s] = (state, nu_mean, nu_max)
            pool_nu_means.append(nu_mean)
            pool_nu_maxes.append(nu_max)
            if verbose:
                print(f"  seed={s}: Nu_bl_mean={nu_mean:.4f}, Nu_bl_max={nu_max:.4f}")

        # Aggregate stats across pool (for baseline normalization)
        self.nu_baseline_pool_mean = float(np.mean(pool_nu_means))
        self.nu_baseline_pool_std = float(np.std(pool_nu_means))
        if verbose:
            print(f"[RBControlEnv] Pool ready: {len(self._warmup_state_pool)} ICs, "
                  f"Nu_bl pool mean = {self.nu_baseline_pool_mean:.4f} "
                  f"± {self.nu_baseline_pool_std:.4f}")

        # reward baseline: mean over the pool
        self.nu_baseline = self.nu_baseline_pool_mean
        self.nu_baseline_mean = self.nu_baseline_pool_mean
        self.nu_baseline_max = float(np.max(pool_nu_maxes))

    def reset(self, key=None, soft_prob=0.0, ic_seed=None):
        """Initialize RB flow and run warmup.

        With ``ic_seed`` the equilibrated state of that IC is taken from the
        pool (and added to it if missing).  Without a seed the reference IC is
        equilibrated once and cached; ``soft_prob`` is the probability of keeping
        the current state instead.

        Returns
        -------
        (state, obs)
        """
        # Multi-IC path
        if ic_seed is not None:
            if ic_seed not in self._warmup_state_pool:
                self.preload_ic_pool([ic_seed], verbose=False)
            state, nu_mean, nu_max = self._warmup_state_pool[ic_seed]
            if self.nu_baseline is None:
                self.nu_baseline = nu_mean
            self._current_state = state
            obs = self._get_obs_jit(state)
            return state, obs

        # reference IC
        if self._current_state is not None and soft_prob > 0:
            if np.random.random() < soft_prob:
                obs = self._get_obs_jit(self._current_state)
                return self._current_state, obs

        if self._warmup_state is None:
            print(f"[RBControlEnv] Warmup: {self.warmup_steps} steps (Ra={self.Ra:.0f}, Pr={self.Pr})...")
            state = self._initial_state()
            for i in range(self.warmup_steps):
                state = self._step_jit(state, self.dt, None)
                if (i + 1) % 100 == 0:
                    nu_val = float(self._compute_nu_jit(state))
                    print(f"  warmup {i+1}/{self.warmup_steps}, Nu={nu_val:.3f}")

            print(f"[RBControlEnv] Measuring baseline Nu ({self.baseline_steps} steps)...")
            nu_sum = 0.0
            nu_max = 0.0
            for i in range(self.baseline_steps):
                state = self._step_jit(state, self.dt, None)
                nu_val = float(self._compute_nu_jit(state))
                nu_sum += nu_val
                nu_max = max(nu_max, nu_val)
            self.nu_baseline_mean = nu_sum / self.baseline_steps
            self.nu_baseline_max = nu_max
            self.nu_baseline = self.nu_baseline_mean
            print(f"  Nu_baseline (mean) = {self.nu_baseline_mean:.4f}")
            print(f"  Nu_baseline (max)  = {self.nu_baseline_max:.4f}")
            self._warmup_state = state
        else:
            print("[RBControlEnv] Hard reset (cached)")

        state = self._warmup_state
        self._current_state = state
        obs = self._get_obs_jit(state)
        return state, obs



    def step_no_scan(self, state, action):
        """Advance one action interval with the heater temperatures of ``action``;
        return (state, obs, reward, info) with reward 1 - Nu / Nu_base."""
        T_segments = self.map_action(action)
        for _ in range(self.action_interval):
            state = self._step_jit(state, self.dt, T_segments)

        nu_val = float(self.compute_nu(state, T_segments))
        nu_reward = 1.0 - nu_val / self.nu_baseline
        reward = nu_reward

        obs = self._get_obs_jit(state)
        self._current_state = state
        info = {'Nu': nu_val}
        return state, obs, float(reward), info
