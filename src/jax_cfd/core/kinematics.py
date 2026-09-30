import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Callable, Tuple

# A motion function takes physical time `t` and the base markers `coords_init` (e.g. (xs, ys))
# and returns current `coords_new`, `vel_tgt`
MotionFn = Callable[[float, Tuple[jnp.ndarray, ...]], Tuple[Tuple[jnp.ndarray, ...], Tuple[jnp.ndarray, ...]]]

def get_motion_fn(motion_config: Dict[str, Any]) -> MotionFn:
    """
    Parses the motion config and returns a compiled JAX kinematic function.
    config examples:
    {'type': 'static'}
    {'type': 'pitching', 'amplitude_deg': 20.0, 'frequency_hz': 0.5}
    {'type': 'plunging', 'amplitude': 0.5, 'frequency_hz': 1.0}
    """
    mtype = motion_config.get('type', 'static').lower()

    if mtype == 'static':
        @jax.jit
        def static_motion(t: float, coords_init: Tuple[jnp.ndarray, ...]):
            """Returns initial positions and zero velocity."""
            zeros = tuple(jnp.zeros_like(c) for c in coords_init)
            return coords_init, zeros
        return static_motion

    elif mtype == 'pitching':
        amp_rad = jnp.radians(motion_config.get('amplitude_deg', 0.0))
        freq = motion_config.get('frequency_hz', 0.0)
        omega = 2.0 * jnp.pi * freq
        
        # We need the center of rotation. If not provided, assume origin or centroid.
        # This should ideally be passed in, but we assume translation is handled in geometry.py
        # so xs_init, ys_init are already around the rotational origin.
        cx, cy = motion_config.get('center', (0.0, 0.0))
        
        @jax.jit
        def pitching_motion(t: float, coords_init: Tuple[jnp.ndarray, ...]):
            """
            Rotates points by theta(t) = A * sin(wt) in the 2D XY plane.
            Velocity is r x w.
            """
            theta = amp_rad * jnp.sin(omega * t)
            dtheta_dt = amp_rad * omega * jnp.cos(omega * t)
            
            xs_init, ys_init = coords_init[0], coords_init[1]
            
            rx = xs_init - cx
            ry = ys_init - cy
            
            xs_new = cx + rx * jnp.cos(theta) - ry * jnp.sin(theta)
            ys_new = cy + rx * jnp.sin(theta) + ry * jnp.cos(theta)
            
            u_tgt = -dtheta_dt * (ys_new - cy)
            v_tgt =  dtheta_dt * (xs_new - cx)
            
            coords_new = [xs_new, ys_new]
            vel_tgt = [u_tgt, v_tgt]
            
            if len(coords_init) > 2:
                coords_new.append(coords_init[2]) # zs remains same
                vel_tgt.append(jnp.zeros_like(coords_init[2]))
                
            return tuple(coords_new), tuple(vel_tgt)
        return pitching_motion

    elif mtype == 'plunging':
        amp = motion_config.get('amplitude', 0.0)
        freq = motion_config.get('frequency_hz', 0.0)
        omega = 2.0 * jnp.pi * freq
        
        @jax.jit
        def plunging_motion(t: float, coords_init: Tuple[jnp.ndarray, ...]):
            """Vertical translation y(t) = A * sin(wt)"""
            y_disp = amp * jnp.sin(omega * t)
            v_vel = amp * omega * jnp.cos(omega * t)
            
            xs_init, ys_init = coords_init[0], coords_init[1]
            
            xs_new = xs_init
            ys_new = ys_init + y_disp
            u_tgt = jnp.zeros_like(xs_init)
            v_tgt = jnp.full_like(ys_init, v_vel)
            
            coords_new = [xs_new, ys_new]
            vel_tgt = [u_tgt, v_tgt]
            
            if len(coords_init) > 2:
                coords_new.append(coords_init[2])
                vel_tgt.append(jnp.zeros_like(coords_init[2]))
                
            return tuple(coords_new), tuple(vel_tgt)
        return plunging_motion
        
    elif mtype == 'undulatory':
        # ── Carangiform undulatory swimming ──────────────────────────────────
        # Traveling lateral wave propagates from head (s=0) to tail (s=L):
        #   y_wave(s, t) = A(s) · sin(2π(s/λ − f·t))
        #   A(s) = amp_head·L + (amp_tail − amp_head)·L · (s/L)²
        # where s = xs − x_nose (arclength along body axis).
        # amp_head and amp_tail are dimensionless fractions of body length L.
        # L and x_nose are inferred from coords_init at runtime.
        wavelength = float(motion_config.get('wavelength', 1.0))
        freq       = float(motion_config.get('frequency_hz', 1.0))
        a_head     = float(motion_config.get('amp_head', 0.02))   # fraction of L
        a_tail     = float(motion_config.get('amp_tail', 0.15))   # fraction of L
        omega      = 2.0 * jnp.pi * freq

        @jax.jit
        def undulatory_motion(t: float, coords_init: Tuple[jnp.ndarray, ...]):
            """
            Pure lateral traveling-wave undulation.

            Returns:
              coords_new: deformed body positions (xs_new, ys_new)
              vel_tgt:    target (no-slip) velocity at each Lagrangian point
            """
            xs0, ys0 = coords_init[0], coords_init[1]

            # Infer body length and nose position from reference configuration
            x_nose = jnp.min(xs0)
            L      = jnp.max(xs0) - x_nose         # effective body length

            # Normalized arclength s ∈ [0, 1] for every body point
            s_norm = (xs0 - x_nose) / L             # 0 at nose, 1 at tail

            # Amplitude envelope: small at head, large at tail
            A = (a_head + (a_tail - a_head) * s_norm ** 2) * L

            # Traveling-wave phase (in body-frame x-coordinates)
            phi = 2.0 * jnp.pi * ((xs0 - x_nose) / wavelength - freq * t)

            y_wave = A * jnp.sin(phi)
            vy     = -A * omega * jnp.cos(phi)      # ∂y_wave/∂t

            xs_new = xs0                             # no axial oscillation
            ys_new = ys0 + y_wave

            u_tgt = jnp.zeros_like(xs0)
            v_tgt = vy

            if len(coords_init) > 2:                 # 3-D guard
                return (xs_new, ys_new, coords_init[2]), (u_tgt, v_tgt, jnp.zeros_like(coords_init[2]))
            return (xs_new, ys_new), (u_tgt, v_tgt)

        return undulatory_motion

    else:
        raise ValueError(f"Unknown motion type: {mtype}")


def make_articulated_motion(
    n_joints: int,
    xs0_np: np.ndarray,
    ys0_np: np.ndarray,
) -> Callable:
    """
    Factory: given N joints and a reference NACA body, returns a JAX-jit function

        body_fn(angles, coords_init) -> ((xs_new, ys_new), (u_tgt, v_tgt))

    The body is divided into (n_joints + 1) equal-length segments.
    Joint k sits between segment k and segment k+1.
    Segment k's angle from the +x axis = cumsum(angles)[k-1]  (segment 0 is always horizontal).

    Forward kinematics (2-D chain):
        seg_starts[k] = start position of segment k  (computed via jax.lax.scan)
        surface_pt    = centerline_pt + y_offset × unit_normal

    y_offset (NACA lateral half-thickness relative to centerline) is precomputed once
    from the reference body and stored as a JAX constant.

    Target velocity is zero: the IBM enforces no-slip at the new positions,
    which is correct for position-control RL (agent sets target angles each step).

    Args:
        n_joints:  Number of joints (body split into n_joints+1 segments).
        xs0_np:    Reference Lagrangian x-coordinates (numpy, shape (N_pts,)).
        ys0_np:    Reference Lagrangian y-coordinates (numpy, shape (N_pts,)).

    Returns:
        body_fn  — JAX-jit function.
    """
    n_segs = n_joints + 1

    # Reference geometry (numpy, computed once)
    nose_idx  = int(np.argmin(xs0_np))
    x_nose    = float(xs0_np[nose_idx])
    y_center  = float(ys0_np[nose_idx])   # centerline y = cy  (NACA nose has yt = 0)
    L         = float(xs0_np.max() - xs0_np.min())
    L_seg     = L / n_segs

    s_norm    = (xs0_np - x_nose) / L                            # (N,) in [0, 1]
    seg_k     = np.minimum((s_norm * n_segs).astype(int),
                           n_segs - 1)                            # which segment
    t_local   = (s_norm - seg_k / n_segs) * n_segs               # local param in [0, 1]
    y_offset  = ys0_np - y_center                                 # signed lateral offset

    # JAX constants (captured in closure)
    seg_k_j     = jnp.array(seg_k)
    t_local_j   = jnp.array(t_local)
    y_offset_j  = jnp.array(y_offset)
    nose_pos_j  = jnp.array([x_nose, y_center])

    @jax.jit
    def body_fn(angles: jnp.ndarray, _coords_init: Tuple) -> Tuple:
        """
        angles: (n_joints,) joint deflection angles [rad].
        Segment 0 runs along +x; segment k has cumulative angle sum(angles[0:k]).
        _coords_init is accepted for API compatibility; geometry is precomputed in closure.
        """
        # Cumulative segment angles: seg_angles[k] = angle of segment k from +x
        seg_angles = jnp.concatenate([jnp.zeros(1), jnp.cumsum(angles)])  # (n_segs,)

        # Forward kinematics: scan over segments to get start position of each
        def seg_step(start_pos, angle_k):
            end_pos = start_pos + L_seg * jnp.array([jnp.cos(angle_k),
                                                      jnp.sin(angle_k)])
            return end_pos, start_pos   # carry = end, output = start

        _, seg_starts = jax.lax.scan(seg_step, nose_pos_j, seg_angles)
        # seg_starts[k] = (x, y) start of segment k,  shape (n_segs, 2)

        # Per-body-point: gather angle and start position
        angle_per_pt = seg_angles[seg_k_j]       # (N,)
        start_per_pt = seg_starts[seg_k_j]        # (N, 2)

        # Centerline position for each point
        cx = start_per_pt[:, 0] + t_local_j * L_seg * jnp.cos(angle_per_pt)
        cy = start_per_pt[:, 1] + t_local_j * L_seg * jnp.sin(angle_per_pt)

        # Unit normal (perpendicular to segment, positive = "upward" in body frame)
        nx = -jnp.sin(angle_per_pt)
        ny =  jnp.cos(angle_per_pt)

        # Surface point = centerline + lateral offset × normal
        xs_new = cx + y_offset_j * nx
        ys_new = cy + y_offset_j * ny

        # Target velocity: zero (position control — IBM enforces no-slip at new positions)
        u_tgt = jnp.zeros_like(xs_new)
        v_tgt = jnp.zeros_like(ys_new)

        return (xs_new, ys_new), (u_tgt, v_tgt)

    return body_fn
