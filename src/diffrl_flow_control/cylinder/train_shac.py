"""SHAC training loop for the cylinder.

Each epoch unrolls ``horizon`` actions through the differentiable solver from the
current state, updates the actor with the pathwise gradient of the truncated
return (reverse-mode AD with segmented checkpointing) and the critic with n-step
TD targets.  Writes history.json (configuration and per-epoch statistics),
checkpoints/{init,epoch_XXXX,best,final}.npz and flow snapshots to
``shac_cfg['output_dir']``.
"""
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np


from diffrl_flow_control.solver.operators import vorticity_2d
from diffrl_flow_control.cylinder.env import CylinderFlowEnv

from diffrl_flow_control.cylinder.ppo import init_actor_critic, init_adam, adam_update
from diffrl_flow_control.cylinder.shac import (
    make_jet_step_fn, make_get_obs_fn,
    shac_actor_loss, shac_critic_loss_batched,
    compute_td_targets,
)


# ═══════════════════════════════════════════════════════════════════════
# Data saving utilities
# ═══════════════════════════════════════════════════════════════════════

def save_flow_snapshot(state, grid, epoch, output_dir):
    """Save flow field arrays (u, v, p, vorticity) as .npz."""
    u, v = np.array(state.velocity[0]), np.array(state.velocity[1])
    p = np.array(state.pressure)
    omega = np.array(vorticity_2d(state.velocity, grid))

    snap_dir = os.path.join(output_dir, 'snapshots')
    os.makedirs(snap_dir, exist_ok=True)

    # Grid coordinates
    xc, yc = grid.cell_centers()
    xu, yu = grid.face_centers(0)
    xv, yv = grid.face_centers(1)

    np.savez_compressed(
        os.path.join(snap_dir, f'flow_epoch_{epoch:04d}.npz'),
        u=u, v=v, p=p, vorticity=omega,
        xc=np.array(xc), yc=np.array(yc),
        xu=np.array(xu), yu=np.array(yu),
        xv=np.array(xv), yv=np.array(yv),
        epoch=epoch,
    )


def save_flow_image(state, grid, epoch, output_dir, cfg_body=None):
    """Save vorticity + pressure contour images."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return  # skip if no matplotlib

    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    u, v = state.velocity
    omega = np.array(vorticity_2d((u, v), grid))
    p = np.array(state.pressure)

    xc, yc = grid.cell_centers()
    xc, yc = np.array(xc), np.array(yc)

    # ── Vorticity field ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), dpi=100)

    vmax = min(5.0, float(np.percentile(np.abs(omega), 99)))
    ax = axes[0]
    im = ax.pcolormesh(xc, yc, omega.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       shading='auto')
    if cfg_body:
        theta = np.linspace(0, 2*np.pi, 100)
        cx, cy, R = cfg_body['center'][0], cfg_body['center'][1], cfg_body['radius']
        ax.plot(cx + R*np.cos(theta), cy + R*np.sin(theta), 'k-', lw=1.5)
        ax.fill(cx + R*np.cos(theta), cy + R*np.sin(theta), 'white', zorder=5)
        ax.plot(cx + R*np.cos(theta), cy + R*np.sin(theta), 'k-', lw=1.5, zorder=6)
    ax.set_xlim(0, float(xc[-1]))
    ax.set_ylim(float(yc[0]), float(yc[-1]))
    ax.set_aspect('equal')
    ax.set_title(f'Vorticity  (Epoch {epoch})')
    plt.colorbar(im, ax=ax, shrink=0.6)

    # ── Pressure field ──
    ax = axes[1]
    pmax = float(np.percentile(np.abs(p), 98))
    if pmax < 1e-10:
        pmax = 1.0
    im = ax.pcolormesh(xc, yc, p.T, cmap='coolwarm', vmin=-pmax, vmax=pmax,
                       shading='auto')
    if cfg_body:
        ax.fill(cx + R*np.cos(theta), cy + R*np.sin(theta), 'white', zorder=5)
        ax.plot(cx + R*np.cos(theta), cy + R*np.sin(theta), 'k-', lw=1.5, zorder=6)
    ax.set_xlim(0, float(xc[-1]))
    ax.set_ylim(float(yc[0]), float(yc[-1]))
    ax.set_aspect('equal')
    ax.set_title(f'Pressure  (Epoch {epoch})')
    plt.colorbar(im, ax=ax, shrink=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f'fields_epoch_{epoch:04d}.png'),
                bbox_inches='tight')
    plt.close(fig)


def save_params(params, name, output_dir):
    """Save actor and critic parameters as checkpoints/<name>.npz."""
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    flat_actor = {f'actor_{i}_{k}': np.array(v)
                  for i, (W, b) in enumerate(params['actor'])
                  for k, v in [('W', W), ('b', b)]}
    flat_critic = {f'critic_{i}_{k}': np.array(v)
                   for i, (W, b) in enumerate(params['critic'])
                   for k, v in [('W', W), ('b', b)]}
    path = os.path.join(ckpt_dir, f'{name}.npz')
    np.savez(path, **flat_actor, **flat_critic)
    return path


def save_training_curves(history, output_dir):
    """Generate training curve plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    epochs = history['epoch']

    # Cd mean
    ax = axes[0, 0]
    ax.plot(epochs, history['cd_mean'], 'b-', lw=1.5, label='Cd (mean)')
    ax.axhline(history['cd_baseline'], color='r', ls='--', lw=1, label='Cd baseline')
    if history.get('cd_min'):
        ax.plot(epochs, history['cd_min'], 'c-', alpha=0.5, lw=0.8, label='Cd (min)')
        ax.plot(epochs, history['cd_max'], 'm-', alpha=0.5, lw=0.8, label='Cd (max)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cd')
    ax.set_title('Drag Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Actor loss
    ax = axes[0, 1]
    ax.plot(epochs, history['actor_loss'], 'r-', lw=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Actor Loss')
    ax.set_title('Actor Loss')
    ax.grid(True, alpha=0.3)

    # Critic loss
    ax = axes[1, 0]
    ax.plot(epochs, history['critic_loss'], 'g-', lw=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Critic Loss')
    ax.set_title('Critic Loss')
    ax.grid(True, alpha=0.3)

    # Gradient norm
    ax = axes[1, 1]
    ax.semilogy(epochs, history['grad_norm'], 'k-', lw=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('|grad|')
    ax.set_title('Actor Gradient Norm')
    ax.grid(True, alpha=0.3)

    # Actions
    ax = axes[2, 0]
    ax.plot(epochs, history['action_mean'], 'b-', lw=1.5, label='|a| mean')
    if history.get('action_std'):
        ax.fill_between(
            epochs,
            np.array(history['action_mean']) - np.array(history['action_std']),
            np.array(history['action_mean']) + np.array(history['action_std']),
            alpha=0.2,
        )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('|Action|')
    ax.set_title('Action Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Discounted return
    ax = axes[2, 1]
    ax.plot(epochs, history['return'], 'purple', lw=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Return')
    ax.set_title('Discounted Return')
    ax.grid(True, alpha=0.3)

    plt.suptitle('SHAC Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════

def train(env_cfg, shac_cfg):
    """SHAC training loop with comprehensive data logging."""
    seed = shac_cfg['seed']
    key = jax.random.PRNGKey(seed)
    output_dir = shac_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Snapshot/image save interval (every N epochs)
    snapshot_every = shac_cfg.get('snapshot_every', 10)
    reset_every = shac_cfg.get('reset_every', 0)  # 0 = no reset
    action_penalty_coef = shac_cfg.get('action_penalty', 0.01)

    # ── 1. Environment setup (warmup + baseline) ──────────────────────
    print('[SHAC] Initializing environment...')
    env = CylinderFlowEnv(env_cfg)
    state, obs = env.reset(key)
    cd_baseline = env.cd_baseline
    print(f'[SHAC] Cd_baseline = {cd_baseline:.4f}')
    print(f'[SHAC] obs_size = {env.obs_size}, action_size = {env.action_size}')

    # Save initial state for periodic reset (prevents state drift)
    state0_reset = jax.tree.map(lambda x: x.copy(), state)

    # Save baseline flow snapshot
    print('[SHAC] Saving baseline flow snapshot...')
    save_flow_snapshot(state, env.grid, epoch=-1, output_dir=output_dir)
    cfg_body = {'center': [env_cfg['cx'], env_cfg['cy']], 'radius': env_cfg['R']}
    save_flow_image(state, env.grid, epoch=-1, output_dir=output_dir, cfg_body=cfg_body)

    # ── 2. SHAC step function (wraps CFD + jet forces) ────────────────
    shac_step_fn = make_jet_step_fn(env.step_fn)

    # ── 3. Observation function (pure, for JIT) ───────────────────────
    get_obs_fn = make_get_obs_fn(
        env.probe_ix_u, env.probe_iy_u,
        env.probe_ix_v, env.probe_iy_v,
    )
    shac_obs_dim = 2 * env.n_probes
    print(f"[SHAC] Using VELOCITY obs ({shac_obs_dim} dim)")

    # ── 4. Initialize actor-critic + optimizers ───────────────────────
    key, subkey = jax.random.split(key)
    # separate actor and critic widths
    actor_dims = shac_cfg.get('actor_hidden_dims', shac_cfg.get('hidden_dims', [256, 256]))
    critic_dims = shac_cfg.get('critic_hidden_dims', shac_cfg.get('hidden_dims', [256, 256]))
    params = init_actor_critic(subkey, shac_obs_dim, env.action_size,
                               actor_dims, critic_dims)
    print(f"[SHAC] Actor: {actor_dims}, Critic: {critic_dims}")

    actor_opt = init_adam(params['actor'], lr=shac_cfg['actor_lr'])
    critic_opt = init_adam(params['critic'], lr=shac_cfg['critic_lr'])

    # Save initial params
    save_params(params, 'init', output_dir)

    # ── 5. JIT compile actor loss + grad ──────────────────────────────
    h = shac_cfg['horizon']
    action_interval = shac_cfg['action_interval']
    dt = env_cfg['dt']
    gamma = shac_cfg['gamma']
    segment_length = shac_cfg['segment_length']
    q_ref = env.q_ref
    max_grad_norm = shac_cfg['max_grad_norm']

    cl_weight = shac_cfg.get('cl_weight', 0.2)
    ema_alpha = float(env_cfg.get('ema_alpha', 1.0))   # environment action smoothing
    print(f'[SHAC] env-matched rollout: cl_weight={cl_weight}, ema_alpha={ema_alpha}, interval-mean Cd/Cl')

    def actor_loss_fn(actor_params, critic_params, state0, q0):
        return shac_actor_loss(
            actor_params, critic_params, state0,
            shac_step_fn, get_obs_fn, q_ref,
            h, action_interval, dt, gamma,
            segment_length, cd_baseline,
            cl_weight=cl_weight,
            action_penalty_coef=action_penalty_coef,
            q0=q0, ema_alpha=ema_alpha,
        )

    print(f'[SHAC] JIT compiling actor loss + grad (this may take a few minutes)...')
    t0 = time.time()
    actor_val_grad_fn = jax.jit(jax.value_and_grad(actor_loss_fn, argnums=0, has_aux=True))

    # Warmup JIT
    q_smooth = jnp.float32(0.0)   # env.Q_actual analogue (0 after reset)
    (loss_test, aux_test), grad_test = actor_val_grad_fn(
        params['actor'], params['critic'], state, q_smooth,
    )
    jax.block_until_ready(loss_test)
    jit_time = time.time() - t0
    print(f'[SHAC] JIT compiled in {jit_time:.1f}s')

    # Check gradient health
    grad_norm = jnp.sqrt(
        sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grad_test))
    )
    print(f'[SHAC] Initial: loss={float(loss_test):.4f}, grad_norm={float(grad_norm):.6f}')

    # JIT compile critic loss
    critic_val_grad_fn = jax.jit(jax.value_and_grad(shac_critic_loss_batched, argnums=0))

    # ── 6. Training loop ──────────────────────────────────────────────
    print(f'\n[SHAC] Training: {shac_cfg["n_epochs"]} epochs, h={h}, '
          f'action_interval={action_interval}')
    print(f'[SHAC] Grid: {env_cfg["Nx"]}x{env_cfg["Ny"]}, '
          f'segment_length={segment_length}')
    print(f'[SHAC] Snapshots every {snapshot_every} epochs')
    if reset_every > 0:
        print(f'[SHAC] State reset every {reset_every} epochs (anti-drift)')
    print(f'[SHAC] Action penalty coef: {action_penalty_coef}')
    print('=' * 70)

    history = {
        # Config metadata
        'cd_baseline': float(cd_baseline),
        'env_config': {k: (float(v) if isinstance(v, (int, float)) else
                          [float(x) for x in v] if isinstance(v, (list, tuple)) else str(v))
                       for k, v in env_cfg.items()},
        'shac_config': {k: (float(v) if isinstance(v, (int, float)) else
                           [float(x) for x in v] if isinstance(v, (list, tuple)) else str(v))
                        for k, v in shac_cfg.items()},
        'jit_compile_time': jit_time,

        # Per-epoch scalars
        'epoch': [],
        'actor_loss': [],
        'critic_loss': [],
        'cd_mean': [],
        'cd_min': [],
        'cd_max': [],
        'cl_mean': [],
        'action_mean': [],
        'action_std': [],
        'action_max': [],
        'grad_norm': [],
        'return': [],
        'V_h': [],
        'time': [],

        'cumulative_time': [],
        'total_cfd_steps': [],

        # Per-epoch per-step data (rewards & actions within each rollout)
        'rewards_per_step': [],    # list of (h,) arrays
        'actions_per_step': [],    # list of (h,) arrays
        'cds_per_step': [],        # list of (h,) arrays
        'cls_per_step': [],        # list of (h,) arrays
    }
    best_cd = float('inf')
    best_epoch = 0
    total_cfd_steps = 0

    t_start_training = time.time()

    for epoch in range(shac_cfg['n_epochs']):
        t_epoch = time.time()

        # ── Actor update (1st-order, through simulator) ──
        (actor_loss, aux), actor_grad = actor_val_grad_fn(
            params['actor'], params['critic'], state, q_smooth,
        )
        jax.block_until_ready(actor_loss)

        actor_grad_norm = jnp.sqrt(
            sum(jnp.sum(g ** 2) for g in jax.tree.leaves(actor_grad))
        )
        actor_opt, params['actor'] = adam_update(
            actor_opt, params['actor'], actor_grad,
            max_grad_norm=max_grad_norm,
        )

        # Extract info
        rewards = aux['rewards']
        cds = aux['cds']
        cls_arr = aux['cls']
        actions = aux['actions']
        obs_history = aux['obs_history']
        state_h = aux['state_h']

        cd_per_step = np.array(cds)
        cd_mean = float(np.mean(cd_per_step))
        cd_min = float(np.min(cd_per_step))
        cd_max = float(np.max(cd_per_step))

        actions_np = np.array(actions)
        action_mean = float(np.mean(np.abs(actions_np)))
        action_std = float(np.std(actions_np))
        action_max = float(np.max(np.abs(actions_np)))

        # Cl from rollout (mean absolute)
        cl_val = float(np.mean(np.abs(np.array(cls_arr))))

        # ── Critic update (0th-order, TD targets) ──
        V_h = float(aux['V_h'])
        td_targets = compute_td_targets(
            jax.lax.stop_gradient(rewards), V_h, gamma, h,
        )

        critic_loss_val = 0.0
        for _ in range(shac_cfg['critic_update_epochs']):
            c_loss, c_grad = critic_val_grad_fn(
                params['critic'], obs_history, td_targets,
            )
            critic_opt, params['critic'] = adam_update(
                critic_opt, params['critic'], c_grad,
            )
            critic_loss_val = float(c_loss)

        # ── Advance state (rolling horizon with periodic reset) ──
        total_cfd_steps += h * action_interval
        if reset_every > 0 and (epoch + 1) % reset_every == 0:
            # Reset to initial vortex shedding state (prevents state drift)
            state = jax.tree.map(lambda x: x.copy(), state0_reset)
            q_smooth = jnp.float32(0.0)
            print(f'  [State reset to baseline at epoch {epoch}]')
        else:
            state = jax.lax.stop_gradient(state_h)
            q_smooth = jax.lax.stop_gradient(aux['q_h'])

        # ── Logging ──
        dt_epoch = time.time() - t_epoch

        history['epoch'].append(epoch)
        history['actor_loss'].append(float(actor_loss))
        history['critic_loss'].append(critic_loss_val)
        history['cd_mean'].append(cd_mean)
        history['cd_min'].append(cd_min)
        history['cd_max'].append(cd_max)
        history['cl_mean'].append(cl_val)
        history['action_mean'].append(action_mean)
        history['action_std'].append(action_std)
        history['action_max'].append(action_max)
        history['grad_norm'].append(float(actor_grad_norm))
        history['return'].append(float(aux['return']))
        history['V_h'].append(V_h)
        history['time'].append(dt_epoch)
        history['cumulative_time'].append(time.time() - t_start_training)
        history['total_cfd_steps'].append(total_cfd_steps)
        history['rewards_per_step'].append([float(r) for r in rewards])
        history['actions_per_step'].append([float(a) for a in actions_np])
        history['cds_per_step'].append([float(c) for c in cd_per_step])
        history['cls_per_step'].append([float(c) for c in np.array(cls_arr)])

        if epoch % shac_cfg['log_every'] == 0:
            print(f'Epoch {epoch:4d} | '
                  f'Cd={cd_mean:.4f} ({cd_min:.4f}~{cd_max:.4f}) | '
                  f'Cl={cl_val:+.4f} | '
                  f'actor_L={float(actor_loss):+.4f} | '
                  f'critic_L={critic_loss_val:.4f} | '
                  f'|grad|={float(actor_grad_norm):.2e} | '
                  f'|a|={action_mean:.3f} (max={action_max:.3f}) | '
                  f'{dt_epoch:.1f}s')

        if cd_mean < best_cd:
            best_cd = cd_mean
            best_epoch = epoch
            # Save best model (overwrite single file)
            ckpt_dir = os.path.join(output_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            flat_actor = {f'actor_{i}_{k}': np.array(v)
                          for i, (W, b) in enumerate(params['actor'])
                          for k, v in [('W', W), ('b', b)]}
            flat_critic = {f'critic_{i}_{k}': np.array(v)
                           for i, (W, b) in enumerate(params['critic'])
                           for k, v in [('W', W), ('b', b)]}
            np.savez(os.path.join(ckpt_dir, 'best.npz'),
                     epoch=epoch, cd=cd_mean, **flat_actor, **flat_critic)

        # ── Save flow snapshots + images ──
        if epoch % snapshot_every == 0 or epoch == shac_cfg['n_epochs'] - 1:
            save_flow_snapshot(state, env.grid, epoch, output_dir)
            save_flow_image(state, env.grid, epoch, output_dir, cfg_body=cfg_body)
            print(f'  [Saved flow snapshot + image: epoch {epoch}]')

        # ── Save checkpoint ──
        if (epoch + 1) % shac_cfg['save_every'] == 0:
            ckpt_path = save_params(params, f'epoch_{epoch+1:04d}', output_dir)
            print(f'  [Saved checkpoint: {ckpt_path}]')

        # ── Incremental history save (crash-safe) ──
        if (epoch + 1) % 10 == 0 or epoch == shac_cfg['n_epochs'] - 1:
            _save_history(history, output_dir)

    # ── 7. Summary ────────────────────────────────────────────────────
    print('=' * 70)
    print(f'[SHAC] Training complete.')
    print(f'  Cd baseline:     {cd_baseline:.4f}')
    print(f'  Best Cd:         {best_cd:.4f}  (epoch {best_epoch})')
    print(f'  Drag reduction:  {(best_cd - cd_baseline) / cd_baseline * 100:+.1f}%')
    print(f'  Total CFD steps: {total_cfd_steps:,}')
    print(f'  Total time:      {sum(history["time"]):.1f}s')

    # Final saves
    _save_history(history, output_dir)
    save_params(params, 'final', output_dir)
    save_training_curves(history, output_dir)
    print(f'  Training curves: {output_dir}/training_curves.png')
    print(f'  History:         {output_dir}/history.json')

    return params, history


def _save_history(history, output_dir):
    """Save history JSON (crash-safe: write to tmp then rename)."""
    hist_path = os.path.join(output_dir, 'history.json')
    tmp_path = hist_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(history, f, indent=2, default=lambda x: float(x))
    os.replace(tmp_path, hist_path)
