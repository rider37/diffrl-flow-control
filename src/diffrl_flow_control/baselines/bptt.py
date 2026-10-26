"""Full-episode backpropagation through time (BPTT), Section 4.4 and Appendix G.

Full-episode BPTT reuses the SHAC trainer of each benchmark (rollout, reward,
segmented checkpointing, networks, learning rate and gradient clipping) and
changes only the actor loss and the update schedule:

  * the differentiated horizon is the whole training episode,
  * the actor loss contains no terminal value (the critic is still fitted and
    logged but does not enter the actor gradient),
  * the flow is reset to a training initial state after every update,
  * the number of updates gives SHAC's total CFD-step budget, unless a fixed
    number of updates is requested (equal-update runs of Fig. 6).

``configure`` returns the modified SHAC configuration and ``actor_loss`` the
loss that replaces ``shac_actor_loss`` in the trainer.
"""
import jax
import jax.numpy as jnp

from diffrl_flow_control.cylinder import shac as cyl_shac
from diffrl_flow_control.rayleigh_benard import shac as rb_shac


def episode_actions(env_cfg, shac_cfg):
    """Training episode, in actions at SHAC's training interval (the environment episode by default)."""
    steps = env_cfg['episode_actions'] * env_cfg['action_interval']
    assert steps % shac_cfg['action_interval'] == 0
    return steps // shac_cfg['action_interval']


def configure(env_cfg, shac_cfg, episode=None, updates=None):
    """SHAC configuration of a full-episode BPTT run with SHAC's CFD budget (or `updates` updates)."""
    cfg = dict(shac_cfg)
    budget = shac_cfg['n_epochs'] * shac_cfg['horizon'] * shac_cfg['action_interval']
    cfg['horizon'] = episode or episode_actions(env_cfg, shac_cfg)
    cfg['reset_every'] = 1
    cfg['n_epochs'] = updates or budget // (cfg['horizon'] * cfg['action_interval'])
    cfg['save_every'] = max(1, cfg['n_epochs'] // 10)
    cfg['snapshot_every'] = cfg['n_epochs']
    return cfg


def _no_terminal_value(critic_params, obs_h, critic_forward):
    V = critic_forward({'critic': critic_params}, obs_h)
    return jnp.zeros_like(V), jax.lax.stop_gradient(V)      # (value in the loss, value logged)


def cylinder_actor_loss(actor_params, critic_params, state0, shac_step_fn, get_obs_fn, q_ref,
                        h, action_interval, dt, gamma, segment_length=10, cd_baseline=0.0,
                        cl_weight=0.2, action_penalty_coef=0.0, q0=jnp.float32(0.0), ema_alpha=1.0):
    state_h, q_h, cds, cls, obs_history, actions = cyl_shac.shac_rollout_env(
        shac_step_fn, state0, q0, actor_params, get_obs_fn,
        q_ref, h, action_interval, dt, segment_length, ema_alpha)
    rewards = -(cds - cd_baseline) - cl_weight * jnp.abs(cls)
    obs_h = get_obs_fn(state_h)
    V_loss, V_log = _no_terminal_value(critic_params, obs_h, cyl_shac.critic_forward)
    discounted_return = jnp.sum(gamma ** jnp.arange(h) * rewards) + gamma ** h * V_loss
    loss = -discounted_return
    if action_penalty_coef > 0:
        loss = loss + action_penalty_coef * jnp.mean(actions ** 2)
    aux = {'rewards': rewards, 'cds': cds, 'cls': cls, 'actions': actions,
           'obs_history': obs_history, 'obs_h': obs_h, 'state_h': state_h, 'q_h': q_h,
           'V_h': V_log, 'return': discounted_return}
    return loss, aux


def rayleigh_benard_actor_loss(actor_params, critic_params, state0, shac_step_fn, get_obs_fn, compute_nu_fn,
                               nu_baseline, h, action_interval, dt, gamma, segment_length=10, action_dim=12,
                               action_penalty_coef=0.0, map_action_fn=None):
    state_h, nus, obs_history, actions = rb_shac.shac_rollout(
        shac_step_fn, state0, actor_params, get_obs_fn, compute_nu_fn, h,
        action_interval, dt, segment_length, action_dim, map_action_fn=map_action_fn)
    rewards = 1.0 - nus / nu_baseline
    obs_h = get_obs_fn(state_h)
    V_loss, V_log = _no_terminal_value(critic_params, obs_h, rb_shac.critic_forward)
    discounted_return = jnp.sum(gamma ** jnp.arange(h) * rewards) + gamma ** h * V_loss
    loss = -discounted_return
    if action_penalty_coef > 0:
        loss = loss + action_penalty_coef * jnp.mean(actions ** 2)
    aux = {'rewards': rewards, 'nus': nus, 'actions': actions, 'obs_history': obs_history,
           'obs_h': obs_h, 'state_h': state_h, 'V_h': V_log, 'return': discounted_return}
    return loss, aux


def actor_loss(problem):
    return cylinder_actor_loss if problem == 'cylinder' else rayleigh_benard_actor_loss
