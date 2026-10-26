"""Gradient-informed PPO (GI-PPO; Son et al., NeurIPS 2023), Section 4.4 and Appendix G.

JAX port of the authors' algorithm (github.com/SonSang/gippo, rl_algorithm/gippo.py,
base.py, ppo.py) for the two environments of this package.  Per epoch:
  1. N actors each advance a window of H = 32 actions through the differentiable
     solver with stochastic actions a = mu + sigma*eps.  GAE advantages use the
     target critic; the gradient of the advantage with respect to each action,
     dA_t/da_t (including the dependence of later states, policy actions and
     critic values on a_t), is obtained by one backward pass per trajectory
     segment and the (gamma*tau)^-(t-t0) reweighting of the original code.
  2. alpha-policy: the actor is regressed onto a + alpha * dA/da (16 full-batch
     Adam steps, retried with a smaller step size if the loss increases).
  3. alpha is adapted for the next epoch from the change of the action-noise
     Jacobian determinant (delta_det), the out-of-range ratio (delta_oorr) and the
     estimated alpha-policy performance.
  4. critic regression on GAE targets (16 iterations, 4 batches) and a Polyak
     update of the target critic.
  5. PPO update of the actor with the virtual policy pi_h = (pi_old + pi_alpha)/2
     as the reference of the clipped ratio (5 epochs, retried as in the original).

Hyperparameters follow Appendix 7.4.2 of the GI-PPO paper (its Ant settings where
they are task-dependent).  Kept from this package's PPO for a controlled comparison:
the networks (widths, ReLU, state-dependent log sigma clipped to [-2, 0.5]), the
tanh action squashing (applied inside the environment step, so log-probabilities
are those of the pre-squash Gaussian sample), the environment, reward, episode
length and resets, no input or value normalisation, the number of transitions per
update and the total CFD steps (PPO's budget).

    python scripts/train.py rayleigh_benard gippo ra1e5 --seed 42
writes runs/<problem>/gippo_<case>_s<seed>/ (history.json, checkpoints/).
"""
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

from diffrl_flow_control.configs import load_config

HP = dict(
    horizon=32, gamma=0.99, tau=0.95,
    critic_lr=1e-3, critic_lr_final=1e-5, critic_iterations=16, critic_num_batch=4, target_critic_alpha=0.2,
    alpha0=0.5, max_alpha=1.0, beta=1.02, delta_det=0.4, delta_oorr=0.5,
    alpha_lr=5e-4, alpha_num_iter=16,
    ppo_lr=1e-4, ppo_epochs=5, ppo_minibatch=2048, e_clip=0.2,
    grad_norm=1.0, adam_betas=(0.7, 0.95), max_optim_iter=8, lr_multiplier=1.5,
)


# ── small Adam with global-norm clipping (betas of the original configuration) ──
def adam_init(params):
    z = jax.tree.map(jnp.zeros_like, params)
    return {'m': z, 'v': jax.tree.map(jnp.zeros_like, params), 't': 0}


def adam_step(opt, params, grads, lr, clip=HP['grad_norm'], betas=HP['adam_betas'], eps=1e-8):
    gn = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))
    scale = jnp.minimum(1.0, clip / (gn + 1e-6))
    grads = jax.tree.map(lambda g: g * scale, grads)
    t = opt['t'] + 1
    m = jax.tree.map(lambda m, g: betas[0] * m + (1 - betas[0]) * g, opt['m'], grads)
    v = jax.tree.map(lambda v, g: betas[1] * v + (1 - betas[1]) * g * g, opt['v'], grads)
    mh = jax.tree.map(lambda m: m / (1 - betas[0] ** t), m)
    vh = jax.tree.map(lambda v: v / (1 - betas[1] ** t), v)
    params = jax.tree.map(lambda p, a, b: p - lr * a / (jnp.sqrt(b) + eps), params, mh, vh)
    return {'m': m, 'v': v, 't': t}, params


class RunningMeanStd:                    # as gippo/utils.py (scalar)
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-5)

    def update(self, x):
        bm, bv, bc = float(np.mean(x)), float(np.var(x)), len(x)
        d = bm - self.mean; tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot


# ── problem adapters: differentiable one-action step, resets, networks ──
def make_problem(problem, case, seed):
    rng = np.random.default_rng(seed)
    if problem == 'cylinder':
        from diffrl_flow_control.cylinder.env import CylinderFlowEnv
        from diffrl_flow_control.cylinder import shac
        from diffrl_flow_control.cylinder.ppo import init_actor_critic, actor_forward, critic_forward
        env_cfg = load_config('cylinder', f'env_{case}'); ppo_cfg = load_config('cylinder', f'ppo_{case}')
        env = CylinderFlowEnv(env_cfg)
        warm, _ = env.reset(jax.random.PRNGKey(seed))
        step_fn = shac.make_jet_step_fn(env.step_fn)
        get_obs = shac.make_get_obs_fn(env.probe_ix_u, env.probe_iy_u, env.probe_ix_v, env.probe_iy_v)
        ai, dt, ema, q_ref = int(env.action_interval), float(env_cfg['dt']), float(env_cfg['ema_alpha']), float(env.q_ref)
        cd0, clw = float(env.cd_baseline), float(env_cfg.get('cl_weight', 0.2))
        seg = max(d for d in range(1, 31) if ai % d == 0)     # checkpoint segment (memory only)
        soft_p = float(env_cfg.get('soft_reset_prob', 0.8))

        def act_step(s, q, a):
            s2, q2, cd, cl = shac.differentiable_simulate_segmented_env(step_fn, s, q, a[0], ai, dt, seg, ema, q_ref)
            return s2, q2, -(cd - cd0) - clw * jnp.abs(cl), cd

        def initial_state():
            return warm, jnp.float32(0.0)

        def reset_info():           # (soft flag, hard-reset state) for an episode end
            return jnp.bool_(rng.random() < soft_p), warm

        metric = lambda m: 100.0 * (1.0 - m / cd0)            # drag reduction, %
        obs_dim, A = 2 * env.n_probes, int(env.action_size)
        L = int(ppo_cfg['episode_length'])
        n_actors = int(ppo_cfg['episodes_per_update'] * ppo_cfg['episode_length']) // HP['horizon']
        ppo_budget = ppo_cfg['n_updates'] * ppo_cfg['episodes_per_update'] * ppo_cfg['episode_length'] * ai
        info = dict(baseline=cd0, metric='drag_reduction_pct')
    else:
        from diffrl_flow_control.rayleigh_benard.env import RBControlEnv
        from diffrl_flow_control.rayleigh_benard import shac
        from diffrl_flow_control.rayleigh_benard.ppo import init_actor_critic, actor_forward, critic_forward
        env_cfg = load_config('rayleigh_benard', f'env_{case}'); ppo_cfg = load_config('rayleigh_benard', f'ppo_{case}')
        env = RBControlEnv(env_cfg)
        pool = list(ppo_cfg['ic_pool_seeds'])
        env.preload_ic_pool(pool, verbose=False)
        env.reset(ic_seed=int(rng.choice(pool)))
        nu0 = float(env.nu_baseline)
        step_fn = shac.make_heater_step_fn(env.step_fn, env.T_hot, env.amplitude)
        get_obs = shac.make_rb_get_obs_fn(env.probe_ix_c, env.probe_iy_c, env.probe_ix_u, env.probe_iy_u,
                                          env.probe_ix_v, env.probe_iy_v)
        compute_nu = shac.make_compute_nu_fn(env.grid, env.T_hot, env.T_cold, segment_map=env.segment_map)
        map_action = shac.make_map_action_fn(env.T_hot, env.amplitude)
        ai, dt = int(env_cfg['action_interval']), float(env_cfg['dt'])
        seg = max(d for d in range(1, 31) if ai % d == 0)     # checkpoint segment (memory only)

        def act_step(s, q, a):
            s2 = shac.differentiable_simulate_segmented(step_fn, s, ai, dt, segment_length=seg, action=a)
            nu = compute_nu(s2, map_action(a))
            return s2, q, 1.0 - nu / nu0, nu

        def pool_state():
            return env.reset(ic_seed=int(rng.choice(pool)))[0]

        def initial_state():
            return pool_state(), jnp.float32(0.0)

        def reset_info():
            return jnp.bool_(False), pool_state()

        metric = lambda m: 100.0 * (1.0 - m / nu0)            # Nu reduction, %
        obs_dim, A = int(env.obs_size), int(env.action_size)
        L = int(ppo_cfg['episode_length'])
        n_actors = -(-int(ppo_cfg['episodes_per_update'] * ppo_cfg['episode_length']) // HP['horizon'])
        ppo_budget = ppo_cfg['n_updates'] * ppo_cfg['episodes_per_update'] * ppo_cfg['episode_length'] * ai
        info = dict(baseline=nu0, metric='nu_reduction_pct', ic_pool=pool)
    params = init_actor_critic(jax.random.PRNGKey(seed), obs_dim, A,
                               ppo_cfg['actor_hidden_dims'], ppo_cfg['critic_hidden_dims'])
    return dict(env_cfg=env_cfg, ppo_cfg=ppo_cfg, act_step=act_step, get_obs=get_obs, initial_state=initial_state,
                reset_info=reset_info, metric=metric, A=A, L=L, n_actors=n_actors, interval=ai,
                ppo_budget=ppo_budget, params=params, actor_forward=actor_forward, critic_forward=critic_forward,
                info=info)


def build_window(P):
    """jit-compiled H-action window and its gradient with respect to additive
    action perturbations (evaluated at zero)."""
    H, gamma, tau, L = HP['horizon'], HP['gamma'], HP['tau'], P['L']
    act_step, get_obs = P['act_step'], P['get_obs']
    actor_forward, critic_forward = P['actor_forward'], P['critic_forward']

    def window(delta, actor, tcritic, s0, q0, k0, eps, soft, hard_s):
        def body(carry, inp):
            s, q, k = carry
            e, d = inp
            o = get_obs(s)
            mean, log_std = actor_forward({'actor': actor}, o)
            a = mean + jnp.exp(log_std) * e + d
            v = critic_forward({'critic': tcritic}, o)
            s2, q2, r, m = act_step(s, q, jnp.tanh(a))
            nv = critic_forward({'critic': tcritic}, get_obs(s2))
            k2 = k + 1
            done = k2 >= L
            keep = jax.tree.map(jax.lax.stop_gradient, s2)
            reset_s = jax.tree.map(lambda x, y: jnp.where(soft, x, y), keep, hard_s)
            s_next = jax.tree.map(lambda x, y: jnp.where(done, y, x), s2, reset_s)
            q_next = jnp.where(done, jnp.float32(0.0), q2)
            return (s_next, q_next, jnp.where(done, 0, k2)), (o, a, mean, log_std, v, r, nv, done, m)
        (sH, qH, kH), outs = jax.lax.scan(jax.checkpoint(body), (s0, q0, k0), (eps, delta))
        o, a, mean, log_std, v, r, nv, done, m = outs

        def gae(last, x):
            r_t, nv_t, v_t, done_t = x
            adv = r_t + gamma * nv_t - v_t + gamma * tau * (1.0 - done_t) * last
            return adv, adv
        _, adv = jax.lax.scan(gae, jnp.float32(0.0), (r, nv, v, done.astype(jnp.float32)), reverse=True)
        start = jnp.concatenate([jnp.array([True]), done[:-1]])
        obj = jnp.sum(jnp.where(start, adv, 0.0))
        aux = dict(obs=o, act=a, mean=mean, log_std=log_std, v=v, r=r, nv=nv, done=done, metric=m,
                   adv=adv, start=start, sH=sH, qH=qH, kH=kH)
        return obj, aux

    vg = jax.jit(jax.value_and_grad(window, argnums=0, has_aux=True))
    c = 1.0 / (gamma * tau)

    def run(actor, tcritic, s0, q0, k0, eps, soft, hard_s):
        (obj, aux), g = vg(jnp.zeros_like(eps), actor, tcritic, s0, q0, k0, eps, soft, hard_s)
        start = np.asarray(aux['start']); w = np.ones(H, np.float32); cv = 1.0
        for t in range(H):
            if start[t]:
                cv = 1.0
            w[t] = cv; cv *= c
        aux['adv_grad'] = g * w[:, None]
        return aux
    return run, window


def neglogp(x, mean, log_std):
    return jnp.sum(0.5 * ((x - mean) / jnp.exp(log_std)) ** 2 + 0.5 * jnp.log(2 * jnp.pi) + log_std, axis=-1)


def train(problem, case, seed, n_epochs=None, out_dir=None):
    P = make_problem(problem, case, seed)
    H, A, N = HP['horizon'], P['A'], P['n_actors']
    epochs_budget = int(round(P['ppo_budget'] / (N * H * P['interval'])))
    n_epochs = n_epochs or epochs_budget
    out_dir = out_dir or os.path.join('runs', problem, f'gippo_{case}_s{seed}')
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
    print(f'[GI-PPO] {problem} {case} seed {seed}: {N} actors x {H} actions x {P["interval"]} steps, '
          f'{n_epochs} epochs = {n_epochs * N * H * P["interval"]:,} CFD steps (PPO {P["ppo_budget"]:,})', flush=True)
    actor_forward, critic_forward = P['actor_forward'], P['critic_forward']
    run_window, _ = build_window(P)
    key = jax.random.PRNGKey(seed + 1)
    actor, critic = P['params']['actor'], P['params']['critic']
    tcritic = jax.tree.map(lambda x: x, critic)
    opt_ppo, opt_alpha, opt_critic = adam_init(actor), adam_init(actor), adam_init(critic)
    alpha, alpha_lr, ppo_lr = HP['alpha0'], HP['alpha_lr'], HP['ppo_lr']
    est_rms = RunningMeanStd()
    actors = []
    for i in range(N):                       # all actors start an episode together, as a vectorised env
        s, q = P['initial_state']()
        actors.append([s, q, 0])

    def mean_logstd(params, obs):
        return jax.vmap(lambda o: actor_forward({'actor': params}, o))(obs)

    def alpha_loss(params, obs, eps, target):
        mu, ls = mean_logstd(params, obs)
        return jnp.mean(jnp.sum((mu + jnp.exp(ls) * eps - target) ** 2, axis=-1))
    alpha_vg = jax.jit(jax.value_and_grad(alpha_loss))

    def critic_loss(params, obs, target):
        return jnp.mean((jax.vmap(lambda o: critic_forward({'critic': params}, o))(obs) - target) ** 2)
    critic_vg = jax.jit(jax.value_and_grad(critic_loss))

    def ppo_loss(params, obs, act, nlp_before, nlp_after, adv):
        mu, ls = mean_logstd(params, obs)
        nlp = neglogp(act, mu, ls)
        t_ratio = nlp_before - nlp_after
        unstable = jnp.any(jnp.abs(t_ratio) > 4.0)
        mid = jnp.log(2.0) - (jnp.log(jnp.exp(t_ratio) + 1.0) - nlp_before)
        ratio = jnp.where(unstable, nlp_after - nlp, mid - nlp)
        ratio = jnp.exp(jnp.clip(ratio, -16.0, 16.0))
        return jnp.mean(jnp.maximum(-adv * ratio, -adv * jnp.clip(ratio, 1 - HP['e_clip'], 1 + HP['e_clip'])))
    ppo_vg = jax.jit(jax.value_and_grad(ppo_loss))
    nlp_fn = jax.jit(lambda params, obs, act: neglogp(act, *mean_logstd(params, obs)))

    hist = {k: [] for k in ('epoch', 'reward_mean', 'metric_mean', 'alpha', 'oorr', 'est_alpha_perf', 'det_min',
                            'det_max', 'alpha_converged', 'alpha_loss_ratio', 'critic_loss', 'ppo_loss_first',
                            'ppo_loss_last', 'ppo_retries', 'adv_grad_norm', 'time', 'cumulative_time',
                            'total_cfd_steps', 'episode_returns')}
    hist.update(hp=HP, info=P['info'], n_actors=N, problem=problem, case=case, seed=seed)
    t_start = time.time(); ep_return = np.zeros(N)
    for epoch in range(n_epochs):
        t0 = time.time()
        buf = []
        for i in range(N):
            key, sub = jax.random.split(key)
            eps = jax.random.normal(sub, (H, A))
            soft, hard_s = P['reset_info']()
            s, q, k = actors[i]
            aux = run_window(actor, tcritic, s, q, jnp.int32(k), eps, soft, hard_s)
            actors[i] = [jax.lax.stop_gradient(aux['sH']), jax.lax.stop_gradient(aux['qH']), int(aux['kH'])]
            aux['eps'] = eps
            buf.append({k_: np.asarray(v_) for k_, v_ in aux.items() if k_ not in ('sH', 'qH', 'kH')})
            r, d = buf[-1]['r'], buf[-1]['done']
            for t in range(H):
                ep_return[i] += r[t]
                if d[t]:
                    hist['episode_returns'].append(float(ep_return[i])); ep_return[i] = 0.0
        cat = lambda k_: np.concatenate([b[k_] for b in buf], axis=0)
        obs, act, eps, adv, v = cat('obs'), cat('act'), cat('eps'), cat('adv'), cat('v')
        mu_old, ls_old, adv_grad = cat('mean'), cat('log_std'), cat('adv_grad')
        nlp_old = np.asarray(neglogp(act, mu_old, ls_old))

        # 2. alpha-policy
        # (as the original: the actor is restored after a failed try, the optimiser state is not)
        target = act + alpha * adv_grad
        backup = actor; converged = False; ratio_l = np.nan; lr_try = alpha_lr
        for _ in range(HP['max_optim_iter']):
            l0 = l1 = None
            for j in range(HP['alpha_num_iter']):
                l, g = alpha_vg(actor, obs, eps, target)
                opt_alpha, actor = adam_step(opt_alpha, actor, g, lr_try)
                if j == 0:
                    l0 = float(l)
                elif j == HP['alpha_num_iter'] - 1:
                    l1 = float(l)
            ratio_l = l1 / l0
            if l0 < l1:
                actor = backup; lr_try = lr_try / HP['lr_multiplier']
                continue
            converged = True
            break
        # 3. alpha for the next epoch
        mu_new, ls_new = mean_logstd(actor, obs)
        det = np.exp(np.sum(np.asarray(ls_new) - ls_old, axis=-1))
        next_alpha = alpha
        if not converged:
            alpha_lr = max(alpha_lr / HP['lr_multiplier'], 1e-5)    # base step size of later epochs
        elif det.min() < 1 - HP['delta_det'] or det.max() > 1 + HP['delta_det']:
            next_alpha = alpha / HP['beta']
        else:
            next_alpha = alpha * HP['beta']
        next_alpha = min(next_alpha, HP['max_alpha'])
        nlp_alpha = np.asarray(nlp_fn(actor, obs, act))
        pac = np.exp(np.minimum(nlp_old - nlp_alpha, 16.0))
        oorr = float(np.mean((pac < 1 - HP['e_clip']) | (pac > 1 + HP['e_clip'])))
        est = float(np.sum(adv * pac) - np.sum(adv)); n_est = est_rms.normalize(est); est_rms.update(np.array([est]))
        if oorr > HP['delta_oorr'] or (est < 0 and n_est < -1.0):
            next_alpha = min(alpha / HP['beta'], HP['max_alpha'])
        alpha_used, alpha = alpha, next_alpha

        # 4. critic + target critic
        frac = epoch / max(1, n_epochs)
        clr = (HP['critic_lr_final'] - HP['critic_lr']) * frac + HP['critic_lr']
        tgt = adv + v; bs = -(-len(tgt) // HP['critic_num_batch'])
        for _ in range(HP['critic_iterations']):
            for b0 in range(0, len(tgt), bs):
                cl, g = critic_vg(critic, obs[b0:b0 + bs], tgt[b0:b0 + bs])
                opt_critic, critic = adam_step(opt_critic, critic, g, clr)
        a_ = HP['target_critic_alpha']
        tcritic = jax.tree.map(lambda t_, c_: a_ * t_ + (1 - a_) * c_, tcritic, critic)

        # 5. PPO with the virtual policy
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        backup = actor; retries = 0; first = last = np.nan
        lr0 = ppo_lr
        for _ in range(HP['max_optim_iter']):
            losses = []
            for _e in range(HP['ppo_epochs']):
                for b0 in range(0, len(adv_n), HP['ppo_minibatch']):
                    sl = slice(b0, b0 + HP['ppo_minibatch'])
                    l, g = ppo_vg(actor, obs[sl], act[sl], nlp_old[sl], nlp_alpha[sl], adv_n[sl])
                    opt_ppo, actor = adam_step(opt_ppo, actor, g, ppo_lr)
                    losses.append(float(l))
            nb = -(-len(adv_n) // HP['ppo_minibatch'])
            first, last = np.mean(losses[:nb]), np.mean(losses[-nb:])
            if last > first:
                actor = backup; retries += 1; ppo_lr = lr0 / HP['lr_multiplier']
                continue
            ppo_lr = lr0
            break
        dt_e = time.time() - t0
        cfd = (epoch + 1) * N * H * P['interval']
        rew, met = cat('r'), cat('metric')
        for k_, v_ in (('epoch', epoch), ('reward_mean', float(rew.mean())), ('metric_mean', float(P['metric'](met.mean()))),
                       ('alpha', alpha_used), ('oorr', oorr), ('est_alpha_perf', est), ('det_min', float(det.min())),
                       ('det_max', float(det.max())), ('alpha_converged', bool(converged)), ('alpha_loss_ratio', float(ratio_l)),
                       ('critic_loss', float(cl)), ('ppo_loss_first', float(first)), ('ppo_loss_last', float(last)),
                       ('ppo_retries', retries), ('adv_grad_norm', float(np.linalg.norm(adv_grad) / np.sqrt(len(adv_grad)))),
                       ('time', dt_e), ('cumulative_time', time.time() - t_start), ('total_cfd_steps', cfd)):
            hist[k_].append(v_)
        print(f'epoch {epoch:4d} | r={rew.mean():+.4f} | {P["info"]["metric"]}={P["metric"](met.mean()):+.2f} | '
              f'alpha={alpha_used:.3g} oorr={oorr:.2f} det=[{det.min():.3f},{det.max():.3f}] conv={converged} | '
              f'|dA/da|={hist["adv_grad_norm"][-1]:.3g} | ppo retries {retries} | {dt_e:.0f}s', flush=True)
        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            save(actor, critic, os.path.join(out_dir, 'checkpoints', 'final.npz' if epoch == n_epochs - 1
                                             else f'epoch_{epoch + 1:04d}.npz'))
            json.dump(hist, open(os.path.join(out_dir, 'history.json'), 'w'))
    return out_dir


def save(actor, critic, path):
    d = {f'actor_{i}_W': np.asarray(W) for i, (W, b) in enumerate(actor)}
    d.update({f'actor_{i}_b': np.asarray(b) for i, (W, b) in enumerate(actor)})
    d.update({f'critic_{i}_W': np.asarray(W) for i, (W, b) in enumerate(critic)})
    d.update({f'critic_{i}_b': np.asarray(b) for i, (W, b) in enumerate(critic)})
    np.savez(path, **d)


