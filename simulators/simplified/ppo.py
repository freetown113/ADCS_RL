from typing import Any, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from simulators.simplified.config import ExperimentConfig
from simulators.simplified.control import policy_to_command
from simulators.simplified.env import EnvState, SatelliteEnv


class TrainState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState


class Transition(NamedTuple):
    obs: jax.Array
    pre_tanh_action: jax.Array
    action: jax.Array
    command_action: jax.Array
    old_log_prob: jax.Array
    old_value: jax.Array
    reward: jax.Array
    done: jax.Array
    angle_rad: jax.Array
    rate_norm: jax.Array
    settled: jax.Array
    wheel_fraction: jax.Array
    wheel_failure: jax.Array
    allocation_error: jax.Array
    progress: jax.Array
    attitude_cost: jax.Array
    rate_cost: jax.Array
    action_cost: jax.Array
    smoothness_cost: jax.Array


class PPOBatch(NamedTuple):
    obs: jax.Array
    pre_tanh_action: jax.Array
    action: jax.Array
    old_log_prob: jax.Array
    old_value: jax.Array
    advantage: jax.Array
    returns: jax.Array


def tree_l2_norm(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def explained_variance(target: jax.Array, prediction: jax.Array) -> jax.Array:
    variance = jnp.var(target)
    return jnp.where(
        variance > 1.0e-8,
        1.0 - jnp.var(target - prediction) / variance,
        0.0,
    )


def gaussian_log_prob(z: jax.Array, mean: jax.Array, log_std: float) -> jax.Array:
    inv_std = jnp.exp(-log_std)
    elementwise = -0.5 * (
        jnp.square((z - mean) * inv_std)
        + 2.0 * log_std
        + jnp.log(2.0 * jnp.pi)
    )
    return jnp.sum(elementwise, axis=-1)


def squashed_log_prob(
    z: jax.Array,
    action: jax.Array,
    mean: jax.Array,
    log_std: float,
) -> jax.Array:
    base = gaussian_log_prob(z, mean, log_std)
    correction = jnp.sum(
        jnp.log(jnp.maximum(1.0 - jnp.square(action), 1.0e-6)), axis=-1
    )
    return base - correction


def sample_action(
    params: hk.Params,
    apply_fn,
    obs: jax.Array,
    key: jax.Array,
    fixed_log_std: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    mean, value = apply_fn(params, obs)
    noise = jax.random.normal(key, shape=mean.shape)
    z = mean + jnp.exp(fixed_log_std) * noise
    action = jnp.tanh(z)
    log_prob = squashed_log_prob(z, action, mean, fixed_log_std)
    return z, action, log_prob, value


def deterministic_action(
    params: hk.Params,
    apply_fn,
    obs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    mean, value = apply_fn(params, obs)
    return jnp.tanh(mean), value


def make_optimizer(config: ExperimentConfig) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(config.ppo.max_grad_norm),
        optax.adam(
            learning_rate=config.ppo.learning_rate,
            eps=config.ppo.adam_eps,
        ),
    )


def collect_rollout(
    env: SatelliteEnv,
    params: hk.Params,
    apply_fn,
    initial_state: EnvState,
    key: jax.Array,
    fixed_log_std: float,
) -> tuple[EnvState, Transition]:
    """Collects exactly one complete episode from every environment."""

    def rollout_step(carry, _):
        state, rng = carry
        rng, action_key = jax.random.split(rng)
        obs = env.observe(state)
        z, action, log_prob, value = sample_action(
            params, apply_fn, obs, action_key, fixed_log_std
        )
        command_action = policy_to_command(env, state, action)
        next_state, reward, done, info = env.step(state, command_action)
        transition = Transition(
            obs=obs,
            pre_tanh_action=z,
            action=action,
            command_action=command_action,
            old_log_prob=log_prob,
            old_value=value,
            reward=reward,
            done=done,
            angle_rad=info.reward_terms.attitude_angle_rad,
            rate_norm=info.reward_terms.rate_norm,
            settled=info.settled,
            wheel_fraction=info.wheel_speed_fraction_max,
            wheel_failure=jnp.any(info.wheel_mask < 0.5, axis=-1).astype(jnp.float32),
            allocation_error=info.allocation_error_norm,
            progress=info.reward_terms.progress,
            attitude_cost=info.reward_terms.attitude_cost,
            rate_cost=info.reward_terms.rate_cost,
            action_cost=info.reward_terms.action_cost,
            smoothness_cost=info.reward_terms.smoothness_cost,
        )
        return (next_state, rng), transition

    (final_state, _), rollout = jax.lax.scan(
        rollout_step,
        (initial_state, key),
        None,
        length=env.episode_steps,
    )
    return final_state, rollout


def compute_gae(
    reward: jax.Array,
    done: jax.Array,
    value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:

    last_value = jnp.zeros_like(value[-1])

    def reverse_step(carry, inputs):
        next_advantage, next_value = carry
        reward_t, done_t, value_t = inputs
        not_done = 1.0 - done_t
        delta = reward_t + gamma * not_done * next_value - value_t
        advantage = delta + gamma * gae_lambda * not_done * next_advantage
        return (advantage, value_t), advantage

    (_, _), advantage = jax.lax.scan(
        reverse_step,
        (jnp.zeros_like(last_value), last_value),
        (reward, done, value),
        reverse=True,
    )
    returns = advantage + value
    return advantage, returns


def flatten_rollout(rollout: Transition, advantage: jax.Array, returns: jax.Array) -> PPOBatch:
    def flatten(x: jax.Array) -> jax.Array:
        return x.reshape((-1,) + x.shape[2:])

    return PPOBatch(
        obs=flatten(rollout.obs),
        pre_tanh_action=flatten(rollout.pre_tanh_action),
        action=flatten(rollout.action),
        old_log_prob=flatten(rollout.old_log_prob),
        old_value=flatten(rollout.old_value),
        advantage=advantage.reshape(-1),
        returns=returns.reshape(-1),
    )


def make_ppo_update(apply_fn, optimizer, config: ExperimentConfig):
    ppo = config.ppo
    fixed_log_std = config.network.fixed_log_std

    def loss_fn(params: hk.Params, batch: PPOBatch):
        mean, value = apply_fn(params, batch.obs)
        new_log_prob = squashed_log_prob(
            batch.pre_tanh_action,
            batch.action,
            mean,
            fixed_log_std,
        )
        log_ratio = jnp.clip(new_log_prob - batch.old_log_prob, -20.0, 20.0)
        ratio = jnp.exp(log_ratio)

        unclipped_objective = ratio * batch.advantage
        clipped_objective = jnp.clip(
            ratio,
            1.0 - ppo.clip_epsilon,
            1.0 + ppo.clip_epsilon,
        ) * batch.advantage
        policy_loss = -jnp.mean(jnp.minimum(unclipped_objective, clipped_objective))

        value_clipped = batch.old_value + jnp.clip(
            value - batch.old_value,
            -ppo.value_clip_epsilon,
            ppo.value_clip_epsilon,
        )
        value_error = jnp.square(value - batch.returns)
        clipped_value_error = jnp.square(value_clipped - batch.returns)
        value_loss = 0.5 * jnp.mean(jnp.maximum(value_error, clipped_value_error))

        gaussian_entropy = mean.shape[-1] * (
            fixed_log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
        )
        total = (
            policy_loss
            + ppo.value_coef * value_loss
            - ppo.entropy_coef * gaussian_entropy
        )

        approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
        clip_fraction = jnp.mean(
            jnp.abs(ratio - 1.0) > ppo.clip_epsilon
        )
        deterministic = jnp.tanh(mean)
        aux = {
            "loss": total,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": gaussian_entropy,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "policy_mean_abs": jnp.mean(jnp.abs(mean)),
            "policy_mean_max": jnp.max(jnp.abs(mean)),
            "deterministic_action_abs": jnp.mean(jnp.abs(deterministic)),
            "deterministic_action_max": jnp.max(jnp.abs(deterministic)),
            "sampled_action_abs": jnp.mean(jnp.abs(batch.action)),
            "action_saturation": jnp.mean(jnp.abs(batch.action) > 0.95),
            "action_reconstruction_error": jnp.max(
                jnp.abs(batch.action - jnp.tanh(batch.pre_tanh_action))
            ),
        }
        return total, aux

    def minibatch_step(train_state: TrainState, batch: PPOBatch):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            train_state.params, batch
        )
        updates, next_opt_state = optimizer.update(
            grads, train_state.opt_state, train_state.params
        )
        next_params = optax.apply_updates(train_state.params, updates)
        grad_norm = tree_l2_norm(grads)
        update_norm = tree_l2_norm(updates)
        param_norm = tree_l2_norm(train_state.params)
        metrics = {
            **metrics,
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "param_norm": param_norm,
            "relative_update": update_norm / (param_norm + 1.0e-12),
            "gradient_clipped": (grad_norm > ppo.max_grad_norm).astype(jnp.float32),
        }
        return TrainState(next_params, next_opt_state), metrics

    def update(train_state: TrainState, batch: PPOBatch, key: jax.Array):
        sample_count = batch.obs.shape[0]
        if sample_count % ppo.num_minibatches != 0:
            raise ValueError(
                "episode_steps * num_envs must be divisible by num_minibatches"
            )
        minibatch_size = sample_count // ppo.num_minibatches
        epoch_keys = jax.random.split(key, ppo.update_epochs)

        def epoch_step(state: TrainState, epoch_key: jax.Array):
            permutation = jax.random.permutation(epoch_key, sample_count)
            shuffled = jax.tree_util.tree_map(lambda x: x[permutation], batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape(
                    (ppo.num_minibatches, minibatch_size) + x.shape[1:]
                ),
                shuffled,
            )
            next_state, minibatch_metrics = jax.lax.scan(
                minibatch_step, state, minibatches
            )
            mean_metrics = jax.tree_util.tree_map(
                lambda x: jnp.mean(x, axis=0), minibatch_metrics
            )
            return next_state, mean_metrics

        next_state, epoch_metrics = jax.lax.scan(
            epoch_step, train_state, epoch_keys
        )
        metrics = jax.tree_util.tree_map(
            lambda x: jnp.mean(x, axis=0), epoch_metrics
        )
        return next_state, metrics

    return jax.jit(update)


def rollout_metrics(
    rollout: Transition,
    advantage: jax.Array,
    returns: jax.Array,
) -> dict[str, jax.Array]:
    episode_return = jnp.sum(rollout.reward, axis=0)
    settled_ever = jnp.any(rollout.settled, axis=0)
    initial_angle = rollout.angle_rad[0]
    final_angle = rollout.angle_rad[-1]
    final_rate = rollout.rate_norm[-1]

    return {
        "episode_return_mean": jnp.mean(episode_return),
        "episode_return_median": jnp.median(episode_return),
        "initial_angle_deg_mean": jnp.mean(jnp.rad2deg(initial_angle)),
        "final_angle_deg_mean": jnp.mean(jnp.rad2deg(final_angle)),
        "final_angle_deg_median": jnp.median(jnp.rad2deg(final_angle)),
        "final_angle_deg_p90": jnp.percentile(jnp.rad2deg(final_angle), 90.0),
        "minimum_angle_deg_mean": jnp.mean(
            jnp.rad2deg(jnp.min(rollout.angle_rad, axis=0))
        ),
        "angle_improvement_deg_mean": jnp.mean(
            jnp.rad2deg(initial_angle - final_angle)
        ),
        "final_rate_mean": jnp.mean(final_rate),
        "final_rate_p90": jnp.percentile(final_rate, 90.0),
        "peak_rate_mean": jnp.mean(jnp.max(rollout.rate_norm, axis=0)),
        "success_rate": jnp.mean(settled_ever.astype(jnp.float32)),
        "wheel_fraction_max": jnp.max(rollout.wheel_fraction),
        "wheel_failure_fraction": jnp.mean(rollout.wheel_failure),
        "allocation_error_max": jnp.max(rollout.allocation_error),
        "reward_mean": jnp.mean(rollout.reward),
        "reward_std": jnp.std(rollout.reward),
        "return_mean": jnp.mean(returns),
        "return_std": jnp.std(returns),
        "advantage_mean": jnp.mean(advantage),
        "advantage_std": jnp.std(advantage),
        "explained_variance": explained_variance(returns, rollout.old_value),
        "progress_mean": jnp.mean(rollout.progress),
        "attitude_cost_mean": jnp.mean(rollout.attitude_cost),
        "rate_cost_mean": jnp.mean(rollout.rate_cost),
        "policy_action_abs_mean": jnp.mean(jnp.abs(rollout.action)),
        "command_action_abs_mean": jnp.mean(jnp.abs(rollout.command_action)),
        "command_action_saturation": jnp.mean(jnp.abs(rollout.command_action) > 0.95),
        "action_cost_mean": jnp.mean(rollout.action_cost),
        "smoothness_cost_mean": jnp.mean(rollout.smoothness_cost),
    }
