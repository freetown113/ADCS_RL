from typing import NamedTuple

import jax
import jax.numpy as jnp

from simulators.fdir.config import PhysicsConfig, RewardConfig, TaskConfig
from simulators.fdir.math3d import attitude_angle, canonicalize_quat


class RewardTerms(NamedTuple):
    reward: jax.Array
    progress: jax.Array
    attitude_cost: jax.Array
    rate_cost: jax.Array
    action_cost: jax.Array
    smoothness_cost: jax.Array
    wheel_speed_cost: jax.Array
    instantaneous_success: jax.Array
    attitude_angle_rad: jax.Array
    rate_norm: jax.Array


def huber(x: jax.Array, delta: float) -> jax.Array:
    abs_x = jnp.abs(x)
    quadratic = 0.5 * jnp.square(x)
    linear = delta * (abs_x - 0.5 * delta)
    return jnp.where(abs_x <= delta, quadratic, linear)


def compute_reward(
    *,
    previous_error_q: jax.Array,
    error_q: jax.Array,
    rate_error: jax.Array,
    wheel_speed: jax.Array,
    action: jax.Array,
    previous_action: jax.Array,
    physics: PhysicsConfig,
    task: TaskConfig,
    config: RewardConfig,
) -> RewardTerms:
    previous_error_q = canonicalize_quat(previous_error_q)
    error_q = canonicalize_quat(error_q)
    angle = attitude_angle(error_q)

    previous_attitude_cost = 4.0 * jnp.sum(jnp.square(previous_error_q[..., 1:]), axis=-1)
    attitude_cost = 4.0 * jnp.sum(jnp.square(error_q[..., 1:]), axis=-1)
    progress = previous_attitude_cost - attitude_cost
    normalized_rate = rate_error / config.rate_scale
    rate_cost = jnp.sum(huber(normalized_rate, config.huber_delta), axis=-1)
    action_cost = jnp.mean(jnp.square(action), axis=-1)
    smoothness_cost = jnp.mean(jnp.square(action - previous_action), axis=-1)
    wheel_fraction = wheel_speed / physics.max_wheel_speed
    wheel_speed_cost = jnp.mean(jnp.square(wheel_fraction), axis=-1)
    rate_norm = jnp.linalg.norm(rate_error, axis=-1)

    success = (
        (angle <= jnp.deg2rad(task.success_angle_deg))
        & (rate_norm <= task.success_rate)
    )

    total_cost = (
        config.attitude_weight * attitude_cost
        + config.rate_weight * rate_cost
        + config.action_weight * action_cost
        + config.smoothness_weight * smoothness_cost
        + config.wheel_speed_weight * wheel_speed_cost
    )
    reward_rate = -total_cost + config.success_bonus_per_second * success.astype(
        rate_error.dtype
    )
    reward = config.progress_weight * progress + physics.control_dt * reward_rate

    return RewardTerms(
        reward=reward,
        progress=progress,
        attitude_cost=attitude_cost,
        rate_cost=rate_cost,
        action_cost=action_cost,
        smoothness_cost=smoothness_cost,
        wheel_speed_cost=wheel_speed_cost,
        instantaneous_success=success,
        attitude_angle_rad=angle,
        rate_norm=rate_norm,
    )
