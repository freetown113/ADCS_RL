from typing import Callable, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp

from simulators.simplified.control import pd_command_action, policy_to_command
from simulators.simplified.env import EnvState, SatelliteEnv
from simulators.simplified.ppo import deterministic_action



class EvaluationTrajectory(NamedTuple):
    q: jax.Array
    target_q: jax.Array
    omega: jax.Array
    wheel_speed: jax.Array
    wheel_mask: jax.Array
    policy_action: jax.Array
    command_action: jax.Array
    reward: jax.Array
    angle_rad: jax.Array
    rate_norm: jax.Array
    settled: jax.Array
    desired_body_torque: jax.Array
    achieved_body_torque: jax.Array
    allocation_error: jax.Array


def pd_controller(env: SatelliteEnv) -> Callable[[EnvState], jax.Array]:
    return lambda state: pd_command_action(env, state)


def zero_controller(env: SatelliteEnv) -> Callable[[EnvState], jax.Array]:
    def controller(state: EnvState) -> jax.Array:
        batch = state.physical.omega.shape[0]
        return jnp.zeros((batch, env.action_size), dtype=state.physical.omega.dtype)
    return controller


def opposite_pd_controller(env: SatelliteEnv) -> Callable[[EnvState], jax.Array]:
    return lambda state: -pd_command_action(env, state)


def rollout_command_controller(
    env: SatelliteEnv,
    initial_state: EnvState,
    command_controller: Callable[[EnvState], jax.Array],
) -> tuple[EnvState, EvaluationTrajectory]:
    def step_fn(state: EnvState, _):
        command = command_controller(state)
        next_state, reward, _, info = env.step(state, command)
        zeros = jnp.zeros_like(command)
        output = EvaluationTrajectory(
            q=next_state.physical.q,
            target_q=next_state.target_q,
            omega=next_state.physical.omega,
            wheel_speed=next_state.physical.wheel_speed,
            wheel_mask=info.wheel_mask,
            policy_action=zeros,
            command_action=command,
            reward=reward,
            angle_rad=info.reward_terms.attitude_angle_rad,
            rate_norm=info.reward_terms.rate_norm,
            settled=info.settled,
            desired_body_torque=info.desired_body_torque,
            achieved_body_torque=info.achieved_body_torque,
            allocation_error=info.allocation_error_norm,
        )
        return next_state, output
    return jax.lax.scan(step_fn, initial_state, None, length=env.episode_steps)


def evaluate_policy(
    env: SatelliteEnv,
    params: hk.Params,
    apply_fn,
    initial_state: EnvState,
) -> tuple[EnvState, EvaluationTrajectory, dict[str, jax.Array]]:
    def step_fn(state: EnvState, _):
        obs = env.observe(state)
        policy_action, _ = deterministic_action(params, apply_fn, obs)
        command = policy_to_command(env, state, policy_action)
        next_state, reward, _, info = env.step(state, command)
        output = EvaluationTrajectory(
            q=next_state.physical.q,
            target_q=next_state.target_q,
            omega=next_state.physical.omega,
            wheel_speed=next_state.physical.wheel_speed,
            wheel_mask=info.wheel_mask,
            policy_action=policy_action,
            command_action=command,
            reward=reward,
            angle_rad=info.reward_terms.attitude_angle_rad,
            rate_norm=info.reward_terms.rate_norm,
            settled=info.settled,
            desired_body_torque=info.desired_body_torque,
            achieved_body_torque=info.achieved_body_torque,
            allocation_error=info.allocation_error_norm,
        )
        return next_state, output

    final_state, trajectory = jax.lax.scan(
        step_fn, initial_state, None, length=env.episode_steps
    )
    return final_state, trajectory, summarize_trajectory(env, trajectory)


def summarize_trajectory(env: SatelliteEnv, trajectory: EvaluationTrajectory) -> dict[str, jax.Array]:
    episode_return = jnp.sum(trajectory.reward, axis=0)
    final_angle_deg = jnp.rad2deg(trajectory.angle_rad[-1])
    initial_angle_deg = jnp.rad2deg(trajectory.angle_rad[0])
    final_rate = trajectory.rate_norm[-1]
    success = jnp.any(trajectory.settled, axis=0)
    first_success_index = jnp.argmax(trajectory.settled, axis=0)
    has_success = jnp.any(trajectory.settled, axis=0)
    final_second_steps = max(
        1, int(round(1.0 / env.config.physics.control_dt))
    )

    return {
        "episode_return_mean": jnp.mean(episode_return),
        "episode_return_median": jnp.median(episode_return),
        "initial_angle_deg_mean": jnp.mean(initial_angle_deg),
        "final_angle_deg_mean": jnp.mean(final_angle_deg),
        "final_angle_deg_median": jnp.median(final_angle_deg),
        "final_angle_deg_p90": jnp.percentile(final_angle_deg, 90.0),
        "minimum_angle_deg_mean": jnp.mean(jnp.rad2deg(jnp.min(trajectory.angle_rad, axis=0))),
        "angle_improvement_deg_mean": jnp.mean(initial_angle_deg - final_angle_deg),
        "final_rate_mean": jnp.mean(final_rate),
        "final_rate_p90": jnp.percentile(final_rate, 90.0),
        "peak_rate_mean": jnp.mean(jnp.max(trajectory.rate_norm, axis=0)),
        "success_rate": jnp.mean(success.astype(jnp.float32)),
        "settle_time_mean_capped": env.config.physics.control_dt * jnp.mean(
            jnp.where(has_success, first_success_index, trajectory.reward.shape[0])
        ),
        "policy_action_abs_mean": jnp.mean(jnp.abs(trajectory.policy_action)),
        "command_action_abs_mean": jnp.mean(jnp.abs(trajectory.command_action)),
        "command_action_abs_final_1s": jnp.mean(
            jnp.abs(trajectory.command_action[-final_second_steps:])
        ),
        "command_action_saturation": jnp.mean(
            jnp.abs(trajectory.command_action) > 0.95
        ),
        "wheel_fraction_max": jnp.max(
            jnp.abs(trajectory.wheel_speed) / env.config.physics.max_wheel_speed
        ),
        "allocation_error_max": jnp.max(trajectory.allocation_error),
        "wheel_failure_fraction": jnp.mean(
            jnp.any(trajectory.wheel_mask < 0.5, axis=-1).astype(jnp.float32)
        ),
    }


def compare_baselines(env: SatelliteEnv, initial_state: EnvState):
    controllers = {
        "pd": pd_controller(env),
        "zero": zero_controller(env),
        "opposite_pd": opposite_pd_controller(env),
    }
    results = {}
    for name, controller in controllers.items():
        _, trajectory = rollout_command_controller(env, initial_state, controller)
        results[name] = summarize_trajectory(env, trajectory)
    return results
