from typing import Callable, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp

from simulators.fdir.control import pd_command_action, policy_to_command
from simulators.fdir.env import EnvState, SatelliteEnv
from simulators.fdir.math3d import attitude_angle, attitude_error
from simulators.fdir.ppo import deterministic_action


class EvaluationTrajectory(NamedTuple):
    q: jax.Array
    target_q: jax.Array
    target_omega_inertial: jax.Array
    omega: jax.Array
    wheel_speed: jax.Array
    orbit_time_s: jax.Array
    orbit_position_eci_m: jax.Array
    orbit_velocity_eci_m_s: jax.Array
    sun_direction_eci: jax.Array
    sun_visible: jax.Array
    gyro_measurement: jax.Array
    wheel_speed_measurement: jax.Array
    star_tracker_measurement: jax.Array
    magnetometer_measurement: jax.Array
    sun_sensor_measurement: jax.Array
    gnss_position_measurement: jax.Array
    gnss_velocity_measurement: jax.Array
    gnss_time_measurement: jax.Array
    gyro_valid: jax.Array
    wheel_valid: jax.Array
    star_tracker_valid: jax.Array
    magnetometer_valid: jax.Array
    sun_sensor_valid: jax.Array
    gnss_valid: jax.Array
    gyro_age_seconds: jax.Array
    wheel_age_seconds: jax.Array
    star_tracker_age_seconds: jax.Array
    magnetometer_age_seconds: jax.Array
    sun_sensor_age_seconds: jax.Array
    gnss_age_seconds: jax.Array
    estimated_q: jax.Array
    estimated_omega: jax.Array
    estimated_gyro_bias: jax.Array
    estimator_attitude_sigma_rad: jax.Array
    estimator_bias_sigma_rad_s: jax.Array
    star_tracker_nis: jax.Array
    magnetometer_nis: jax.Array
    sun_sensor_nis: jax.Array
    star_tracker_update_accepted: jax.Array
    magnetometer_update_accepted: jax.Array
    sun_sensor_update_accepted: jax.Array
    wheel_mask: jax.Array
    estimated_wheel_authority: jax.Array
    wheel_motor_health: jax.Array
    wheel_tach_health: jax.Array
    star_health: jax.Array
    magnetometer_health: jax.Array
    sun_health: jax.Array
    gnss_health: jax.Array
    estimator_confidence: jax.Array
    supervisory_mode: jax.Array
    mission_phase: jax.Array
    target_elevation_rad: jax.Array
    target_reference_valid: jax.Array
    target_in_beam: jax.Array
    pointing_axis_error_rad: jax.Array
    magnetorquer_commanded_dipole_Am2: jax.Array
    magnetorquer_actual_dipole_Am2: jax.Array
    magnetorquer_torque_body_Nm: jax.Array
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
            target_omega_inertial=next_state.target_omega_inertial,
            omega=next_state.physical.omega,
            wheel_speed=next_state.physical.wheel_speed,
            orbit_time_s=next_state.orbit.time_s,
            orbit_position_eci_m=next_state.orbit.position_eci_m,
            orbit_velocity_eci_m_s=next_state.orbit.velocity_eci_m_s,
            sun_direction_eci=next_state.orbit.sun_direction_eci,
            sun_visible=next_state.orbit.sun_visible,
            gyro_measurement=info.gyro_measurement,
            wheel_speed_measurement=info.wheel_speed_measurement,
            star_tracker_measurement=info.star_tracker_measurement,
            magnetometer_measurement=info.magnetometer_measurement,
            sun_sensor_measurement=info.sun_sensor_measurement,
            gnss_position_measurement=info.gnss_position_measurement,
            gnss_velocity_measurement=info.gnss_velocity_measurement,
            gnss_time_measurement=info.gnss_time_measurement,
            gyro_valid=info.gyro_valid,
            wheel_valid=info.wheel_valid,
            star_tracker_valid=info.star_tracker_valid,
            magnetometer_valid=info.magnetometer_valid,
            sun_sensor_valid=info.sun_sensor_valid,
            gnss_valid=info.gnss_valid,
            gyro_age_seconds=info.gyro_age_seconds,
            wheel_age_seconds=info.wheel_age_seconds,
            star_tracker_age_seconds=info.star_tracker_age_seconds,
            magnetometer_age_seconds=info.magnetometer_age_seconds,
            sun_sensor_age_seconds=info.sun_sensor_age_seconds,
            gnss_age_seconds=info.gnss_age_seconds,
            estimated_q=info.estimated_q,
            estimated_omega=info.estimated_omega,
            estimated_gyro_bias=info.estimated_gyro_bias,
            estimator_attitude_sigma_rad=info.estimator_attitude_sigma_rad,
            estimator_bias_sigma_rad_s=info.estimator_bias_sigma_rad_s,
            star_tracker_nis=info.star_tracker_nis,
            magnetometer_nis=info.magnetometer_nis,
            sun_sensor_nis=info.sun_sensor_nis,
            star_tracker_update_accepted=info.star_tracker_update_accepted,
            magnetometer_update_accepted=info.magnetometer_update_accepted,
            sun_sensor_update_accepted=info.sun_sensor_update_accepted,
            wheel_mask=info.wheel_mask,
            estimated_wheel_authority=info.estimated_wheel_authority,
            wheel_motor_health=info.wheel_motor_health,
            wheel_tach_health=info.wheel_tach_health,
            star_health=info.star_health,
            magnetometer_health=info.magnetometer_health,
            sun_health=info.sun_health,
            gnss_health=info.gnss_health,
            estimator_confidence=info.estimator_confidence,
            supervisory_mode=info.supervisory_mode,
            mission_phase=info.mission_phase,
            target_elevation_rad=info.target_elevation_rad,
            target_reference_valid=info.target_reference_valid,
            target_in_beam=info.target_in_beam,
            pointing_axis_error_rad=info.pointing_axis_error_rad,
            magnetorquer_commanded_dipole_Am2=info.magnetorquer_commanded_dipole_Am2,
            magnetorquer_actual_dipole_Am2=info.magnetorquer_actual_dipole_Am2,
            magnetorquer_torque_body_Nm=info.magnetorquer_torque_body_Nm,
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
            target_omega_inertial=next_state.target_omega_inertial,
            omega=next_state.physical.omega,
            wheel_speed=next_state.physical.wheel_speed,
            orbit_time_s=next_state.orbit.time_s,
            orbit_position_eci_m=next_state.orbit.position_eci_m,
            orbit_velocity_eci_m_s=next_state.orbit.velocity_eci_m_s,
            sun_direction_eci=next_state.orbit.sun_direction_eci,
            sun_visible=next_state.orbit.sun_visible,
            gyro_measurement=info.gyro_measurement,
            wheel_speed_measurement=info.wheel_speed_measurement,
            star_tracker_measurement=info.star_tracker_measurement,
            magnetometer_measurement=info.magnetometer_measurement,
            sun_sensor_measurement=info.sun_sensor_measurement,
            gnss_position_measurement=info.gnss_position_measurement,
            gnss_velocity_measurement=info.gnss_velocity_measurement,
            gnss_time_measurement=info.gnss_time_measurement,
            gyro_valid=info.gyro_valid,
            wheel_valid=info.wheel_valid,
            star_tracker_valid=info.star_tracker_valid,
            magnetometer_valid=info.magnetometer_valid,
            sun_sensor_valid=info.sun_sensor_valid,
            gnss_valid=info.gnss_valid,
            gyro_age_seconds=info.gyro_age_seconds,
            wheel_age_seconds=info.wheel_age_seconds,
            star_tracker_age_seconds=info.star_tracker_age_seconds,
            magnetometer_age_seconds=info.magnetometer_age_seconds,
            sun_sensor_age_seconds=info.sun_sensor_age_seconds,
            gnss_age_seconds=info.gnss_age_seconds,
            estimated_q=info.estimated_q,
            estimated_omega=info.estimated_omega,
            estimated_gyro_bias=info.estimated_gyro_bias,
            estimator_attitude_sigma_rad=info.estimator_attitude_sigma_rad,
            estimator_bias_sigma_rad_s=info.estimator_bias_sigma_rad_s,
            star_tracker_nis=info.star_tracker_nis,
            magnetometer_nis=info.magnetometer_nis,
            sun_sensor_nis=info.sun_sensor_nis,
            star_tracker_update_accepted=info.star_tracker_update_accepted,
            magnetometer_update_accepted=info.magnetometer_update_accepted,
            sun_sensor_update_accepted=info.sun_sensor_update_accepted,
            wheel_mask=info.wheel_mask,
            estimated_wheel_authority=info.estimated_wheel_authority,
            wheel_motor_health=info.wheel_motor_health,
            wheel_tach_health=info.wheel_tach_health,
            star_health=info.star_health,
            magnetometer_health=info.magnetometer_health,
            sun_health=info.sun_health,
            gnss_health=info.gnss_health,
            estimator_confidence=info.estimator_confidence,
            supervisory_mode=info.supervisory_mode,
            mission_phase=info.mission_phase,
            target_elevation_rad=info.target_elevation_rad,
            target_reference_valid=info.target_reference_valid,
            target_in_beam=info.target_in_beam,
            pointing_axis_error_rad=info.pointing_axis_error_rad,
            magnetorquer_commanded_dipole_Am2=info.magnetorquer_commanded_dipole_Am2,
            magnetorquer_actual_dipole_Am2=info.magnetorquer_actual_dipole_Am2,
            magnetorquer_torque_body_Nm=info.magnetorquer_torque_body_Nm,
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
    estimator_attitude_error = attitude_angle(
        attitude_error(trajectory.q, trajectory.estimated_q)
    )
    estimator_rate_error = jnp.linalg.norm(
        trajectory.omega - trajectory.estimated_omega, axis=-1
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
        "gyro_sample_age_mean_seconds": jnp.mean(trajectory.gyro_age_seconds),
        "gyro_sample_age_max_seconds": jnp.max(trajectory.gyro_age_seconds),
        "wheel_sample_age_mean_seconds": jnp.mean(trajectory.wheel_age_seconds),
        "wheel_sample_age_max_seconds": jnp.max(trajectory.wheel_age_seconds),
        "star_tracker_valid_fraction": jnp.mean(
            trajectory.star_tracker_valid.astype(jnp.float32)
        ),
        "magnetometer_valid_fraction": jnp.mean(
            trajectory.magnetometer_valid.astype(jnp.float32)
        ),
        "sun_sensor_valid_fraction": jnp.mean(
            trajectory.sun_sensor_valid.astype(jnp.float32)
        ),
        "gnss_valid_fraction": jnp.mean(
            trajectory.gnss_valid.astype(jnp.float32)
        ),
        "star_tracker_sample_age_max_seconds": jnp.max(
            trajectory.star_tracker_age_seconds
        ),
        "magnetometer_sample_age_max_seconds": jnp.max(
            trajectory.magnetometer_age_seconds
        ),
        "sun_sensor_sample_age_max_seconds": jnp.max(
            trajectory.sun_sensor_age_seconds
        ),
        "gnss_sample_age_max_seconds": jnp.max(trajectory.gnss_age_seconds),
        "estimator_attitude_error_deg_mean": jnp.mean(
            jnp.rad2deg(estimator_attitude_error)
        ),
        "estimator_attitude_error_deg_max": jnp.max(
            jnp.rad2deg(estimator_attitude_error)
        ),
        "estimator_rate_error_mean": jnp.mean(estimator_rate_error),
        "estimator_attitude_sigma_deg_mean": jnp.mean(
            jnp.rad2deg(trajectory.estimator_attitude_sigma_rad)
        ),
        "estimator_bias_sigma_mean": jnp.mean(
            trajectory.estimator_bias_sigma_rad_s
        ),
        "star_tracker_update_acceptance": jnp.mean(
            trajectory.star_tracker_update_accepted.astype(jnp.float32)
        ),
        "magnetometer_update_acceptance": jnp.mean(
            trajectory.magnetometer_update_accepted.astype(jnp.float32)
        ),
        "sun_sensor_update_acceptance": jnp.mean(
            trajectory.sun_sensor_update_accepted.astype(jnp.float32)
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
