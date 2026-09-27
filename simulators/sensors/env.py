from typing import NamedTuple

import jax
import jax.numpy as jnp

from simulators.sensors.config import ExperimentConfig, validate_config
from simulators.sensors.estimator import (
    EstimatorState,
    attitude_sigma_rad,
    estimator_substep,
    gyro_bias_sigma_rad_s,
    reset_estimator_state,
)
from simulators.sensors.math3d import attitude_error, axis_angle_to_quat, sample_quaternion_in_cone
from simulators.sensors.orbit import OrbitState, orbit_substep, reset_orbit_state
from simulators.sensors.physics import (
    PhysicalState,
    physics_substep,
    physics_substep_motor_direct,
)
from simulators.sensors.reward import RewardTerms, compute_reward
from simulators.sensors.sensors import (
    SensorState,
    reset_sensor_state,
    sensor_age_seconds,
    sensor_substep,
)


class EnvState(NamedTuple):
    physical: PhysicalState
    orbit: OrbitState
    sensors: SensorState
    estimator: EstimatorState
    target_q: jax.Array
    wheel_mask: jax.Array
    previous_action: jax.Array
    step_count: jax.Array
    success_streak: jax.Array
    fault_wheel: jax.Array
    fault_start_step: jax.Array
    fault_end_step: jax.Array
    fault_key: jax.Array


class EstimatorEvents(NamedTuple):
    star_tracker_processed: jax.Array
    magnetometer_processed: jax.Array
    sun_sensor_processed: jax.Array
    star_tracker_accepted: jax.Array
    magnetometer_accepted: jax.Array
    sun_sensor_accepted: jax.Array

class StepInfo(NamedTuple):
    reward_terms: RewardTerms
    desired_body_torque: jax.Array
    achieved_body_torque: jax.Array
    allocation_error_norm: jax.Array
    motor_torque_max: jax.Array
    wheel_speed_fraction_max: jax.Array
    settled: jax.Array
    wheel_mask: jax.Array
    gyro_measurement: jax.Array
    wheel_speed_measurement: jax.Array
    star_tracker_measurement: jax.Array
    magnetometer_measurement: jax.Array
    sun_sensor_measurement: jax.Array
    gnss_position_measurement: jax.Array
    gnss_velocity_measurement: jax.Array
    gnss_time_measurement: jax.Array
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
    star_tracker_update_processed: jax.Array
    magnetometer_update_processed: jax.Array
    sun_sensor_update_processed: jax.Array
    star_tracker_update_accepted: jax.Array
    magnetometer_update_accepted: jax.Array
    sun_sensor_update_accepted: jax.Array


class SatelliteEnv:
    def __init__(self, config: ExperimentConfig):
        validate_config(config)
        self.config = config
        p = config.physics
        t = config.task
        self.action_size = 4 if config.control.mode == "motor_direct" else 3
        self.episode_steps = int(round(t.episode_seconds / p.control_dt))
        self.success_dwell_steps = max(
            1, int(round(t.success_dwell_seconds / p.control_dt))
        )
        if self.episode_steps < 1:
            raise ValueError("episode_seconds must be at least one control step")

        obs_size = 3 + 3 + 4  # attitude vector, rate, wheel speed
        if config.observation.include_previous_action:
            obs_size += self.action_size
        if config.observation.include_wheel_mask:
            obs_size += 4
        self.observation_size = obs_size

    def _sample_fault_wheel(self, key: jax.Array, batch_size: int) -> jax.Array:
        fault = self.config.faults
        if fault.wheel_selection == "fixed":
            return jnp.full((batch_size,), fault.wheel_index, dtype=jnp.int32)
        return jax.random.randint(key, (batch_size,), 0, 4, dtype=jnp.int32)

    def _mask_from_schedule(
        self,
        step_count: jax.Array,
        fault_wheel: jax.Array,
        start_step: jax.Array,
        end_step: jax.Array,
    ) -> jax.Array:
        active = (step_count >= start_step) & (step_count < end_step)
        wheel_ids = jnp.arange(4, dtype=jnp.int32)[None, :]
        failed = active[:, None] & (wheel_ids == fault_wheel[:, None])
        return 1.0 - failed.astype(jnp.float32)

    def _reset_fault_state(self, key: jax.Array, batch_size: int):
        fault = self.config.faults
        p = self.config.physics
        key_wheel, key_start, key_duration, key_stream = jax.random.split(key, 4)
        fault_wheel = self._sample_fault_wheel(key_wheel, batch_size)
        fault_keys = jax.random.split(key_stream, batch_size)

        never = self.episode_steps + 1
        start_step = jnp.full((batch_size,), never, dtype=jnp.int32)
        end_step = jnp.full((batch_size,), never, dtype=jnp.int32)

        if fault.mode == "permanent":
            start = int(round(fault.start_seconds / p.control_dt))
            start_step = jnp.full((batch_size,), start, dtype=jnp.int32)
            end_step = jnp.full((batch_size,), never, dtype=jnp.int32)
        elif fault.mode == "fixed_interval":
            start = int(round(fault.start_seconds / p.control_dt))
            duration = int(round(fault.duration_seconds / p.control_dt))
            start_step = jnp.full((batch_size,), start, dtype=jnp.int32)
            end_step = jnp.full((batch_size,), start + duration, dtype=jnp.int32)
        elif fault.mode == "random_interval":
            start_seconds = jax.random.uniform(
                key_start,
                (batch_size,),
                minval=fault.random_start_min_seconds,
                maxval=fault.random_start_max_seconds,
            )
            duration_seconds = jax.random.uniform(
                key_duration,
                (batch_size,),
                minval=fault.random_duration_min_seconds,
                maxval=fault.random_duration_max_seconds,
            )
            start_step = jnp.floor(start_seconds / p.control_dt).astype(jnp.int32)
            duration_step = jnp.maximum(
                1, jnp.ceil(duration_seconds / p.control_dt).astype(jnp.int32)
            )
            end_step = start_step + duration_step

        initial_count = jnp.zeros((batch_size,), dtype=jnp.int32)
        if fault.mode in ("permanent", "fixed_interval", "random_interval"):
            mask = self._mask_from_schedule(
                initial_count, fault_wheel, start_step, end_step
            )
        else:
            mask = jnp.ones((batch_size, 4), dtype=jnp.float32)
        return mask, fault_wheel, start_step, end_step, fault_keys

    def _next_stochastic_fault(
        self,
        wheel_mask: jax.Array,
        fault_wheel: jax.Array,
        fault_key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        fault = self.config.faults
        dt = self.config.physics.control_dt
        splits = jax.vmap(lambda k: jax.random.split(k, 4))(fault_key)
        next_key = splits[:, 0]
        fail_u = jax.vmap(jax.random.uniform)(splits[:, 1])
        recover_u = jax.vmap(jax.random.uniform)(splits[:, 2])
        random_wheel = jax.vmap(
            lambda k: jax.random.randint(k, (), 0, 4, dtype=jnp.int32)
        )(splits[:, 3])

        failed_now = jnp.any(wheel_mask < 0.5, axis=-1)
        p_fail = 1.0 - jnp.exp(-fault.failure_rate_per_second * dt)
        p_recover = 1.0 - jnp.exp(-fault.recovery_rate_per_second * dt)
        new_failure = (~failed_now) & (fail_u < p_fail)
        recovery = failed_now & (recover_u < p_recover)

        selected = jnp.where(
            fault.wheel_selection == "random",
            random_wheel,
            jnp.full_like(random_wheel, fault.wheel_index),
        )
        next_fault_wheel = jnp.where(new_failure, selected, fault_wheel)
        next_failed = (failed_now & ~recovery) | new_failure

        wheel_ids = jnp.arange(4, dtype=jnp.int32)[None, :]
        failed_mask = next_failed[:, None] & (
            wheel_ids == next_fault_wheel[:, None]
        )
        next_mask = 1.0 - failed_mask.astype(jnp.float32)
        return next_mask, next_fault_wheel, next_key

    def _next_wheel_mask(self, state: EnvState, next_step_count: jax.Array):
        fault = self.config.faults
        if fault.mode in ("permanent", "fixed_interval", "random_interval"):
            mask = self._mask_from_schedule(
                next_step_count,
                state.fault_wheel,
                state.fault_start_step,
                state.fault_end_step,
            )
            return mask, state.fault_wheel, state.fault_key
        if fault.mode == "stochastic":
            return self._next_stochastic_fault(
                state.wheel_mask, state.fault_wheel, state.fault_key
            )
        return state.wheel_mask, state.fault_wheel, state.fault_key

    def reset(self, key: jax.Array, batch_size: int) -> EnvState:
        task = self.config.task
        key_q, key_omega, key_fault = jax.random.split(key, 3)

        target_q = jnp.tile(
            jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
            (batch_size, 1),
        )

        if task.reset_mode == "fixed":
            axis = jnp.asarray(task.fixed_axis, dtype=jnp.float32)
            axis = axis / (jnp.linalg.norm(axis) + 1.0e-12)
            angle = jnp.deg2rad(jnp.asarray(task.fixed_angle_deg, jnp.float32))
            q_single = axis_angle_to_quat(axis[None, :], angle[None])[0]
            q = jnp.tile(q_single, (batch_size, 1))
        elif task.reset_mode == "cone":
            q = sample_quaternion_in_cone(
                key_q,
                batch_size,
                jnp.deg2rad(task.max_initial_angle_deg),
            )
        else:
            raise ValueError(f"Unknown reset_mode: {task.reset_mode}")

        if task.max_initial_rate > 0.0:
            omega = jax.random.uniform(
                key_omega,
                shape=(batch_size, 3),
                minval=-task.max_initial_rate,
                maxval=task.max_initial_rate,
            )
        else:
            omega = jnp.zeros((batch_size, 3), dtype=jnp.float32)

        wheel_mask, fault_wheel, start_step, end_step, fault_keys = (
            self._reset_fault_state(key_fault, batch_size)
        )
        physical = PhysicalState(
            q=q,
            omega=omega,
            wheel_speed=jnp.zeros((batch_size, 4), dtype=jnp.float32),
        )
        orbit = reset_orbit_state(batch_size, self.config.orbit)
        sensors = reset_sensor_state(
            physical,
            orbit,
            self.config.sensors,
            self.config.physics,
            self.config.orbit,
        )
        estimator = reset_estimator_state(sensors, self.config.estimator)
        return EnvState(
            physical=physical,
            orbit=orbit,
            sensors=sensors,
            estimator=estimator,
            target_q=target_q,
            wheel_mask=wheel_mask,
            previous_action=jnp.zeros(
                (batch_size, self.action_size), dtype=jnp.float32
            ),
            step_count=jnp.zeros((batch_size,), dtype=jnp.int32),
            success_streak=jnp.zeros((batch_size,), dtype=jnp.int32),
            fault_wheel=fault_wheel,
            fault_start_step=start_step,
            fault_end_step=end_step,
            fault_key=fault_keys,
        )

    def observe(self, state: EnvState) -> jax.Array:
        if self.config.estimator.enabled:
            attitude_q = state.estimator.q
            omega = state.estimator.omega
        else:
            attitude_q = state.physical.q
            omega = state.sensors.gyro

        error_q = attitude_error(state.target_q, attitude_q)
        attitude_vector = 2.0 * error_q[..., 1:]
        normalized_omega = omega / self.config.observation.omega_scale
        normalized_wheel_speed = (
            state.sensors.wheel_speed / self.config.physics.max_wheel_speed
        )

        fields = [attitude_vector, normalized_omega, normalized_wheel_speed]
        if self.config.observation.include_previous_action:
            fields.append(state.previous_action)
        if self.config.observation.include_wheel_mask:
            fields.append(state.wheel_mask)
        return jnp.concatenate(fields, axis=-1)

    def step(
        self,
        state: EnvState,
        action: jax.Array,
        external_body_torque: jax.Array | None = None,
    ) -> tuple[EnvState, jax.Array, jax.Array, StepInfo]:
        p = self.config.physics
        action = jnp.clip(action, -1.0, 1.0)

        if external_body_torque is None:
            external_body_torque = jnp.zeros_like(state.physical.omega)

        base_physics_step = state.step_count * p.substeps
        substep_indices = jnp.arange(p.substeps, dtype=jnp.int32)

        if self.config.control.mode == "motor_direct":
            commanded_motor_torque = action * p.motor_control_limit

            def substep(carry, substep_index):
                physical, orbit, sensors, estimator = carry
                next_physical, actuation = physics_substep_motor_direct(
                    physical,
                    commanded_motor_torque,
                    state.wheel_mask,
                    external_body_torque,
                    p,
                )
                next_orbit = orbit_substep(orbit, p.physics_dt, self.config.orbit)
                absolute_step = base_physics_step + substep_index + 1
                next_sensors = sensor_substep(
                    sensors,
                    next_physical,
                    next_orbit,
                    absolute_step,
                    self.config.sensors,
                    p,
                    self.config.orbit,
                )
                if self.config.estimator.enabled:
                    next_estimator = estimator_substep(
                        estimator,
                        next_sensors,
                        absolute_step,
                        p,
                        self.config.orbit,
                        self.config.estimator,
                    )
                else:
                    next_estimator = estimator
                events = EstimatorEvents(
                    star_tracker_processed=(
                        next_estimator.last_star_tracker_step
                        > estimator.last_star_tracker_step
                    ),
                    magnetometer_processed=(
                        next_estimator.last_magnetometer_step
                        > estimator.last_magnetometer_step
                    ),
                    sun_sensor_processed=(
                        next_estimator.last_sun_sensor_step
                        > estimator.last_sun_sensor_step
                    ),
                    star_tracker_accepted=(
                        next_estimator.star_tracker_update_accepted
                        & (
                            next_estimator.last_star_tracker_step
                            > estimator.last_star_tracker_step
                        )
                    ),
                    magnetometer_accepted=(
                        next_estimator.magnetometer_update_accepted
                        & (
                            next_estimator.last_magnetometer_step
                            > estimator.last_magnetometer_step
                        )
                    ),
                    sun_sensor_accepted=(
                        next_estimator.sun_sensor_update_accepted
                        & (
                            next_estimator.last_sun_sensor_step
                            > estimator.last_sun_sensor_step
                        )
                    ),
                )
                return (
                    next_physical, next_orbit, next_sensors, next_estimator
                ), (actuation, events)
        else:
            torque_limit = jnp.asarray(p.body_torque_limit, dtype=action.dtype)
            desired_body_torque = action * torque_limit

            def substep(carry, substep_index):
                physical, orbit, sensors, estimator = carry
                next_physical, actuation = physics_substep(
                    physical,
                    desired_body_torque,
                    state.wheel_mask,
                    external_body_torque,
                    p,
                    measured_wheel_speed=sensors.wheel_speed,
                )
                next_orbit = orbit_substep(orbit, p.physics_dt, self.config.orbit)
                absolute_step = base_physics_step + substep_index + 1
                next_sensors = sensor_substep(
                    sensors,
                    next_physical,
                    next_orbit,
                    absolute_step,
                    self.config.sensors,
                    p,
                    self.config.orbit,
                )
                if self.config.estimator.enabled:
                    next_estimator = estimator_substep(
                        estimator,
                        next_sensors,
                        absolute_step,
                        p,
                        self.config.orbit,
                        self.config.estimator,
                    )
                else:
                    next_estimator = estimator
                events = EstimatorEvents(
                    star_tracker_processed=(
                        next_estimator.last_star_tracker_step
                        > estimator.last_star_tracker_step
                    ),
                    magnetometer_processed=(
                        next_estimator.last_magnetometer_step
                        > estimator.last_magnetometer_step
                    ),
                    sun_sensor_processed=(
                        next_estimator.last_sun_sensor_step
                        > estimator.last_sun_sensor_step
                    ),
                    star_tracker_accepted=(
                        next_estimator.star_tracker_update_accepted
                        & (
                            next_estimator.last_star_tracker_step
                            > estimator.last_star_tracker_step
                        )
                    ),
                    magnetometer_accepted=(
                        next_estimator.magnetometer_update_accepted
                        & (
                            next_estimator.last_magnetometer_step
                            > estimator.last_magnetometer_step
                        )
                    ),
                    sun_sensor_accepted=(
                        next_estimator.sun_sensor_update_accepted
                        & (
                            next_estimator.last_sun_sensor_step
                            > estimator.last_sun_sensor_step
                        )
                    ),
                )
                return (
                    next_physical, next_orbit, next_sensors, next_estimator
                ), (actuation, events)

        (
            next_physical,
            next_orbit,
            next_sensors,
            next_estimator,
        ), (actuation_sequence, estimator_events) = jax.lax.scan(
            substep,
            (state.physical, state.orbit, state.sensors, state.estimator),
            substep_indices,
        )


        previous_error_q = attitude_error(state.target_q, state.physical.q)
        error_q = attitude_error(state.target_q, next_physical.q)
        reward_terms = compute_reward(
            previous_error_q=previous_error_q,
            error_q=error_q,
            omega=next_physical.omega,
            wheel_speed=next_physical.wheel_speed,
            action=action,
            previous_action=state.previous_action,
            physics=p,
            task=self.config.task,
            config=self.config.reward,
        )

        next_streak = jnp.where(
            reward_terms.instantaneous_success,
            state.success_streak + 1,
            jnp.zeros_like(state.success_streak),
        )
        settled = next_streak >= self.success_dwell_steps
        next_step_count = state.step_count + 1
        done = next_step_count >= self.episode_steps
        next_mask, next_fault_wheel, next_fault_key = self._next_wheel_mask(
            state, next_step_count
        )

        next_state = EnvState(
            physical=next_physical,
            orbit=next_orbit,
            sensors=next_sensors,
            estimator=next_estimator,
            target_q=state.target_q,
            wheel_mask=next_mask,
            previous_action=action,
            step_count=next_step_count,
            success_streak=next_streak,
            fault_wheel=next_fault_wheel,
            fault_start_step=state.fault_start_step,
            fault_end_step=state.fault_end_step,
            fault_key=next_fault_key,
        )

        last_actuation = jax.tree_util.tree_map(lambda x: x[-1], actuation_sequence)
        allocation_norm_sequence = jnp.linalg.norm(
            actuation_sequence.allocation_error, axis=-1
        )
        absolute_next_physics_step = next_step_count * p.substeps
        sensor_ages = sensor_age_seconds(
            next_sensors, absolute_next_physics_step, p
        )
        info = StepInfo(
            reward_terms=reward_terms,
            desired_body_torque=last_actuation.desired_body_torque,
            achieved_body_torque=last_actuation.achieved_body_torque,
            allocation_error_norm=jnp.max(allocation_norm_sequence, axis=0),
            motor_torque_max=jnp.max(
                jnp.abs(actuation_sequence.motor_torque), axis=(0, 2)
            ),
            wheel_speed_fraction_max=jnp.max(
                jnp.abs(next_physical.wheel_speed) / p.max_wheel_speed,
                axis=-1,
            ),
            settled=settled,
            wheel_mask=state.wheel_mask,
            gyro_measurement=next_sensors.gyro,
            wheel_speed_measurement=next_sensors.wheel_speed,
            star_tracker_measurement=next_sensors.star_tracker_q,
            magnetometer_measurement=next_sensors.magnetometer_body_t,
            sun_sensor_measurement=next_sensors.sun_direction_body,
            gnss_position_measurement=next_sensors.gnss_position_eci_m,
            gnss_velocity_measurement=next_sensors.gnss_velocity_eci_m_s,
            gnss_time_measurement=next_sensors.gnss_time_s,
            star_tracker_valid=next_sensors.star_tracker_valid,
            magnetometer_valid=next_sensors.magnetometer_valid,
            sun_sensor_valid=next_sensors.sun_sensor_valid,
            gnss_valid=next_sensors.gnss_valid,
            gyro_age_seconds=sensor_ages.gyro,
            wheel_age_seconds=sensor_ages.wheel,
            star_tracker_age_seconds=sensor_ages.star_tracker,
            magnetometer_age_seconds=sensor_ages.magnetometer,
            sun_sensor_age_seconds=sensor_ages.sun_sensor,
            gnss_age_seconds=sensor_ages.gnss,
            estimated_q=next_estimator.q,
            estimated_omega=next_estimator.omega,
            estimated_gyro_bias=next_estimator.gyro_bias,
            estimator_attitude_sigma_rad=attitude_sigma_rad(next_estimator),
            estimator_bias_sigma_rad_s=gyro_bias_sigma_rad_s(next_estimator),
            star_tracker_nis=next_estimator.star_tracker_nis,
            magnetometer_nis=next_estimator.magnetometer_nis,
            sun_sensor_nis=next_estimator.sun_sensor_nis,
            star_tracker_update_processed=jnp.any(
                estimator_events.star_tracker_processed, axis=0
            ),
            magnetometer_update_processed=jnp.any(
                estimator_events.magnetometer_processed, axis=0
            ),
            sun_sensor_update_processed=jnp.any(
                estimator_events.sun_sensor_processed, axis=0
            ),
            star_tracker_update_accepted=jnp.any(
                estimator_events.star_tracker_accepted, axis=0
            ),
            magnetometer_update_accepted=jnp.any(
                estimator_events.magnetometer_accepted, axis=0
            ),
            sun_sensor_update_accepted=jnp.any(
                estimator_events.sun_sensor_accepted, axis=0
            ),
        )
        return next_state, reward_terms.reward, done.astype(jnp.float32), info
