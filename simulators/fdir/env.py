import jax
import jax.numpy as jnp

from typing import NamedTuple

from simulators.fdir.config import ExperimentConfig, validate_config
from simulators.fdir.estimator import (
    EstimatorState,
    attitude_sigma_rad,
    estimator_substep,
    gyro_bias_sigma_rad_s,
    reset_estimator_state,
)
from simulators.fdir.math3d import attitude_error, axis_angle_to_quat, sample_quaternion_in_cone, quat_multiply, rotate_body_to_inertial, rotate_inertial_to_body
from simulators.fdir.guidance import GuidanceTarget, body_axis_vector, safe_sun_target, tracking_rate_error_body
from simulators.fdir.fdir import FAILED, FDIRState, filter_sensors_for_estimator, reset_fdir_state, update_fdir
from simulators.fdir.supervisor import DETUMBLE, MOMENTUM_UNLOAD, SAFE_SUN, SupervisorState, mode_one_hot, reset_supervisor_state, update_supervisor
from simulators.fdir.orbit import OrbitState, orbit_state_at_time, orbit_substep
from simulators.fdir.physics import (
    PhysicalState,
    allocate_body_torque_command,
    physics_substep,
    physics_substep_motor_direct,
)
from simulators.fdir.reward import RewardTerms, compute_reward
from simulators.fdir.mission import (
    MissionState,
    find_ground_pass_windows,
    reset_mission_state,
    select_ground_pass_start_times,
    update_mission,
    usable_ground_pass_start_intervals,
)
from simulators.fdir.magnetorquer import (
    MagnetorquerState,
    bdot_detumble_command,
    magnetic_torque_body,
    magnetorquer_substep,
    momentum_unload_command,
    reset_magnetorquer_state,
)
from simulators.fdir.reward import RewardTerms, compute_reward
from simulators.fdir.sensors import (
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
    fdir: FDIRState
    supervisor: SupervisorState
    mission: MissionState
    magnetorquer: MagnetorquerState
    target_q: jax.Array
    target_omega_inertial: jax.Array
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
    # Simulator truth is retained in diagnostics only, never actor/control logic.
    wheel_mask: jax.Array
    estimated_wheel_authority: jax.Array
    wheel_speed_margin: jax.Array
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
    magnetorquer_commanded_dipole_Am2: jax.Array
    magnetorquer_actual_dipole_Am2: jax.Array
    magnetorquer_torque_body_Nm: jax.Array
    target_reference_valid: jax.Array
    pointing_axis_error_rad: jax.Array
    target_in_beam: jax.Array
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

        self._ground_pass_intervals = []
        if (config.guidance.mode in ("ground_target", "ground_station")
                and config.mission.ground_pass_enabled
                and config.mission.ground_pass_reset_mode != "configured"):
            windows = find_ground_pass_windows(config.orbit, config.guidance, config.mission)
            self._ground_pass_intervals = usable_ground_pass_start_intervals(
                windows, config.task.episode_seconds
            )
            if not self._ground_pass_intervals:
                raise ValueError(
                    "No ground-pass window can contain the complete episode above "
                    "the configured entry elevation. Change orbit/target/episode or "
                    "use ground_pass_reset_mode='configured'."
                )

        obs_size = 3 + 3 + 4  # attitude vector, rate, wheel speed
        if config.observation.include_previous_action:
            obs_size += self.action_size
        if config.observation.include_wheel_mask:
            obs_size += 4
        if config.observation.include_estimator_confidence:
            obs_size += 1
        if config.observation.include_supervisory_mode:
            obs_size += 6
        self.observation_size = obs_size

    def _safe_override(self, mission_target: GuidanceTarget, orbit: OrbitState, supervisor_mode: jax.Array) -> GuidanceTarget:
        safe = safe_sun_target(orbit, self.config.guidance)
        use_safe = supervisor_mode == SAFE_SUN
        return GuidanceTarget(
            q_body_to_inertial=jnp.where(use_safe[:, None], safe.q_body_to_inertial, mission_target.q_body_to_inertial),
            omega_inertial_rad_s=jnp.where(use_safe[:, None], safe.omega_inertial_rad_s, mission_target.omega_inertial_rad_s),
            reference_valid=jnp.where(use_safe, safe.reference_valid, mission_target.reference_valid),
            reference_direction_eci=jnp.where(use_safe[:, None], safe.reference_direction_eci, mission_target.reference_direction_eci),
        )


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
        return jnp.where(failed, self.config.faults.fault_torque_fraction, 1.0).astype(jnp.float32)

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

        failed_now = jnp.any(wheel_mask < 0.999, axis=-1)
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
        next_mask = jnp.where(failed_mask, fault.fault_torque_fraction, 1.0).astype(jnp.float32)
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
        key_q, key_omega, key_fault, key_sensor, key_mission = jax.random.split(key, 5)

        start_times = select_ground_pass_start_times(
            key_mission, batch_size, self._ground_pass_intervals, task, self.config.mission
        ) if self.config.guidance.mode in ("ground_target", "ground_station") else jnp.zeros((batch_size,), dtype=jnp.float32)
        orbit = orbit_state_at_time(start_times, self.config.orbit)
        mission_state, mission_target = reset_mission_state(
            orbit, self.config.guidance, self.config.mission, self.config.physics, self.config.orbit
        )
        target = mission_target

        if task.reset_mode == "fixed":
            axis = jnp.asarray(task.fixed_axis, dtype=jnp.float32)
            axis = axis / (jnp.linalg.norm(axis) + 1.0e-12)
            angle = jnp.deg2rad(jnp.asarray(task.fixed_angle_deg, jnp.float32))
            error_single = axis_angle_to_quat(axis[None, :], angle[None])[0]
            error_q = jnp.tile(error_single, (batch_size, 1))
        elif task.reset_mode == "cone":
            error_q = sample_quaternion_in_cone(
                key_q,
                batch_size,
                jnp.deg2rad(task.max_initial_angle_deg),
            )
        else:
            raise ValueError(f"Unknown reset_mode: {task.reset_mode}")

        q = quat_multiply(target.q_body_to_inertial, error_q)

        target_omega_body = rotate_inertial_to_body(
            q, target.omega_inertial_rad_s
        )
        if task.max_initial_rate > 0.0:
            rate_error = jax.random.uniform(
                key_omega,
                shape=(batch_size, 3),
                minval=-task.max_initial_rate,
                maxval=task.max_initial_rate,
            )
        else:
            rate_error = jnp.zeros((batch_size, 3), dtype=jnp.float32)
        omega = target_omega_body + rate_error

        wheel_mask, fault_wheel, start_step, end_step, fault_keys = (
            self._reset_fault_state(key_fault, batch_size)
        )
        physical = PhysicalState(
            q=q,
            omega=omega,
            wheel_speed=jnp.zeros((batch_size, 4), dtype=jnp.float32),
            motor_torque=jnp.zeros((batch_size, 4), dtype=jnp.float32),
        )
        sensors = reset_sensor_state(
            physical,
            orbit,
            key_sensor,
            self.config.sensors,
            self.config.physics,
            self.config.orbit,
        )
        estimator = reset_estimator_state(
            sensors, self.config.physics, self.config.sensors,
            self.config.orbit, self.config.estimator
        )
        fdir = reset_fdir_state(sensors, estimator, self.config.fdir, self.config.sensors)
        supervisor = reset_supervisor_state(
            estimator, fdir, self.config.physics, self.config.fdir, self.config.supervisor
        )
        target = self._safe_override(mission_target, orbit, supervisor.mode)
        magnetorquer = reset_magnetorquer_state(batch_size, q.dtype)
        return EnvState(
            physical=physical,
            orbit=orbit,
            sensors=sensors,
            estimator=estimator,
            fdir=fdir,
            supervisor=supervisor,
            mission=mission_state,
            magnetorquer=magnetorquer,
            target_q=target.q_body_to_inertial,
            target_omega_inertial=target.omega_inertial_rad_s,
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
        rate_error = tracking_rate_error_body(
            attitude_q, omega, state.target_omega_inertial
        )
        normalized_omega = rate_error / self.config.observation.omega_scale
        normalized_wheel_speed = (
            state.sensors.wheel_speed / self.config.physics.max_wheel_speed
        )

        fields = [attitude_vector, normalized_omega, normalized_wheel_speed]
        if self.config.observation.include_previous_action:
            fields.append(state.previous_action)
        if self.config.observation.include_wheel_mask:
            if self.config.observation.wheel_mask_source == "fdir":
                fields.append(state.fdir.wheel_authority_estimate)
            else:
                fields.append(jnp.ones_like(state.fdir.wheel_authority_estimate))
        if self.config.observation.include_estimator_confidence:
            fields.append(state.fdir.estimator_confidence[:, None])
        if self.config.observation.include_supervisory_mode:
            fields.append(mode_one_hot(state.supervisor.mode, dtype=attitude_vector.dtype))
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
                physical, orbit, sensors, estimator, fdir, magnetorquer = carry
                det_cmd = bdot_detumble_command(sensors.gyro, sensors.magnetometer_body_t, self.config.magnetorquer)
                unload_cmd, _ = momentum_unload_command(sensors.wheel_speed, sensors.magnetometer_body_t, p, self.config.magnetorquer)
                mtq_command = jnp.where((state.supervisor.mode == DETUMBLE)[:, None], det_cmd, jnp.where((state.supervisor.mode == MOMENTUM_UNLOAD)[:, None], unload_cmd, 0.0))
                next_magnetorquer = magnetorquer_substep(magnetorquer, mtq_command, p.physics_dt, self.config.magnetorquer)
                true_b_body = rotate_inertial_to_body(physical.q, orbit.magnetic_field_eci_t)
                mtq_torque = magnetic_torque_body(next_magnetorquer.actual_dipole_body_Am2, true_b_body)
                predicted_mtq_torque = magnetic_torque_body(next_magnetorquer.actual_dipole_body_Am2, sensors.magnetometer_body_t)
                ff_motor = allocate_body_torque_command(-predicted_mtq_torque, sensors.wheel_speed, fdir.wheel_authority_estimate, p)
                base_motor_command = commanded_motor_torque + jnp.where((state.supervisor.mode == MOMENTUM_UNLOAD)[:, None], ff_motor, 0.0)
                base_motor_command = jnp.where((state.supervisor.mode == DETUMBLE)[:, None], 0.0, base_motor_command)
                flight_motor_command = jnp.where(
                    fdir.wheel_motor_health == FAILED,
                    0.0,
                    base_motor_command,
                )
                next_physical, actuation = physics_substep_motor_direct(
                    physical,
                    flight_motor_command,
                    state.wheel_mask,
                    external_body_torque + mtq_torque,
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
                estimator_sensors = filter_sensors_for_estimator(next_sensors, fdir)
                if self.config.estimator.enabled:
                    next_estimator = estimator_substep(
                        estimator,
                        estimator_sensors,
                        absolute_step,
                        p,
                        self.config.sensors,
                        self.config.orbit,
                        self.config.estimator,
                    )
                else:
                    next_estimator = estimator
                next_fdir = update_fdir(
                    fdir, next_sensors, next_estimator, actuation.commanded_motor_torque,
                    absolute_step, p, self.config.sensors, self.config.estimator, self.config.fdir,
                )
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
                    next_physical, next_orbit, next_sensors, next_estimator, next_fdir, next_magnetorquer
                ), (actuation, events, mtq_command, next_magnetorquer.actual_dipole_body_Am2, mtq_torque)
        else:
            torque_limit = jnp.asarray(p.body_torque_limit, dtype=action.dtype)
            desired_body_torque = action * torque_limit

            def substep(carry, substep_index):
                physical, orbit, sensors, estimator, fdir, magnetorquer = carry
                det_cmd = bdot_detumble_command(sensors.gyro, sensors.magnetometer_body_t, self.config.magnetorquer)
                unload_cmd, _ = momentum_unload_command(sensors.wheel_speed, sensors.magnetometer_body_t, p, self.config.magnetorquer)
                mtq_command = jnp.where((state.supervisor.mode == DETUMBLE)[:, None], det_cmd, jnp.where((state.supervisor.mode == MOMENTUM_UNLOAD)[:, None], unload_cmd, 0.0))
                next_magnetorquer = magnetorquer_substep(magnetorquer, mtq_command, p.physics_dt, self.config.magnetorquer)
                true_b_body = rotate_inertial_to_body(physical.q, orbit.magnetic_field_eci_t)
                mtq_torque = magnetic_torque_body(next_magnetorquer.actual_dipole_body_Am2, true_b_body)
                predicted_mtq_torque = magnetic_torque_body(next_magnetorquer.actual_dipole_body_Am2, sensors.magnetometer_body_t)
                body_command = desired_body_torque + jnp.where((state.supervisor.mode == MOMENTUM_UNLOAD)[:, None], -predicted_mtq_torque, 0.0)
                body_command = jnp.where((state.supervisor.mode == DETUMBLE)[:, None], 0.0, body_command)
                next_physical, actuation = physics_substep(
                    physical,
                    body_command,
                    fdir.wheel_authority_estimate,
                    external_body_torque + mtq_torque,
                    p,
                    measured_wheel_speed=sensors.wheel_speed,
                    physical_wheel_authority=state.wheel_mask,
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
                estimator_sensors = filter_sensors_for_estimator(next_sensors, fdir)
                if self.config.estimator.enabled:
                    next_estimator = estimator_substep(
                        estimator,
                        estimator_sensors,
                        absolute_step,
                        p,
                        self.config.sensors,
                        self.config.orbit,
                        self.config.estimator,
                    )
                else:
                    next_estimator = estimator
                next_fdir = update_fdir(
                    fdir, next_sensors, next_estimator, actuation.commanded_motor_torque,
                    absolute_step, p, self.config.sensors, self.config.estimator, self.config.fdir,
                )
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
                    next_physical, next_orbit, next_sensors, next_estimator, next_fdir, next_magnetorquer
                ), (actuation, events, mtq_command, next_magnetorquer.actual_dipole_body_Am2, mtq_torque)

        (
            next_physical,
            next_orbit,
            next_sensors,
            next_estimator,
            next_fdir,
            next_magnetorquer,
        ), (actuation_sequence, estimator_events, mtq_command_sequence, mtq_actual_sequence, mtq_torque_sequence) = jax.lax.scan(
            substep,
            (state.physical, state.orbit, state.sensors, state.estimator, state.fdir, state.magnetorquer),
            substep_indices,
        )


        next_supervisor = update_supervisor(
            state.supervisor, next_estimator, next_fdir, p, self.config.fdir, self.config.supervisor
        )
        next_mission, mission_target = update_mission(
            state.mission, next_orbit, self.config.guidance, self.config.mission, p, self.config.orbit
        )
        next_target = self._safe_override(mission_target, next_orbit, next_supervisor.mode)
        previous_error_q = attitude_error(state.target_q, state.physical.q)
        error_q = attitude_error(next_target.q_body_to_inertial, next_physical.q)
        true_rate_error = tracking_rate_error_body(
            next_physical.q, next_physical.omega, next_target.omega_inertial_rad_s
        )
        reward_terms = compute_reward(
            previous_error_q=previous_error_q,
            error_q=error_q,
            rate_error=true_rate_error,
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
            fdir=next_fdir,
            supervisor=next_supervisor,
            mission=next_mission,
            magnetorquer=next_magnetorquer,
            target_q=next_target.q_body_to_inertial,
            target_omega_inertial=next_target.omega_inertial_rad_s,
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
        axis_name = (
            self.config.guidance.earth_tracking_body_axis
            if self.config.guidance.mode in ("ground_target", "ground_station")
            else self.config.guidance.sun_pointing_body_axis
            if self.config.guidance.mode == "sun_pointing"
            else "+Z"
        )
        body_axis = jnp.broadcast_to(
            body_axis_vector(axis_name, next_physical.q.dtype), next_physical.omega.shape
        )
        actual_axis_eci = rotate_body_to_inertial(next_physical.q, body_axis)
        axis_dot = jnp.sum(actual_axis_eci * next_target.reference_direction_eci, axis=-1)
        pointing_axis_error = jnp.arccos(jnp.clip(axis_dot, -1.0, 1.0))
        target_in_beam = (
            next_target.reference_valid
            & (pointing_axis_error <= jnp.deg2rad(self.config.guidance.antenna_half_beamwidth_deg))
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
            estimated_wheel_authority=next_fdir.wheel_authority_estimate,
            wheel_speed_margin=next_fdir.wheel_speed_margin,
            wheel_motor_health=next_fdir.wheel_motor_health,
            wheel_tach_health=next_fdir.wheel_tach_health,
            star_health=next_fdir.star_health,
            magnetometer_health=next_fdir.magnetometer_health,
            sun_health=next_fdir.sun_health,
            gnss_health=next_fdir.gnss_health,
            estimator_confidence=next_fdir.estimator_confidence,
            supervisory_mode=next_supervisor.mode,
            mission_phase=next_mission.phase,
            target_elevation_rad=next_mission.target_elevation_rad,
            magnetorquer_commanded_dipole_Am2=mtq_command_sequence[-1],
            magnetorquer_actual_dipole_Am2=mtq_actual_sequence[-1],
            magnetorquer_torque_body_Nm=mtq_torque_sequence[-1],
            target_reference_valid=next_target.reference_valid,
            pointing_axis_error_rad=pointing_axis_error,
            target_in_beam=target_in_beam,
            gyro_measurement=next_sensors.gyro,
            wheel_speed_measurement=next_sensors.wheel_speed,
            star_tracker_measurement=next_sensors.star_tracker_q,
            magnetometer_measurement=next_sensors.magnetometer_body_t,
            sun_sensor_measurement=next_sensors.sun_direction_body,
            gnss_position_measurement=next_sensors.gnss_position_eci_m,
            gnss_velocity_measurement=next_sensors.gnss_velocity_eci_m_s,
            gnss_time_measurement=next_sensors.gnss_time_s,
            gyro_valid=next_sensors.gyro_valid,
            wheel_valid=next_sensors.wheel_valid,
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
