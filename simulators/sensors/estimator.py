from typing import NamedTuple

import jax
import jax.numpy as jnp

from simulators.sensors.config import EstimatorConfig, OrbitConfig, PhysicsConfig
from simulators.sensors.math3d import (
    canonicalize_quat,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    rotate_inertial_to_body,
    rotation_vector_to_quat,
)
from simulators.sensors.orbit import magnetic_field_eci, sun_direction_eci
from simulators.sensors.sensors import SensorState


class EstimatorState(NamedTuple):
    q: jax.Array
    gyro_bias: jax.Array
    omega: jax.Array
    covariance: jax.Array

    last_star_tracker_step: jax.Array
    last_magnetometer_step: jax.Array
    last_sun_sensor_step: jax.Array

    star_tracker_nis: jax.Array
    magnetometer_nis: jax.Array
    sun_sensor_nis: jax.Array

    star_tracker_update_accepted: jax.Array
    magnetometer_update_accepted: jax.Array
    sun_sensor_update_accepted: jax.Array


def _identity_quaternion(batch_size: int, dtype) -> jax.Array:
    q = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    return jnp.broadcast_to(q, (batch_size, 4))


def _skew(vector: jax.Array) -> jax.Array:
    """Returns matrices such that ``skew(v) @ x == v × x``."""
    x, y, z = [vector[..., i] for i in range(3)]
    zero = jnp.zeros_like(x)
    return jnp.stack(
        [
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ],
        axis=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def _normalize(vector: jax.Array, eps: float = 1.0e-12) -> jax.Array:
    return vector / (jnp.linalg.norm(vector, axis=-1, keepdims=True) + eps)


def _initial_covariance(
    batch_size: int,
    dtype,
    config: EstimatorConfig,
) -> jax.Array:
    attitude_sigma = jnp.deg2rad(
        jnp.asarray(config.initial_attitude_sigma_deg, dtype=dtype)
    )
    bias_sigma = jnp.asarray(
        config.initial_gyro_bias_sigma_rad_s, dtype=dtype
    )
    diagonal = jnp.concatenate(
        [
            jnp.full((3,), attitude_sigma**2, dtype=dtype),
            jnp.full((3,), bias_sigma**2, dtype=dtype),
        ]
    )
    return jnp.broadcast_to(jnp.diag(diagonal), (batch_size, 6, 6))


def reset_estimator_state(
    sensors: SensorState,
    config: EstimatorConfig,
) -> EstimatorState:
    """Initializes from delivered telemetry only, never from physical truth."""
    batch_size = sensors.gyro.shape[0]
    dtype = sensors.gyro.dtype
    identity_q = _identity_quaternion(batch_size, dtype)

    initialize_from_star = (
        sensors.star_tracker_valid
        & config.initialize_from_star_tracker
        & config.use_star_tracker
    )
    q = jnp.where(
        initialize_from_star[:, None],
        quat_normalize(sensors.star_tracker_q),
        identity_q,
    )
    bias = jnp.zeros_like(sensors.gyro)
    omega = sensors.gyro - bias
    covariance = _initial_covariance(batch_size, dtype, config)

    minus_one = jnp.full((batch_size,), -1, dtype=jnp.int32)
    last_star = jnp.where(
        initialize_from_star,
        sensors.star_tracker_sample_step,
        minus_one,
    )

    last_mag = jnp.where(
        initialize_from_star & sensors.magnetometer_valid,
        sensors.magnetometer_sample_step,
        minus_one,
    )
    last_sun = jnp.where(
        initialize_from_star & sensors.sun_sensor_valid,
        sensors.sun_sensor_sample_step,
        minus_one,
    )

    zero_score = jnp.zeros((batch_size,), dtype=dtype)
    false = jnp.zeros((batch_size,), dtype=jnp.bool_)
    return EstimatorState(
        q=q,
        gyro_bias=bias,
        omega=omega,
        covariance=covariance,
        last_star_tracker_step=last_star,
        last_magnetometer_step=last_mag,
        last_sun_sensor_step=last_sun,
        star_tracker_nis=zero_score,
        magnetometer_nis=zero_score,
        sun_sensor_nis=zero_score,
        star_tracker_update_accepted=false,
        magnetometer_update_accepted=false,
        sun_sensor_update_accepted=false,
    )


def _propagate(
    state: EstimatorState,
    gyro_measurement: jax.Array,
    dt: float,
    config: EstimatorConfig,
) -> EstimatorState:
    omega = gyro_measurement - state.gyro_bias
    q = quat_normalize(
        quat_multiply(state.q, rotation_vector_to_quat(omega * dt))
    )

    batch_size = omega.shape[0]
    dtype = omega.dtype
    eye3 = jnp.broadcast_to(jnp.eye(3, dtype=dtype), (batch_size, 3, 3))
    eye6 = jnp.broadcast_to(jnp.eye(6, dtype=dtype), (batch_size, 6, 6))
    zeros = jnp.zeros((batch_size, 3, 3), dtype=dtype)

    top = jnp.concatenate([-_skew(omega), -eye3], axis=-1)
    bottom = jnp.concatenate([zeros, zeros], axis=-1)
    continuous_f = jnp.concatenate([top, bottom], axis=-2)
    transition = eye6 + dt * continuous_f

    gyro_noise = jnp.asarray(
        config.gyro_process_noise_rad_s_sqrt_hz, dtype=dtype
    )
    bias_rw = jnp.asarray(
        config.gyro_bias_random_walk_rad_s2_sqrt_hz, dtype=dtype
    )
    process_diagonal = jnp.concatenate(
        [
            jnp.full((3,), gyro_noise**2 * dt, dtype=dtype),
            jnp.full((3,), bias_rw**2 * dt, dtype=dtype),
        ]
    )
    process_covariance = jnp.broadcast_to(
        jnp.diag(process_diagonal), (batch_size, 6, 6)
    )
    covariance = (
        transition @ state.covariance @ jnp.swapaxes(transition, -1, -2)
        + process_covariance
    )
    covariance = _stabilize_covariance(covariance, config)

    return state._replace(
        q=q,
        omega=omega,
        covariance=covariance,
    )


def _stabilize_covariance(
    covariance: jax.Array,
    config: EstimatorConfig,
) -> jax.Array:
    covariance = 0.5 * (
        covariance + jnp.swapaxes(covariance, -1, -2)
    )
    diagonal_indices = jnp.arange(6)
    diagonal = covariance[..., diagonal_indices, diagonal_indices]
    floor = jnp.asarray(config.covariance_floor, dtype=covariance.dtype)
    covariance = covariance.at[
        ..., diagonal_indices, diagonal_indices
    ].set(jnp.maximum(diagonal, floor))
    return covariance


def _kalman_update(
    state: EstimatorState,
    residual: jax.Array,
    measurement_matrix: jax.Array,
    measurement_sigma: float,
    update_mask: jax.Array,
    nis_gate: float,
    config: EstimatorConfig,
) -> tuple[EstimatorState, jax.Array, jax.Array]:
    """Applies one batched three-dimensional error-state measurement update."""
    batch_size = residual.shape[0]
    dtype = residual.dtype
    eye3 = jnp.broadcast_to(jnp.eye(3, dtype=dtype), (batch_size, 3, 3))
    eye6 = jnp.broadcast_to(jnp.eye(6, dtype=dtype), (batch_size, 6, 6))
    measurement_covariance = (
        jnp.asarray(measurement_sigma, dtype=dtype) ** 2 * eye3
    )

    hp = measurement_matrix @ state.covariance
    innovation_covariance = (
        hp @ jnp.swapaxes(measurement_matrix, -1, -2)
        + measurement_covariance
        + jnp.asarray(config.innovation_regularization, dtype=dtype) * eye3
    )
    solved_residual = jnp.linalg.solve(
        innovation_covariance, residual[..., None]
    )[..., 0]
    nis = jnp.sum(residual * solved_residual, axis=-1)

    gain = jnp.swapaxes(
        jnp.linalg.solve(innovation_covariance, hp), -1, -2
    )
    correction = (gain @ residual[..., None])[..., 0]
    delta_theta = correction[..., :3]
    delta_bias = correction[..., 3:]

    q_candidate = quat_normalize(
        quat_multiply(state.q, rotation_vector_to_quat(delta_theta))
    )
    bias_candidate = state.gyro_bias + delta_bias

    kh = gain @ measurement_matrix
    joseph_left = eye6 - kh
    covariance_candidate = (
        joseph_left
        @ state.covariance
        @ jnp.swapaxes(joseph_left, -1, -2)
        + gain
        @ measurement_covariance
        @ jnp.swapaxes(gain, -1, -2)
    )

    reset_jacobian = eye6.at[..., :3, :3].set(
        eye3 - 0.5 * _skew(delta_theta)
    )
    covariance_candidate = (
        reset_jacobian
        @ covariance_candidate
        @ jnp.swapaxes(reset_jacobian, -1, -2)
    )
    covariance_candidate = _stabilize_covariance(
        covariance_candidate, config
    )

    finite = (
        jnp.all(jnp.isfinite(residual), axis=-1)
        & jnp.isfinite(nis)
        & jnp.all(jnp.isfinite(q_candidate), axis=-1)
    )
    accepted = update_mask & finite & (nis <= nis_gate)

    next_state = state._replace(
        q=jnp.where(accepted[:, None], q_candidate, state.q),
        gyro_bias=jnp.where(
            accepted[:, None], bias_candidate, state.gyro_bias
        ),
        covariance=jnp.where(
            accepted[:, None, None],
            covariance_candidate,
            state.covariance,
        ),
    )
    return next_state, nis, accepted


def _packet_age_seconds(
    absolute_physics_step: jax.Array,
    sample_step: jax.Array,
    physics_dt: float,
) -> jax.Array:
    return jnp.maximum(
        absolute_physics_step - sample_step,
        0,
    ).astype(jnp.float32) * physics_dt


def _latency_compensated_quaternion(
    measurement_q: jax.Array,
    omega: jax.Array,
    age_seconds: jax.Array,
    enabled: bool,
) -> jax.Array:
    if not enabled:
        return quat_normalize(measurement_q)
    delta_q = rotation_vector_to_quat(omega * age_seconds[:, None])
    return quat_normalize(quat_multiply(measurement_q, delta_q))


def _latency_compensated_body_vector(
    measurement_body: jax.Array,
    omega: jax.Array,
    age_seconds: jax.Array,
    enabled: bool,
) -> jax.Array:
    if not enabled:
        return measurement_body
    delta_q = rotation_vector_to_quat(omega * age_seconds[:, None])
    return rotate_inertial_to_body(delta_q, measurement_body)


def _star_tracker_update(
    state: EstimatorState,
    sensors: SensorState,
    absolute_physics_step: jax.Array,
    physics: PhysicsConfig,
    config: EstimatorConfig,
) -> EstimatorState:
    new_packet = (
        sensors.star_tracker_sample_step > state.last_star_tracker_step
    )
    update_mask = (
        new_packet
        & sensors.star_tracker_valid
        & config.use_star_tracker
    )
    age = _packet_age_seconds(
        absolute_physics_step,
        sensors.star_tracker_sample_step,
        physics.physics_dt,
    )
    measurement_q = _latency_compensated_quaternion(
        sensors.star_tracker_q,
        state.omega,
        age,
        config.compensate_fixed_latency,
    )
    error_q = canonicalize_quat(
        quat_multiply(quat_conjugate(state.q), measurement_q)
    )
    residual = 2.0 * error_q[..., 1:]

    batch_size = residual.shape[0]
    dtype = residual.dtype
    eye3 = jnp.broadcast_to(jnp.eye(3, dtype=dtype), (batch_size, 3, 3))
    zeros = jnp.zeros((batch_size, 3, 3), dtype=dtype)
    h = jnp.concatenate([eye3, zeros], axis=-1)
    next_state, nis, accepted = _kalman_update(
        state,
        residual,
        h,
        jnp.deg2rad(config.star_tracker_noise_std_deg),
        update_mask,
        config.star_tracker_nis_gate,
        config,
    )
    first_star_solution = (
        new_packet
        & sensors.star_tracker_valid
        & config.use_star_tracker
        & config.hard_acquire_first_star_tracker
        & (state.last_star_tracker_step < 0)
    )
    next_state = next_state._replace(
        q=jnp.where(
            first_star_solution[:, None],
            measurement_q,
            next_state.q,
        ),
        
        covariance=jnp.where(
            first_star_solution[:, None, None],
            state.covariance,
            next_state.covariance,
        ),
    )
    accepted = accepted | first_star_solution
    return next_state._replace(
        last_star_tracker_step=jnp.where(
            new_packet,
            sensors.star_tracker_sample_step,
            state.last_star_tracker_step,
        ),
        star_tracker_nis=jnp.where(
            new_packet, nis, state.star_tracker_nis
        ),
        star_tracker_update_accepted=jnp.where(
            new_packet, accepted, state.star_tracker_update_accepted
        ),
    )


def _vector_update(
    state: EstimatorState,
    measurement_body: jax.Array,
    reference_inertial: jax.Array,
    sample_step: jax.Array,
    valid: jax.Array,
    last_step: jax.Array,
    absolute_physics_step: jax.Array,
    physics: PhysicsConfig,
    measurement_noise_std_deg: float,
    nis_gate: float,
    enabled: bool,
    compensate_latency: bool,
    config: EstimatorConfig,
) -> tuple[EstimatorState, jax.Array, jax.Array, jax.Array]:
    new_packet = sample_step > last_step
    update_mask = new_packet & valid & enabled
    age = _packet_age_seconds(
        absolute_physics_step, sample_step, physics.physics_dt
    )
    measurement = _latency_compensated_body_vector(
        measurement_body,
        state.omega,
        age,
        compensate_latency,
    )
    measurement = _normalize(measurement)
    reference = _normalize(reference_inertial)
    prediction = _normalize(rotate_inertial_to_body(state.q, reference))
    residual = measurement - prediction

    batch_size = residual.shape[0]
    dtype = residual.dtype
    zeros = jnp.zeros((batch_size, 3, 3), dtype=dtype)
    # For right local errors, z_true ~= z_hat + [z_hat x] delta_theta.
    h = jnp.concatenate([_skew(prediction), zeros], axis=-1)
    next_state, nis, accepted = _kalman_update(
        state,
        residual,
        h,
        jnp.deg2rad(measurement_noise_std_deg),
        update_mask,
        nis_gate,
        config,
    )
    next_last_step = jnp.where(new_packet, sample_step, last_step)
    return next_state, next_last_step, nis, accepted


def estimator_substep(
    state: EstimatorState,
    sensors: SensorState,
    absolute_physics_step: jax.Array,
    physics: PhysicsConfig,
    orbit: OrbitConfig,
    config: EstimatorConfig,
) -> EstimatorState:
    """Propagates the MEKF and consumes all newly arrived attitude packets."""
    state = _propagate(state, sensors.gyro, physics.physics_dt, config)
    state = _star_tracker_update(
        state, sensors, absolute_physics_step, physics, config
    )

    magnetic_reference = magnetic_field_eci(
        sensors.gnss_position_eci_m, orbit
    )
    mag_valid = sensors.magnetometer_valid & sensors.gnss_valid
    state, last_mag, mag_nis, mag_accepted = _vector_update(
        state=state,
        measurement_body=sensors.magnetometer_body_t,
        reference_inertial=magnetic_reference,
        sample_step=sensors.magnetometer_sample_step,
        valid=mag_valid,
        last_step=state.last_magnetometer_step,
        absolute_physics_step=absolute_physics_step,
        physics=physics,
        measurement_noise_std_deg=config.magnetometer_noise_std_deg,
        nis_gate=config.magnetometer_nis_gate,
        enabled=config.use_magnetometer,
        compensate_latency=config.compensate_fixed_latency,
        config=config,
    )
    state = state._replace(
        last_magnetometer_step=last_mag,
        magnetometer_nis=jnp.where(
            sensors.magnetometer_sample_step > state.last_magnetometer_step,
            mag_nis,
            state.magnetometer_nis,
        ),
        magnetometer_update_accepted=jnp.where(
            sensors.magnetometer_sample_step > state.last_magnetometer_step,
            mag_accepted,
            state.magnetometer_update_accepted,
        ),
    )

    sun_reference = sun_direction_eci(orbit, sensors.gyro.dtype)
    sun_reference = jnp.broadcast_to(
        sun_reference, sensors.sun_direction_body.shape
    )
    old_last_sun = state.last_sun_sensor_step
    state, last_sun, sun_nis, sun_accepted = _vector_update(
        state=state,
        measurement_body=sensors.sun_direction_body,
        reference_inertial=sun_reference,
        sample_step=sensors.sun_sensor_sample_step,
        valid=sensors.sun_sensor_valid,
        last_step=old_last_sun,
        absolute_physics_step=absolute_physics_step,
        physics=physics,
        measurement_noise_std_deg=config.sun_sensor_noise_std_deg,
        nis_gate=config.sun_sensor_nis_gate,
        enabled=config.use_sun_sensor,
        compensate_latency=config.compensate_fixed_latency,
        config=config,
    )
    state = state._replace(
        last_sun_sensor_step=last_sun,
        sun_sensor_nis=jnp.where(
            sensors.sun_sensor_sample_step > old_last_sun,
            sun_nis,
            state.sun_sensor_nis,
        ),
        sun_sensor_update_accepted=jnp.where(
            sensors.sun_sensor_sample_step > old_last_sun,
            sun_accepted,
            state.sun_sensor_update_accepted,
        ),
    )

    return state._replace(omega=sensors.gyro - state.gyro_bias)


def attitude_sigma_rad(state: EstimatorState) -> jax.Array:
    variance = jnp.trace(state.covariance[..., :3, :3], axis1=-2, axis2=-1) / 3.0
    return jnp.sqrt(jnp.maximum(variance, 0.0))


def gyro_bias_sigma_rad_s(state: EstimatorState) -> jax.Array:
    variance = jnp.trace(state.covariance[..., 3:, 3:], axis1=-2, axis2=-1) / 3.0
    return jnp.sqrt(jnp.maximum(variance, 0.0))
