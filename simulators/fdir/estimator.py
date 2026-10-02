from typing import NamedTuple

import jax
import jax.numpy as jnp

from simulators.fdir.config import EstimatorConfig, OrbitConfig, PhysicsConfig, SensorConfig
from simulators.fdir.math3d import (
    canonicalize_quat,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_to_rotation_vector,
    rotate_inertial_to_body,
    rotation_matrix_to_quat,
    rotation_vector_to_quat,
)
from simulators.fdir.orbit import magnetic_field_eci, sun_direction_eci
from simulators.fdir.sensors import SensorState


class ReplayHistory(NamedTuple):
    step: jax.Array
    q: jax.Array
    gyro_bias: jax.Array
    covariance: jax.Array
    acquired: jax.Array

    gyro_sample: jax.Array
    gyro_present: jax.Array
    gyro_valid: jax.Array
    gyro_zoh: jax.Array

    star_q: jax.Array
    star_present: jax.Array
    star_valid: jax.Array

    mag_body: jax.Array
    mag_reference_eci: jax.Array
    mag_present: jax.Array
    mag_valid: jax.Array

    sun_body: jax.Array
    sun_reference_eci: jax.Array
    sun_present: jax.Array
    sun_valid: jax.Array


class EstimatorState(NamedTuple):
    q: jax.Array
    gyro_bias: jax.Array
    omega: jax.Array
    covariance: jax.Array
    attitude_acquired: jax.Array

    last_gyro_step: jax.Array
    last_star_tracker_step: jax.Array
    last_magnetometer_step: jax.Array
    last_sun_sensor_step: jax.Array

    star_tracker_nis: jax.Array
    magnetometer_nis: jax.Array
    sun_sensor_nis: jax.Array

    star_tracker_update_accepted: jax.Array
    magnetometer_update_accepted: jax.Array
    sun_sensor_update_accepted: jax.Array

    history: ReplayHistory


def _identity_quaternion(batch_size: int, dtype) -> jax.Array:
    return jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=dtype), (batch_size, 4))


def _skew(vector: jax.Array) -> jax.Array:
    x, y, z = [vector[..., i] for i in range(3)]
    zero = jnp.zeros_like(x)
    return jnp.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=-1).reshape(
        vector.shape[:-1] + (3, 3)
    )


def _normalize(vector: jax.Array, eps: float = 1.0e-12) -> jax.Array:
    return vector / (jnp.linalg.norm(vector, axis=-1, keepdims=True) + eps)


def _history_length(config: EstimatorConfig, physics: PhysicsConfig) -> int:
    return int(round(config.fixed_lag_history_seconds / physics.physics_dt)) + 1


def _initial_covariance(batch_size: int, dtype, config: EstimatorConfig) -> jax.Array:
    attitude_sigma = jnp.deg2rad(jnp.asarray(config.initial_attitude_sigma_deg, dtype=dtype))
    bias_sigma = jnp.asarray(config.initial_gyro_bias_sigma_rad_s, dtype=dtype)
    diagonal = jnp.concatenate([
        jnp.full((3,), attitude_sigma**2, dtype=dtype),
        jnp.full((3,), bias_sigma**2, dtype=dtype),
    ])
    return jnp.broadcast_to(jnp.diag(diagonal), (batch_size, 6, 6))


def _stabilize_covariance(covariance: jax.Array, config: EstimatorConfig) -> jax.Array:
    covariance = 0.5 * (covariance + jnp.swapaxes(covariance, -1, -2))
    idx = jnp.arange(6)
    diagonal = covariance[..., idx, idx]
    floor = jnp.asarray(config.covariance_floor, dtype=covariance.dtype)
    return covariance.at[..., idx, idx].set(jnp.maximum(diagonal, floor))


def _triad_attitude(meas1_body: jax.Array, ref1_eci: jax.Array, meas2_body: jax.Array,
                    ref2_eci: jax.Array) -> tuple[jax.Array, jax.Array]:
    b1 = _normalize(meas1_body)
    i1 = _normalize(ref1_eci)
    b2_cross = jnp.cross(b1, _normalize(meas2_body))
    i2_cross = jnp.cross(i1, _normalize(ref2_eci))
    b_cross_norm = jnp.linalg.norm(b2_cross, axis=-1)
    i_cross_norm = jnp.linalg.norm(i2_cross, axis=-1)
    valid = (b_cross_norm > 0.1) & (i_cross_norm > 0.1)
    b2 = _normalize(b2_cross)
    i2 = _normalize(i2_cross)
    b3 = jnp.cross(b1, b2)
    i3 = jnp.cross(i1, i2)
    body_triad = jnp.stack([b1, b2, b3], axis=-1)
    inertial_triad = jnp.stack([i1, i2, i3], axis=-1)
    rotation = inertial_triad @ jnp.swapaxes(body_triad, -1, -2)
    return rotation_matrix_to_quat(rotation), valid


def _set_attitude_covariance(covariance: jax.Array, sigma_rad: jax.Array,
                             config: EstimatorConfig) -> jax.Array:
    dtype = covariance.dtype
    eye3 = jnp.broadcast_to(jnp.eye(3, dtype=dtype), covariance.shape[:-2] + (3, 3))
    out = covariance.at[..., :3, :3].set((sigma_rad**2)[..., None, None] * eye3)
    out = out.at[..., :3, 3:].set(jnp.zeros_like(out[..., :3, 3:]))
    out = out.at[..., 3:, :3].set(jnp.zeros_like(out[..., 3:, :3]))
    return _stabilize_covariance(out, config)


def reset_estimator_state(
    sensors: SensorState,
    physics: PhysicsConfig,
    sensor_config: SensorConfig,
    orbit_config: OrbitConfig,
    config: EstimatorConfig,
) -> EstimatorState:
    """Initializes from delivered telemetry only, never from plant truth."""
    batch_size = sensors.gyro.shape[0]
    dtype = sensors.gyro.dtype
    identity_q = _identity_quaternion(batch_size, dtype)
    covariance = _initial_covariance(batch_size, dtype, config)

    star_init = sensors.star_tracker_valid & config.initialize_from_star_tracker & config.use_star_tracker
    q_star = quat_normalize(sensors.star_tracker_q)

    # If no star solution is immediately available, two non-collinear vector
    # observations can still provide a global TRIAD acquisition.
    mag_ref = magnetic_field_eci(sensors.gnss_position_eci_m, orbit_config)
    sun_ref = jnp.broadcast_to(sun_direction_eci(orbit_config, dtype), sensors.sun_direction_body.shape)
    q_triad, triad_geometry_ok = _triad_attitude(
        sensors.magnetometer_body_t, mag_ref,
        sensors.sun_direction_body, sun_ref,
    )
    triad_init = (
        (~star_init)
        & sensors.magnetometer_valid
        & sensors.sun_sensor_valid
        & sensors.gnss_valid
        & config.use_magnetometer
        & config.use_sun_sensor
        & triad_geometry_ok
    )
    q = jnp.where(star_init[:, None], q_star, jnp.where(triad_init[:, None], q_triad, identity_q))
    acquired = star_init | triad_init

    star_sigma = jnp.full((batch_size,), jnp.deg2rad(config.star_hard_acquisition_sigma_deg), dtype=dtype)
    triad_sigma = jnp.full(
        (batch_size,),
        jnp.deg2rad(max(config.magnetometer_noise_std_deg, config.sun_sensor_noise_std_deg)),
        dtype=dtype,
    )
    covariance = jnp.where(
        star_init[:, None, None],
        _set_attitude_covariance(covariance, star_sigma, config),
        jnp.where(
            triad_init[:, None, None],
            _set_attitude_covariance(covariance, triad_sigma, config),
            covariance,
        ),
    )

    bias = jnp.zeros_like(sensors.gyro)
    gyro0 = jnp.where(sensors.gyro_valid[:, None], sensors.gyro, jnp.zeros_like(sensors.gyro))
    omega = gyro0 - bias

    length = _history_length(config, physics)
    steps = -jnp.ones((length, batch_size), dtype=jnp.int32)
    steps = steps.at[-1].set(jnp.zeros((batch_size,), dtype=jnp.int32))

    def hist_state(value: jax.Array) -> jax.Array:
        return jnp.broadcast_to(value[None, ...], (length,) + value.shape)

    zeros3 = jnp.zeros((length, batch_size, 3), dtype=dtype)
    zeros4 = jnp.zeros((length, batch_size, 4), dtype=dtype)
    false = jnp.zeros((length, batch_size), dtype=jnp.bool_)
    gyro_sample = zeros3.at[-1].set(gyro0)
    gyro_present = false.at[-1].set(sensors.gyro_valid)
    gyro_valid = false.at[-1].set(sensors.gyro_valid)
    gyro_zoh = zeros3.at[-1].set(gyro0)

    history = ReplayHistory(
        step=steps,
        q=hist_state(q),
        gyro_bias=hist_state(bias),
        covariance=hist_state(covariance),
        acquired=hist_state(acquired),
        gyro_sample=gyro_sample,
        gyro_present=gyro_present,
        gyro_valid=gyro_valid,
        gyro_zoh=gyro_zoh,
        star_q=zeros4,
        star_present=false,
        star_valid=false,
        mag_body=zeros3,
        mag_reference_eci=zeros3,
        mag_present=false,
        mag_valid=false,
        sun_body=zeros3,
        sun_reference_eci=zeros3,
        sun_present=false,
        sun_valid=false,
    )

    minus_one = -jnp.ones((batch_size,), dtype=jnp.int32)
    zero_score = jnp.zeros((batch_size,), dtype=dtype)
    false_b = jnp.zeros((batch_size,), dtype=jnp.bool_)
    return EstimatorState(
        q=q,
        gyro_bias=bias,
        omega=omega,
        covariance=covariance,
        attitude_acquired=acquired,
        last_gyro_step=jnp.where(sensors.gyro_valid, sensors.gyro_sample_step, minus_one),
        last_star_tracker_step=jnp.where(star_init, sensors.star_tracker_sample_step, minus_one),
        last_magnetometer_step=jnp.where(triad_init, sensors.magnetometer_sample_step, minus_one),
        last_sun_sensor_step=jnp.where(triad_init, sensors.sun_sensor_sample_step, minus_one),
        star_tracker_nis=zero_score,
        magnetometer_nis=zero_score,
        sun_sensor_nis=zero_score,
        star_tracker_update_accepted=false_b,
        magnetometer_update_accepted=false_b,
        sun_sensor_update_accepted=false_b,
        history=history,
    )


def _propagate_arrays(q: jax.Array, bias: jax.Array, covariance: jax.Array, gyro_measurement: jax.Array,
                      dt: float, config: EstimatorConfig) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    omega = gyro_measurement - bias
    q_next = quat_normalize(quat_multiply(q, rotation_vector_to_quat(omega * dt)))

    batch = omega.shape[0]
    dtype = omega.dtype
    eye3 = jnp.broadcast_to(jnp.eye(3, dtype=dtype), (batch, 3, 3))
    zeros = jnp.zeros((batch, 3, 3), dtype=dtype)
    k = _skew(omega)
    k2 = k @ k
    w = jnp.linalg.norm(omega, axis=-1)
    theta = w * dt
    w2 = jnp.square(w)
    w3 = w2 * w
    small = w < 1.0e-6

    a = jnp.where(small, dt - w2 * dt**3 / 6.0, jnp.sin(theta) / jnp.maximum(w, 1.0e-12))
    b = jnp.where(small, 0.5 * dt**2 - w2 * dt**4 / 24.0,
                  (1.0 - jnp.cos(theta)) / jnp.maximum(w2, 1.0e-12))
    c = jnp.where(small, dt**3 / 6.0 - w2 * dt**5 / 120.0,
                  (theta - jnp.sin(theta)) / jnp.maximum(w3, 1.0e-12))
    phi11 = eye3 - a[:, None, None] * k + b[:, None, None] * k2
    integral = dt * eye3 - b[:, None, None] * k + c[:, None, None] * k2
    phi12 = -integral
    top = jnp.concatenate([phi11, phi12], axis=-1)
    bottom = jnp.concatenate([zeros, eye3], axis=-1)
    phi = jnp.concatenate([top, bottom], axis=-2)

    qg = jnp.asarray(config.gyro_process_noise_rad_s_sqrt_hz, dtype=dtype) ** 2
    qb = jnp.asarray(config.gyro_bias_random_walk_rad_s2_sqrt_hz, dtype=dtype) ** 2
    q_theta = (qg * dt + qb * dt**3 / 3.0) * eye3
    q_cross = (-qb * dt**2 / 2.0) * eye3
    q_bias = (qb * dt) * eye3
    qd = jnp.concatenate([
        jnp.concatenate([q_theta, q_cross], axis=-1),
        jnp.concatenate([q_cross, q_bias], axis=-1),
    ], axis=-2)
    p_next = phi @ covariance @ jnp.swapaxes(phi, -1, -2) + qd
    return q_next, bias, _stabilize_covariance(p_next, config), omega


def _kalman_update_arrays(q: jax.Array, bias: jax.Array, covariance: jax.Array, residual: jax.Array,
                          h: jax.Array, measurement_sigma: float, update_mask: jax.Array,
                          nis_gate: float, config: EstimatorConfig) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    batch = residual.shape[0]
    dim = residual.shape[-1]
    dtype = residual.dtype
    eye_m = jnp.broadcast_to(jnp.eye(dim, dtype=dtype), (batch, dim, dim))
    eye6 = jnp.broadcast_to(jnp.eye(6, dtype=dtype), (batch, 6, 6))
    r = (jnp.asarray(measurement_sigma, dtype=dtype) ** 2) * eye_m
    hp = h @ covariance
    s = hp @ jnp.swapaxes(h, -1, -2) + r + config.innovation_regularization * eye_m
    solved = jnp.linalg.solve(s, residual[..., None])[..., 0]
    nis = jnp.sum(residual * solved, axis=-1)
    gain = jnp.swapaxes(jnp.linalg.solve(s, hp), -1, -2)
    correction = (gain @ residual[..., None])[..., 0]
    dtheta, dbias = correction[..., :3], correction[..., 3:]
    q_candidate = quat_normalize(quat_multiply(q, rotation_vector_to_quat(dtheta)))
    bias_candidate = bias + dbias

    left = eye6 - gain @ h
    p_candidate = left @ covariance @ jnp.swapaxes(left, -1, -2) + gain @ r @ jnp.swapaxes(gain, -1, -2)
    reset = eye6.at[..., :3, :3].set(
        jnp.broadcast_to(jnp.eye(3, dtype=dtype), (batch, 3, 3)) - 0.5 * _skew(dtheta)
    )
    p_candidate = _stabilize_covariance(reset @ p_candidate @ jnp.swapaxes(reset, -1, -2), config)
    finite = jnp.all(jnp.isfinite(residual), axis=-1) & jnp.isfinite(nis) & jnp.all(jnp.isfinite(q_candidate), axis=-1)
    accepted = update_mask & finite & (nis <= nis_gate)
    return (
        jnp.where(accepted[:, None], q_candidate, q),
        jnp.where(accepted[:, None], bias_candidate, bias),
        jnp.where(accepted[:, None, None], p_candidate, covariance),
        nis,
        accepted,
    )


def _star_update(q: jax.Array, bias: jax.Array, p: jax.Array, acquired: jax.Array, measurement_q: jax.Array,
                 present: jax.Array, valid: jax.Array, config: EstimatorConfig) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    use = present & valid & config.use_star_tracker
    first = use & (~acquired) & config.hard_acquire_first_star_tracker
    acquisition_sigma = jnp.full(
        (q.shape[0],), jnp.deg2rad(config.star_hard_acquisition_sigma_deg), dtype=q.dtype
    )
    p_acquired = _set_attitude_covariance(p, acquisition_sigma, config)
    q0 = jnp.where(first[:, None], quat_normalize(measurement_q), q)
    p0 = jnp.where(first[:, None, None], p_acquired, p)
    acquired0 = acquired | first

    error_q = canonicalize_quat(quat_multiply(quat_conjugate(q0), quat_normalize(measurement_q)))
    residual = quat_to_rotation_vector(error_q)
    eye3 = jnp.broadcast_to(jnp.eye(3, dtype=q.dtype), (q.shape[0], 3, 3))
    zeros = jnp.zeros_like(eye3)
    h = jnp.concatenate([eye3, zeros], axis=-1)
    regular = use & acquired0 & (~first)
    q1, b1, p1, nis, accepted_regular = _kalman_update_arrays(
        q0, bias, p0, residual, h, jnp.deg2rad(config.star_tracker_noise_std_deg),
        regular, config.star_tracker_nis_gate, config,
    )
    return q1, b1, p1, acquired0, nis, first | accepted_regular


def _tangent_basis(direction: jax.Array) -> jax.Array:
    z_axis = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0], dtype=direction.dtype), direction.shape)
    x_axis = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0], dtype=direction.dtype), direction.shape)
    helper = jnp.where((jnp.abs(direction[..., 2]) < 0.9)[..., None], z_axis, x_axis)
    e1 = _normalize(jnp.cross(direction, helper))
    e2 = jnp.cross(direction, e1)
    return jnp.stack([e1, e2], axis=-2)


def _vector_update(q: jax.Array, bias: jax.Array, p: jax.Array, measurement_body: jax.Array,
                   reference_eci: jax.Array, present: jax.Array, valid: jax.Array, enabled: bool,
                   sigma_deg: float, nis_gate: float, acquired: jax.Array,
                   config: EstimatorConfig) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    measurement = _normalize(measurement_body)
    reference = _normalize(reference_eci)
    prediction = _normalize(rotate_inertial_to_body(q, reference))
    tangent = _tangent_basis(prediction)  # [..., 2, 3]
    residual3 = measurement - prediction
    residual = (tangent @ residual3[..., None])[..., 0]
    zeros = jnp.zeros((q.shape[0], 3, 3), dtype=q.dtype)
    h3 = jnp.concatenate([_skew(prediction), zeros], axis=-1)
    h = tangent @ h3
    update_mask = present & valid & enabled & acquired
    q1, b1, p1, nis, accepted = _kalman_update_arrays(
        q, bias, p, residual, h, jnp.deg2rad(sigma_deg), update_mask, nis_gate, config
    )
    return q1, b1, p1, nis, accepted


def _apply_measurements(q: jax.Array, bias: jax.Array, p: jax.Array, acquired: jax.Array,
                        history: ReplayHistory, index: jax.Array,
                        config: EstimatorConfig) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    mag_present = history.mag_present[index]
    sun_present = history.sun_present[index]
    mag_valid = history.mag_valid[index]
    sun_valid = history.sun_valid[index]

    # Global vector acquisition when both independent vectors are available and
    # the filter has no global attitude solution yet.
    q_triad, triad_geometry = _triad_attitude(
        history.mag_body[index], history.mag_reference_eci[index],
        history.sun_body[index], history.sun_reference_eci[index],
    )
    triad = (
        (~acquired) & mag_present & sun_present & mag_valid & sun_valid
        & config.use_magnetometer & config.use_sun_sensor & triad_geometry
    )
    triad_sigma_value = jnp.deg2rad(max(config.magnetometer_noise_std_deg, config.sun_sensor_noise_std_deg))
    triad_sigma = jnp.full((q.shape[0],), triad_sigma_value, dtype=q.dtype)
    q = jnp.where(triad[:, None], q_triad, q)
    p = jnp.where(triad[:, None, None], _set_attitude_covariance(p, triad_sigma, config), p)
    acquired = acquired | triad

    q, bias, p, acquired, star_nis, star_acc = _star_update(
        q, bias, p, acquired,
        history.star_q[index], history.star_present[index], history.star_valid[index], config,
    )
    q, bias, p, mag_nis, mag_acc = _vector_update(
        q, bias, p, history.mag_body[index], history.mag_reference_eci[index],
        history.mag_present[index], history.mag_valid[index], config.use_magnetometer,
        config.magnetometer_noise_std_deg, config.magnetometer_nis_gate, acquired, config,
    )
    q, bias, p, sun_nis, sun_acc = _vector_update(
        q, bias, p, history.sun_body[index], history.sun_reference_eci[index],
        history.sun_present[index], history.sun_valid[index], config.use_sun_sensor,
        config.sun_sensor_noise_std_deg, config.sun_sensor_nis_gate, acquired, config,
    )
    return q, bias, p, acquired, star_nis, star_acc, mag_nis, mag_acc, sun_nis, sun_acc


def _shift_history(history: ReplayHistory, current_step: jax.Array) -> ReplayHistory:
    def shift(arr: jax.Array, fill: jax.Array) -> jax.Array:
        return jnp.concatenate([arr[1:], fill[None, ...]], axis=0)

    batch = current_step.shape[0]
    zeros3 = jnp.zeros((batch, 3), dtype=history.q.dtype)
    zeros4 = jnp.zeros((batch, 4), dtype=history.q.dtype)
    false = jnp.zeros((batch,), dtype=jnp.bool_)
    return ReplayHistory(
        step=shift(history.step, current_step),
        q=shift(history.q, history.q[-1]),
        gyro_bias=shift(history.gyro_bias, history.gyro_bias[-1]),
        covariance=shift(history.covariance, history.covariance[-1]),
        acquired=shift(history.acquired, history.acquired[-1]),
        gyro_sample=shift(history.gyro_sample, zeros3),
        gyro_present=shift(history.gyro_present, false),
        gyro_valid=shift(history.gyro_valid, false),
        gyro_zoh=shift(history.gyro_zoh, history.gyro_zoh[-1]),
        star_q=shift(history.star_q, zeros4),
        star_present=shift(history.star_present, false),
        star_valid=shift(history.star_valid, false),
        mag_body=shift(history.mag_body, zeros3),
        mag_reference_eci=shift(history.mag_reference_eci, zeros3),
        mag_present=shift(history.mag_present, false),
        mag_valid=shift(history.mag_valid, false),
        sun_body=shift(history.sun_body, zeros3),
        sun_reference_eci=shift(history.sun_reference_eci, zeros3),
        sun_present=shift(history.sun_present, false),
        sun_valid=shift(history.sun_valid, false),
    )


def _insert(history: ReplayHistory, index: int, field: str, value: jax.Array, mask: jax.Array) -> ReplayHistory:
    arr = getattr(history, field)
    while mask.ndim < value.ndim:
        mask = mask[..., None]
    updated = arr.at[index].set(jnp.where(mask, value, arr[index]))
    return history._replace(**{field: updated})


def estimator_substep(
    state: EstimatorState,
    sensors: SensorState,
    absolute_physics_step: jax.Array,
    physics: PhysicsConfig,
    sensor_config: SensorConfig,
    orbit: OrbitConfig,
    config: EstimatorConfig,
) -> EstimatorState:
    """Processes timestamped telemetry and returns the current fixed-lag MEKF state."""
    history = _shift_history(state.history, absolute_physics_step)
    last_index = history.q.shape[0] - 1

    new_gyro = sensors.gyro_sample_step > state.last_gyro_step
    new_star = sensors.star_tracker_sample_step > state.last_star_tracker_step
    new_mag = sensors.magnetometer_sample_step > state.last_magnetometer_step
    new_sun = sensors.sun_sensor_sample_step > state.last_sun_sensor_step

    def idx_for(latency_seconds: float) -> int:
        if not config.use_fixed_lag_replay:
            return last_index
        return last_index - int(round(latency_seconds / physics.physics_dt))

    gyro_idx = idx_for(sensor_config.gyro_latency_seconds)
    star_idx = idx_for(sensor_config.star_tracker_latency_seconds)
    mag_idx = idx_for(sensor_config.magnetometer_latency_seconds)
    sun_idx = idx_for(sensor_config.sun_sensor_latency_seconds)

    history = _insert(history, gyro_idx, "gyro_sample", sensors.gyro, new_gyro & sensors.gyro_valid)
    gp = history.gyro_present.at[gyro_idx].set(history.gyro_present[gyro_idx] | new_gyro)
    gv = history.gyro_valid.at[gyro_idx].set(jnp.where(new_gyro, sensors.gyro_valid, history.gyro_valid[gyro_idx]))
    history = history._replace(gyro_present=gp, gyro_valid=gv)

    history = _insert(history, star_idx, "star_q", sensors.star_tracker_q, new_star & sensors.star_tracker_valid)
    sp = history.star_present.at[star_idx].set(history.star_present[star_idx] | new_star)
    sv = history.star_valid.at[star_idx].set(jnp.where(new_star, sensors.star_tracker_valid, history.star_valid[star_idx]))
    history = history._replace(star_present=sp, star_valid=sv)

    mag_sample_time = sensors.magnetometer_sample_step.astype(sensors.gyro.dtype) * physics.physics_dt
    dt_from_gnss = mag_sample_time - sensors.gnss_time_s
    gnss_position_at_mag = sensors.gnss_position_eci_m + sensors.gnss_velocity_eci_m_s * dt_from_gnss[:, None]
    mag_reference = magnetic_field_eci(gnss_position_at_mag, orbit)
    history = _insert(history, mag_idx, "mag_body", sensors.magnetometer_body_t, new_mag & sensors.magnetometer_valid)
    history = _insert(history, mag_idx, "mag_reference_eci", mag_reference, new_mag & sensors.gnss_valid)
    mp = history.mag_present.at[mag_idx].set(history.mag_present[mag_idx] | new_mag)
    mv = history.mag_valid.at[mag_idx].set(jnp.where(new_mag, sensors.magnetometer_valid & sensors.gnss_valid, history.mag_valid[mag_idx]))
    history = history._replace(mag_present=mp, mag_valid=mv)

    sun_reference = jnp.broadcast_to(sun_direction_eci(orbit, sensors.gyro.dtype), sensors.sun_direction_body.shape)
    history = _insert(history, sun_idx, "sun_body", sensors.sun_direction_body, new_sun & sensors.sun_sensor_valid)
    history = _insert(history, sun_idx, "sun_reference_eci", sun_reference, new_sun)
    sup = history.sun_present.at[sun_idx].set(history.sun_present[sun_idx] | new_sun)
    suv = history.sun_valid.at[sun_idx].set(jnp.where(new_sun, sensors.sun_sensor_valid, history.sun_valid[sun_idx]))
    history = history._replace(sun_present=sup, sun_valid=suv)

    # Always advance one step
    # roll farther back when a delayed packet arrived.
    start = jnp.asarray(last_index, dtype=jnp.int32)
    start = jnp.where(jnp.any(new_gyro), jnp.minimum(start, gyro_idx), start)
    start = jnp.where(jnp.any(new_star), jnp.minimum(start, star_idx), start)
    start = jnp.where(jnp.any(new_mag), jnp.minimum(start, mag_idx), start)
    start = jnp.where(jnp.any(new_sun), jnp.minimum(start, sun_idx), start)

    # A delayed sample at t=0 updates the already-existing reset state directly.
    start_step = history.step[start, 0]
    zero_start = start_step == 0

    star_nis0 = state.star_tracker_nis
    mag_nis0 = state.magnetometer_nis
    sun_nis0 = state.sun_sensor_nis
    star_acc0 = state.star_tracker_update_accepted
    mag_acc0 = state.magnetometer_update_accepted
    sun_acc0 = state.sun_sensor_update_accepted

    def initialize_from_zero(_):
        q0 = history.q[start]
        b0 = history.gyro_bias[start]
        p0 = history.covariance[start]
        a0 = history.acquired[start]
        q0, b0, p0, a0, sn, sa, mn, ma, un, ua = _apply_measurements(
            q0, b0, p0, a0, history, start, config
        )
        h = history._replace(
            q=history.q.at[start].set(q0),
            gyro_bias=history.gyro_bias.at[start].set(b0),
            covariance=history.covariance.at[start].set(p0),
            acquired=history.acquired.at[start].set(a0),
        )
        return h, q0, b0, p0, a0, history.gyro_zoh[start], start + 1, sn, mn, un, sa, ma, ua

    def initialize_normal(_):
        base = jnp.maximum(start - 1, 0)
        return (
            history,
            history.q[base], history.gyro_bias[base], history.covariance[base], history.acquired[base],
            history.gyro_zoh[base], start,
            star_nis0, mag_nis0, sun_nis0, star_acc0, mag_acc0, sun_acc0,
        )

    (history, q0, b0, p0, acquired0, gyro0, loop_start,
     star_nis, mag_nis, sun_nis, star_acc, mag_acc, sun_acc) = jax.lax.cond(
        zero_start, initialize_from_zero, initialize_normal, operand=None
    )

    def body(i, carry):
        h, q, bias, p, acquired, gyro, sn, mn, un, sa, ma, ua = carry
        use_new_gyro = h.gyro_present[i] & h.gyro_valid[i]
        gyro = jnp.where(use_new_gyro[:, None], h.gyro_sample[i], gyro)
        q, bias, p, _ = _propagate_arrays(q, bias, p, gyro, physics.physics_dt, config)
        q, bias, p, acquired, sn_i, sa_i, mn_i, ma_i, un_i, ua_i = _apply_measurements(
            q, bias, p, acquired, h, i, config
        )
        sn = jnp.where(h.star_present[i], sn_i, sn)
        sa = jnp.where(h.star_present[i], sa_i, sa)
        mn = jnp.where(h.mag_present[i], mn_i, mn)
        ma = jnp.where(h.mag_present[i], ma_i, ma)
        un = jnp.where(h.sun_present[i], un_i, un)
        ua = jnp.where(h.sun_present[i], ua_i, ua)
        h = h._replace(
            q=h.q.at[i].set(q),
            gyro_bias=h.gyro_bias.at[i].set(bias),
            covariance=h.covariance.at[i].set(p),
            acquired=h.acquired.at[i].set(acquired),
            gyro_zoh=h.gyro_zoh.at[i].set(gyro),
        )
        return h, q, bias, p, acquired, gyro, sn, mn, un, sa, ma, ua

    history, q, bias, p, acquired, gyro, star_nis, mag_nis, sun_nis, star_acc, mag_acc, sun_acc = jax.lax.fori_loop(
        loop_start,
        last_index + 1,
        body,
        (history, q0, b0, p0, acquired0, gyro0, star_nis, mag_nis, sun_nis, star_acc, mag_acc, sun_acc),
    )

    omega = gyro - bias
    return state._replace(
        q=q,
        gyro_bias=bias,
        omega=omega,
        covariance=p,
        attitude_acquired=acquired,
        last_gyro_step=jnp.where(new_gyro, sensors.gyro_sample_step, state.last_gyro_step),
        last_star_tracker_step=jnp.where(new_star, sensors.star_tracker_sample_step, state.last_star_tracker_step),
        last_magnetometer_step=jnp.where(new_mag, sensors.magnetometer_sample_step, state.last_magnetometer_step),
        last_sun_sensor_step=jnp.where(new_sun, sensors.sun_sensor_sample_step, state.last_sun_sensor_step),
        star_tracker_nis=star_nis,
        magnetometer_nis=mag_nis,
        sun_sensor_nis=sun_nis,
        star_tracker_update_accepted=star_acc,
        magnetometer_update_accepted=mag_acc,
        sun_sensor_update_accepted=sun_acc,
        history=history,
    )


def attitude_sigma_rad(state: EstimatorState) -> jax.Array:
    variance = jnp.trace(state.covariance[..., :3, :3], axis1=-2, axis2=-1) / 3.0
    return jnp.sqrt(jnp.maximum(variance, 0.0))


def gyro_bias_sigma_rad_s(state: EstimatorState) -> jax.Array:
    variance = jnp.trace(state.covariance[..., 3:, 3:], axis1=-2, axis2=-1) / 3.0
    return jnp.sqrt(jnp.maximum(variance, 0.0))
