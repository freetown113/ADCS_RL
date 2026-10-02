from typing import NamedTuple

import jax
import jax.numpy as jnp

from simulators.fdir.config import OrbitConfig, PhysicsConfig, SensorConfig
from simulators.fdir.math3d import (
    quat_multiply,
    quat_normalize,
    rotate_body_to_inertial,
    rotate_inertial_to_body,
    rotation_vector_to_quat,
)
from simulators.fdir.orbit import OrbitState, earth_eclipse_mask, magnetic_field_eci, sun_direction_eci
from simulators.fdir.physics import PhysicalState


class SensorState(NamedTuple):
    # Latest delivered telemetry.
    gyro: jax.Array
    wheel_speed: jax.Array
    star_tracker_q: jax.Array
    magnetometer_body_t: jax.Array
    sun_direction_body: jax.Array
    gnss_position_eci_m: jax.Array
    gnss_velocity_eci_m_s: jax.Array
    gnss_time_s: jax.Array

    gyro_valid: jax.Array
    wheel_valid: jax.Array
    star_tracker_valid: jax.Array
    magnetometer_valid: jax.Array
    sun_sensor_valid: jax.Array
    gnss_valid: jax.Array

    gyro_sample_step: jax.Array
    wheel_sample_step: jax.Array
    star_tracker_sample_step: jax.Array
    magnetometer_sample_step: jax.Array
    sun_sensor_sample_step: jax.Array
    gnss_sample_step: jax.Array

    # Sensor internal state / calibration truth (never exposed to the actor).
    rng_key: jax.Array
    gyro_bias_true: jax.Array
    gyro_scale: jax.Array
    gyro_misalignment_q: jax.Array
    wheel_tach_scale: jax.Array
    star_alignment_q: jax.Array
    magnetometer_bias_true_t: jax.Array
    magnetometer_scale: jax.Array
    magnetometer_misalignment_q: jax.Array
    sun_misalignment_q: jax.Array
    star_tracker_locked: jax.Array
    star_acquisition_elapsed_s: jax.Array

    # Simulator-private source histories used only to realize fixed latency.
    q_history: jax.Array
    omega_history: jax.Array
    wheel_speed_history: jax.Array
    position_history: jax.Array
    velocity_history: jax.Array
    gyro_bias_history: jax.Array
    star_locked_history: jax.Array


class SensorAges(NamedTuple):
    gyro: jax.Array
    wheel: jax.Array
    star_tracker: jax.Array
    magnetometer: jax.Array
    sun_sensor: jax.Array
    gnss: jax.Array


def _period_steps(rate_hz: float, physics_dt: float) -> int:
    return int(round(1.0 / (rate_hz * physics_dt)))


def _latency_steps(latency_seconds: float, physics_dt: float) -> int:
    return int(round(latency_seconds / physics_dt))


def history_length(config: SensorConfig, physics: PhysicsConfig) -> int:
    latencies = (
        config.gyro_latency_seconds,
        config.wheel_tach_latency_seconds,
        config.star_tracker_latency_seconds,
        config.magnetometer_latency_seconds,
        config.sun_sensor_latency_seconds,
        config.gnss_latency_seconds,
    )
    return max(_latency_steps(value, physics.physics_dt) for value in latencies) + 1


def _normalize(vector: jax.Array, eps: float = 1.0e-12) -> jax.Array:
    return vector / (jnp.linalg.norm(vector, axis=-1, keepdims=True) + eps)


def _batched_split(keys: jax.Array, count: int) -> tuple[jax.Array, list[jax.Array]]:
    split = jax.vmap(lambda k: jax.random.split(k, count + 1))(keys)
    return split[:, 0], [split[:, i + 1] for i in range(count)]


def _normal(keys: jax.Array, tail_shape: tuple[int, ...]) -> jax.Array:
    return jax.vmap(lambda k: jax.random.normal(k, tail_shape))(keys)


def _uniform(keys: jax.Array) -> jax.Array:
    return jax.vmap(jax.random.uniform)(keys)


def _quantize(value: jax.Array, quantum: float) -> jax.Array:
    if quantum <= 0.0:
        return value
    q = jnp.asarray(quantum, dtype=value.dtype)
    return q * jnp.round(value / q)


def _random_alignment(keys: jax.Array, std_deg: float, dtype) -> jax.Array:
    std = jnp.deg2rad(jnp.asarray(std_deg, dtype=dtype))
    return rotation_vector_to_quat(_normal(keys, (3,)).astype(dtype) * std)


def _body_magnetic_field(q: jax.Array, position_eci_m: jax.Array, orbit_config: OrbitConfig) -> jax.Array:
    return rotate_inertial_to_body(q, magnetic_field_eci(position_eci_m, orbit_config))


def _body_sun_direction(q: jax.Array, orbit_config: OrbitConfig) -> jax.Array:
    sun_eci = sun_direction_eci(orbit_config, q.dtype)
    sun_eci = jnp.broadcast_to(sun_eci, q[..., 1:].shape)
    return _normalize(rotate_inertial_to_body(q, sun_eci))


def _star_visibility(q: jax.Array, omega: jax.Array, position: jax.Array, orbit: OrbitState,
                     config: SensorConfig, orbit_config: OrbitConfig,
                     locked: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns acquisition_ok and tracking_ok for a single star-tracker head."""
    dtype = q.dtype
    boresight_body = _normalize(jnp.asarray(config.star_tracker_boresight_body, dtype=dtype))
    boresight_body = jnp.broadcast_to(boresight_body, omega.shape)
    boresight_eci = _normalize(rotate_body_to_inertial(q, boresight_body))

    sun = _normalize(orbit.sun_direction_eci)
    sun_cos = jnp.sum(boresight_eci * sun, axis=-1)
    sun_sep = jnp.arccos(jnp.clip(sun_cos, -1.0, 1.0))
    sun_ok = sun_sep >= jnp.deg2rad(config.star_tracker_sun_exclusion_deg)

    radius = jnp.linalg.norm(position, axis=-1)
    nadir = _normalize(-position)
    earth_sep = jnp.arccos(jnp.clip(jnp.sum(boresight_eci * nadir, axis=-1), -1.0, 1.0))
    earth_angular_radius = jnp.arcsin(
        jnp.clip(jnp.asarray(orbit_config.earth_radius_m, dtype=dtype) / radius, 0.0, 1.0)
    )
    earth_ok = earth_sep >= (
        earth_angular_radius + jnp.deg2rad(config.star_tracker_earth_limb_exclusion_deg)
    )

    clear = sun_ok & earth_ok
    rate_deg_s = jnp.rad2deg(jnp.linalg.norm(omega, axis=-1))
    acquisition_ok = clear & (rate_deg_s <= config.star_tracker_max_acquisition_rate_deg_s)
    tracking_ok = clear & (rate_deg_s <= config.star_tracker_max_tracking_rate_deg_s)
    return acquisition_ok, tracking_ok


def _advance_star_lock(state: SensorState, physical: PhysicalState, orbit: OrbitState,
                       config: SensorConfig, physics: PhysicsConfig,
                       orbit_config: OrbitConfig) -> tuple[jax.Array, jax.Array]:
    acquisition_ok, tracking_ok = _star_visibility(
        physical.q, physical.omega, orbit.position_eci_m, orbit, config, orbit_config,
        state.star_tracker_locked,
    )
    still_locked = state.star_tracker_locked & tracking_ok
    elapsed = jnp.where(
        still_locked,
        state.star_acquisition_elapsed_s,
        jnp.where(
            acquisition_ok,
            state.star_acquisition_elapsed_s + physics.physics_dt,
            jnp.zeros_like(state.star_acquisition_elapsed_s),
        ),
    )
    acquired = (~state.star_tracker_locked) & acquisition_ok & (
        elapsed >= config.star_tracker_acquisition_time_s
    )
    locked = still_locked | acquired
    elapsed = jnp.where(locked, elapsed, jnp.where(acquisition_ok, elapsed, 0.0))
    return locked, elapsed


def _gyro_measurement(omega: jax.Array, bias: jax.Array, scale: jax.Array, misalignment_q: jax.Array,
                      noise_keys: jax.Array, config: SensorConfig) -> jax.Array:
    measured = rotate_body_to_inertial(misalignment_q, omega)
    measured = measured * scale + bias
    sample_period = 1.0 / config.gyro_rate_hz
    noise_std = config.gyro_noise_density_rad_s_sqrt_hz / jnp.sqrt(sample_period)
    measured = measured + noise_std * _normal(noise_keys, (3,)).astype(omega.dtype)
    measured = _quantize(measured, config.gyro_quantization_rad_s)
    return jnp.clip(measured, -config.gyro_clip_rad_s, config.gyro_clip_rad_s)


def _wheel_measurement(speed: jax.Array, scale: jax.Array, noise_keys: jax.Array,
                       config: SensorConfig) -> jax.Array:
    measured = speed * scale
    measured = measured + config.wheel_tach_noise_std_rad_s * _normal(
        noise_keys, (4,)
    ).astype(speed.dtype)
    return _quantize(measured, config.wheel_tach_quantization_rad_s)


def _star_measurement(q: jax.Array, alignment_q: jax.Array, noise_keys: jax.Array, outlier_keys: jax.Array,
                      config: SensorConfig) -> tuple[jax.Array, jax.Array]:
    dtype = q.dtype
    noise_std = jnp.deg2rad(jnp.asarray(config.star_tracker_noise_std_deg, dtype=dtype))
    small_noise = rotation_vector_to_quat(_normal(noise_keys, (3,)).astype(dtype) * noise_std)
    outlier_std = jnp.deg2rad(jnp.asarray(config.star_tracker_outlier_std_deg, dtype=dtype))
    outlier_noise = rotation_vector_to_quat(_normal(outlier_keys, (3,)).astype(dtype) * outlier_std)
    outlier = _uniform(outlier_keys) < config.star_tracker_outlier_probability
    random_q = jnp.where(outlier[:, None], outlier_noise, small_noise)
    measured = quat_normalize(quat_multiply(quat_multiply(q, alignment_q), random_q))
    return measured, outlier


def _mag_measurement(field: jax.Array, bias: jax.Array, scale: jax.Array, misalignment_q: jax.Array,
                     noise_keys: jax.Array, config: SensorConfig) -> jax.Array:
    measured = rotate_body_to_inertial(misalignment_q, field) * scale + bias
    measured = measured + config.magnetometer_noise_std_t * _normal(
        noise_keys, (3,)
    ).astype(field.dtype)
    measured = _quantize(measured, config.magnetometer_quantization_t)
    return jnp.clip(measured, -config.magnetometer_clip_t, config.magnetometer_clip_t)


def _sun_measurement(direction: jax.Array, misalignment_q: jax.Array, noise_keys: jax.Array,
                     config: SensorConfig) -> jax.Array:
    dtype = direction.dtype
    measured = rotate_body_to_inertial(misalignment_q, direction)
    noise_std = jnp.deg2rad(jnp.asarray(config.sun_sensor_noise_std_deg, dtype=dtype))
    noise_q = rotation_vector_to_quat(_normal(noise_keys, (3,)).astype(dtype) * noise_std)
    return _normalize(rotate_body_to_inertial(noise_q, measured))


def _gnss_measurement(position: jax.Array, velocity: jax.Array, pos_keys: jax.Array, vel_keys: jax.Array,
                      config: SensorConfig) -> tuple[jax.Array, jax.Array]:
    position_m = position + config.gnss_position_noise_std_m * _normal(pos_keys, (3,)).astype(position.dtype)
    velocity_m = velocity + config.gnss_velocity_noise_std_m_s * _normal(vel_keys, (3,)).astype(velocity.dtype)
    return position_m, velocity_m


def reset_sensor_state(
    physical: PhysicalState,
    orbit: OrbitState,
    key: jax.Array,
    config: SensorConfig,
    physics: PhysicsConfig,
    orbit_config: OrbitConfig,
) -> SensorState:
    """Initializes calibration, sensor internal state, and t=0 telemetry."""
    batch_size = physical.omega.shape[0]
    dtype = physical.omega.dtype
    history_size = history_length(config, physics)
    env_keys = jax.random.split(key, batch_size)
    env_keys, keys = _batched_split(env_keys, 14)

    gyro_bias = config.gyro_initial_bias_std_rad_s * _normal(keys[0], (3,)).astype(dtype)
    gyro_scale = 1.0 + config.gyro_scale_factor_std * _normal(keys[1], (3,)).astype(dtype)
    gyro_mis_q = _random_alignment(keys[2], config.gyro_misalignment_std_deg, dtype)
    wheel_scale = 1.0 + config.wheel_tach_scale_factor_std * _normal(keys[3], (4,)).astype(dtype)
    star_align_q = _random_alignment(keys[4], config.star_tracker_alignment_std_deg, dtype)
    mag_bias = config.magnetometer_bias_std_t * _normal(keys[5], (3,)).astype(dtype)
    mag_scale = 1.0 + config.magnetometer_scale_factor_std * _normal(keys[6], (3,)).astype(dtype)
    mag_mis_q = _random_alignment(keys[7], config.magnetometer_misalignment_std_deg, dtype)
    sun_mis_q = _random_alignment(keys[8], config.sun_sensor_misalignment_std_deg, dtype)

    initially_locked = jnp.full((batch_size,), config.star_tracker_initially_locked, jnp.bool_)
    acquisition_ok, tracking_ok = _star_visibility(
        physical.q, physical.omega, orbit.position_eci_m, orbit, config, orbit_config,
        initially_locked,
    )
    initially_locked = initially_locked & tracking_ok
    acquisition_elapsed = jnp.zeros((batch_size,), dtype=dtype)

    def history(value: jax.Array) -> jax.Array:
        return jnp.broadcast_to(value[None, ...], (history_size,) + value.shape)

    q_history = history(physical.q)
    omega_history = history(physical.omega)
    wheel_history = history(physical.wheel_speed)
    position_history = history(orbit.position_eci_m)
    velocity_history = history(orbit.velocity_eci_m_s)
    bias_history = history(gyro_bias)
    lock_history = history(initially_locked)

    def immediate(latency: float) -> bool:
        return _latency_steps(latency, physics.physics_dt) == 0

    gyro_immediate = immediate(config.gyro_latency_seconds)
    wheel_immediate = immediate(config.wheel_tach_latency_seconds)
    star_immediate = immediate(config.star_tracker_latency_seconds)
    mag_immediate = immediate(config.magnetometer_latency_seconds)
    sun_immediate = immediate(config.sun_sensor_latency_seconds)
    gnss_immediate = immediate(config.gnss_latency_seconds)

    magnetic = _body_magnetic_field(physical.q, orbit.position_eci_m, orbit_config)
    sun_body = _body_sun_direction(physical.q, orbit_config)
    eclipsed = earth_eclipse_mask(orbit.position_eci_m, orbit.sun_direction_eci, orbit_config)
    sun_available = (~eclipsed) | (not config.sun_sensor_eclipse_enabled)

    # Fresh random keys for t=0 measurements.
    env_keys, mk = _batched_split(env_keys, 12)
    gyro_m = _gyro_measurement(physical.omega, gyro_bias, gyro_scale, gyro_mis_q, mk[0], config)
    wheel_m = _wheel_measurement(physical.wheel_speed, wheel_scale, mk[1], config)
    star_m, _ = _star_measurement(physical.q, star_align_q, mk[2], mk[3], config)
    mag_m = _mag_measurement(magnetic, mag_bias, mag_scale, mag_mis_q, mk[4], config)
    sun_m = _sun_measurement(sun_body, sun_mis_q, mk[5], config)
    gnss_p, gnss_v = _gnss_measurement(orbit.position_eci_m, orbit.velocity_eci_m_s, mk[6], mk[7], config)

    gyro_ok = gyro_immediate & (_uniform(mk[8]) >= config.gyro_packet_loss_probability)
    wheel_ok = wheel_immediate & (_uniform(mk[9]) >= config.wheel_tach_packet_loss_probability)
    star_packet = star_immediate & (_uniform(mk[10]) >= config.star_tracker_packet_loss_probability)
    star_valid = star_packet & initially_locked
    mag_ok = mag_immediate
    sun_packet = sun_immediate
    sun_valid = sun_packet & sun_available
    gnss_ok = gnss_immediate & (_uniform(mk[11]) >= config.gnss_packet_loss_probability)

    def scalar_bool(value) -> jax.Array:
        if isinstance(value, bool):
            return jnp.full((batch_size,), value, dtype=jnp.bool_)
        return value.astype(jnp.bool_)

    def scalar_step(valid) -> jax.Array:
        valid = scalar_bool(valid)
        return jnp.where(valid, jnp.zeros((batch_size,), jnp.int32), -jnp.ones((batch_size,), jnp.int32))

    return SensorState(
        gyro=jnp.where(scalar_bool(gyro_ok)[:, None], gyro_m, jnp.zeros_like(gyro_m)),
        wheel_speed=jnp.where(scalar_bool(wheel_ok)[:, None], wheel_m, jnp.zeros_like(wheel_m)),
        star_tracker_q=jnp.where(star_valid[:, None], star_m, jnp.zeros_like(star_m)),
        magnetometer_body_t=jnp.where(scalar_bool(mag_ok)[:, None], mag_m, jnp.zeros_like(mag_m)),
        sun_direction_body=jnp.where(sun_valid[:, None], sun_m, jnp.zeros_like(sun_m)),
        gnss_position_eci_m=jnp.where(gnss_ok[:, None], gnss_p, jnp.zeros_like(gnss_p)),
        gnss_velocity_eci_m_s=jnp.where(gnss_ok[:, None], gnss_v, jnp.zeros_like(gnss_v)),
        gnss_time_s=jnp.where(gnss_ok, orbit.time_s, jnp.zeros_like(orbit.time_s)),
        gyro_valid=scalar_bool(gyro_ok),
        wheel_valid=jnp.broadcast_to(scalar_bool(wheel_ok)[:, None], physical.wheel_speed.shape),
        star_tracker_valid=star_valid,
        magnetometer_valid=scalar_bool(mag_ok),
        sun_sensor_valid=sun_valid,
        gnss_valid=gnss_ok,
        gyro_sample_step=scalar_step(gyro_ok),
        wheel_sample_step=jnp.where(
            scalar_bool(wheel_ok)[:, None], jnp.zeros_like(physical.wheel_speed, dtype=jnp.int32),
            -jnp.ones_like(physical.wheel_speed, dtype=jnp.int32),
        ),
        star_tracker_sample_step=scalar_step(star_packet),
        magnetometer_sample_step=scalar_step(mag_ok),
        sun_sensor_sample_step=scalar_step(sun_packet),
        gnss_sample_step=scalar_step(gnss_ok),
        rng_key=env_keys,
        gyro_bias_true=gyro_bias,
        gyro_scale=gyro_scale,
        gyro_misalignment_q=gyro_mis_q,
        wheel_tach_scale=wheel_scale,
        star_alignment_q=star_align_q,
        magnetometer_bias_true_t=mag_bias,
        magnetometer_scale=mag_scale,
        magnetometer_misalignment_q=mag_mis_q,
        sun_misalignment_q=sun_mis_q,
        star_tracker_locked=initially_locked,
        star_acquisition_elapsed_s=acquisition_elapsed,
        q_history=q_history,
        omega_history=omega_history,
        wheel_speed_history=wheel_history,
        position_history=position_history,
        velocity_history=velocity_history,
        gyro_bias_history=bias_history,
        star_locked_history=lock_history,
    )


def _due(absolute_physics_step: jax.Array, rate_hz: float, latency_seconds: float,
         physics_dt: float) -> tuple[jax.Array, jax.Array, int]:
    period = _period_steps(rate_hz, physics_dt)
    latency = _latency_steps(latency_seconds, physics_dt)
    source_step = absolute_physics_step - latency
    due = (source_step >= 0) & (jnp.mod(source_step, period) == 0)
    return due, source_step, latency


def sensor_substep(
    state: SensorState,
    physical: PhysicalState,
    orbit: OrbitState,
    absolute_physics_step: jax.Array,
    config: SensorConfig,
    physics: PhysicsConfig,
    orbit_config: OrbitConfig,
) -> SensorState:
    """Advances sensor internal dynamics and delivers packets due at this step."""
    rng, keys = _batched_split(state.rng_key, 18)

    # Continuous gyro bias random walk is generated in sensor truth, independently
    # from the MEKF's bias estimate/process model.
    bias_rw = config.gyro_bias_random_walk_rad_s_per_sqrt_s * jnp.sqrt(physics.physics_dt)
    gyro_bias = state.gyro_bias_true + bias_rw * _normal(keys[0], (3,)).astype(physical.omega.dtype)

    star_locked, star_elapsed = _advance_star_lock(
        state, physical, orbit, config, physics, orbit_config
    )

    q_history = jnp.concatenate([state.q_history[1:], physical.q[None, ...]], axis=0)
    omega_history = jnp.concatenate([state.omega_history[1:], physical.omega[None, ...]], axis=0)
    wheel_history = jnp.concatenate([state.wheel_speed_history[1:], physical.wheel_speed[None, ...]], axis=0)
    position_history = jnp.concatenate([state.position_history[1:], orbit.position_eci_m[None, ...]], axis=0)
    velocity_history = jnp.concatenate([state.velocity_history[1:], orbit.velocity_eci_m_s[None, ...]], axis=0)
    bias_history = jnp.concatenate([state.gyro_bias_history[1:], gyro_bias[None, ...]], axis=0)
    lock_history = jnp.concatenate([state.star_locked_history[1:], star_locked[None, ...]], axis=0)

    gyro_due, gyro_source, gyro_latency = _due(absolute_physics_step, config.gyro_rate_hz, config.gyro_latency_seconds, physics.physics_dt)
    wheel_due, wheel_source, wheel_latency = _due(absolute_physics_step, config.wheel_tach_rate_hz, config.wheel_tach_latency_seconds, physics.physics_dt)
    star_due, star_source, star_latency = _due(absolute_physics_step, config.star_tracker_rate_hz, config.star_tracker_latency_seconds, physics.physics_dt)
    mag_due, mag_source, mag_latency = _due(absolute_physics_step, config.magnetometer_rate_hz, config.magnetometer_latency_seconds, physics.physics_dt)
    sun_due, sun_source, sun_latency = _due(absolute_physics_step, config.sun_sensor_rate_hz, config.sun_sensor_latency_seconds, physics.physics_dt)
    gnss_due, gnss_source, gnss_latency = _due(absolute_physics_step, config.gnss_rate_hz, config.gnss_latency_seconds, physics.physics_dt)

    gyro_candidate = _gyro_measurement(
        omega_history[-1 - gyro_latency], bias_history[-1 - gyro_latency],
        state.gyro_scale, state.gyro_misalignment_q, keys[1], config,
    )
    wheel_candidate = _wheel_measurement(
        wheel_history[-1 - wheel_latency], state.wheel_tach_scale, keys[2], config
    )
    star_candidate, _ = _star_measurement(
        quat_normalize(q_history[-1 - star_latency]), state.star_alignment_q,
        keys[3], keys[4], config,
    )
    source_star_locked = lock_history[-1 - star_latency]

    mag_q = q_history[-1 - mag_latency]
    mag_position = position_history[-1 - mag_latency]
    mag_truth = _body_magnetic_field(mag_q, mag_position, orbit_config)
    mag_candidate = _mag_measurement(
        mag_truth, state.magnetometer_bias_true_t, state.magnetometer_scale,
        state.magnetometer_misalignment_q, keys[5], config,
    )

    sun_q = q_history[-1 - sun_latency]
    sun_position = position_history[-1 - sun_latency]
    sun_truth = _body_sun_direction(sun_q, orbit_config)
    sun_candidate = _sun_measurement(sun_truth, state.sun_misalignment_q, keys[6], config)
    sun_eci = jnp.broadcast_to(sun_direction_eci(orbit_config, sun_position.dtype), sun_position.shape)
    eclipsed = earth_eclipse_mask(sun_position, sun_eci, orbit_config)
    sun_available = (~eclipsed) | (not config.sun_sensor_eclipse_enabled)

    gnss_p, gnss_v = _gnss_measurement(
        position_history[-1 - gnss_latency], velocity_history[-1 - gnss_latency],
        keys[7], keys[8], config,
    )
    gnss_time = gnss_source.astype(physical.omega.dtype) * physics.physics_dt

    gyro_deliver = gyro_due & (_uniform(keys[9]) >= config.gyro_packet_loss_probability)
    wheel_deliver = wheel_due & (_uniform(keys[10]) >= config.wheel_tach_packet_loss_probability)
    star_packet = star_due & (_uniform(keys[11]) >= config.star_tracker_packet_loss_probability)
    star_valid = star_packet & source_star_locked
    mag_deliver = mag_due & (_uniform(keys[12]) >= config.magnetometer_packet_loss_probability)
    sun_packet = sun_due & (_uniform(keys[13]) >= config.sun_sensor_packet_loss_probability)
    sun_valid = sun_packet & sun_available
    gnss_deliver = gnss_due & (_uniform(keys[14]) >= config.gnss_packet_loss_probability)

    gyro = jnp.where(gyro_deliver[:, None], gyro_candidate, state.gyro)
    wheel_speed = jnp.where(wheel_deliver[:, None], wheel_candidate, state.wheel_speed)
    star_q = jnp.where(star_valid[:, None], star_candidate, state.star_tracker_q)
    magnetic = jnp.where(mag_deliver[:, None], mag_candidate, state.magnetometer_body_t)
    sun_body = jnp.where(sun_valid[:, None], sun_candidate, state.sun_direction_body)
    gnss_position = jnp.where(gnss_deliver[:, None], gnss_p, state.gnss_position_eci_m)
    gnss_velocity = jnp.where(gnss_deliver[:, None], gnss_v, state.gnss_velocity_eci_m_s)
    gnss_time_out = jnp.where(gnss_deliver, gnss_time, state.gnss_time_s)

    return state._replace(
        gyro=gyro,
        wheel_speed=wheel_speed,
        star_tracker_q=star_q,
        magnetometer_body_t=magnetic,
        sun_direction_body=sun_body,
        gnss_position_eci_m=gnss_position,
        gnss_velocity_eci_m_s=gnss_velocity,
        gnss_time_s=gnss_time_out,
        gyro_valid=state.gyro_valid | gyro_deliver,
        wheel_valid=state.wheel_valid | jnp.broadcast_to(wheel_deliver[:, None], state.wheel_valid.shape),
        # A delivered star/Sun solution explicitly carries validity. Packet loss
        # leaves the previous telemetry untouched and only increases its age.
        star_tracker_valid=jnp.where(star_packet, source_star_locked, state.star_tracker_valid),
        magnetometer_valid=state.magnetometer_valid | mag_deliver,
        sun_sensor_valid=jnp.where(sun_packet, sun_available, state.sun_sensor_valid),
        gnss_valid=state.gnss_valid | gnss_deliver,
        gyro_sample_step=jnp.where(gyro_deliver, gyro_source, state.gyro_sample_step),
        wheel_sample_step=jnp.where(
            wheel_deliver[:, None], wheel_source[:, None], state.wheel_sample_step
        ),
        star_tracker_sample_step=jnp.where(star_packet, star_source, state.star_tracker_sample_step),
        magnetometer_sample_step=jnp.where(mag_deliver, mag_source, state.magnetometer_sample_step),
        sun_sensor_sample_step=jnp.where(sun_packet, sun_source, state.sun_sensor_sample_step),
        gnss_sample_step=jnp.where(gnss_deliver, gnss_source, state.gnss_sample_step),
        rng_key=rng,
        gyro_bias_true=gyro_bias,
        star_tracker_locked=star_locked,
        star_acquisition_elapsed_s=star_elapsed,
        q_history=q_history,
        omega_history=omega_history,
        wheel_speed_history=wheel_history,
        position_history=position_history,
        velocity_history=velocity_history,
        gyro_bias_history=bias_history,
        star_locked_history=lock_history,
    )


def _age(valid: jax.Array, sample_step: jax.Array, absolute_physics_step: jax.Array,
         physics_dt: float) -> jax.Array:
    invalid_age = jnp.asarray(jnp.inf, dtype=jnp.float32)
    while absolute_physics_step.ndim < sample_step.ndim:
        absolute_physics_step = absolute_physics_step[..., None]
    return jnp.where(
        valid,
        (absolute_physics_step - sample_step) * physics_dt,
        invalid_age,
    )


def sensor_age_seconds(state: SensorState, absolute_physics_step: jax.Array,
                       physics: PhysicsConfig) -> SensorAges:
    return SensorAges(
        gyro=_age(state.gyro_valid, state.gyro_sample_step, absolute_physics_step, physics.physics_dt),
        wheel=_age(state.wheel_valid, state.wheel_sample_step, absolute_physics_step, physics.physics_dt),
        star_tracker=_age(state.star_tracker_valid, state.star_tracker_sample_step, absolute_physics_step, physics.physics_dt),
        magnetometer=_age(state.magnetometer_valid, state.magnetometer_sample_step, absolute_physics_step, physics.physics_dt),
        sun_sensor=_age(state.sun_sensor_valid, state.sun_sensor_sample_step, absolute_physics_step, physics.physics_dt),
        gnss=_age(state.gnss_valid, state.gnss_sample_step, absolute_physics_step, physics.physics_dt),
    )
