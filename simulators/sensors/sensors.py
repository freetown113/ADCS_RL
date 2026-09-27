from typing import NamedTuple

import jax
import jax.numpy as jnp

from simulators.sensors.config import OrbitConfig, PhysicsConfig, SensorConfig
from simulators.sensors.math3d import quat_normalize, rotate_inertial_to_body
from simulators.sensors.orbit import (
    OrbitState,
    earth_eclipse_mask,
    magnetic_field_eci,
    sun_direction_eci,
)
from .physics import PhysicalState


class SensorState(NamedTuple):
    """State contains latest received measurements and histories buffer"""

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

    q_history: jax.Array
    omega_history: jax.Array
    wheel_speed_history: jax.Array
    position_history: jax.Array
    velocity_history: jax.Array


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


def _body_magnetic_field(
    q_body_to_inertial: jax.Array,
    position_eci_m: jax.Array,
    orbit_config: OrbitConfig,
) -> jax.Array:
    field_eci = magnetic_field_eci(position_eci_m, orbit_config)
    return rotate_inertial_to_body(q_body_to_inertial, field_eci)


def _body_sun_direction(
    q_body_to_inertial: jax.Array,
    orbit_config: OrbitConfig,
) -> jax.Array:
    sun_eci = sun_direction_eci(orbit_config, q_body_to_inertial.dtype)
    sun_eci = jnp.broadcast_to(sun_eci, q_body_to_inertial[..., 1:].shape)
    body = rotate_inertial_to_body(q_body_to_inertial, sun_eci)
    return body / (jnp.linalg.norm(body, axis=-1, keepdims=True) + 1.0e-12)


def reset_sensor_state(
    physical: PhysicalState,
    orbit: OrbitState,
    config: SensorConfig,
    physics: PhysicsConfig,
    orbit_config: OrbitConfig,
) -> SensorState:
    """Initializes all sensor clocks at t=0"""
    batch_size = physical.omega.shape[0]
    history_size = history_length(config, physics)

    def history(value: jax.Array) -> jax.Array:
        return jnp.broadcast_to(value[None, ...], (history_size,) + value.shape)

    q_history = history(physical.q)
    omega_history = history(physical.omega)
    wheel_history = history(physical.wheel_speed)
    position_history = history(orbit.position_eci_m)
    velocity_history = history(orbit.velocity_eci_m_s)

    gyro_immediate = _latency_steps(
        config.gyro_latency_seconds, physics.physics_dt
    ) == 0
    wheel_immediate = _latency_steps(
        config.wheel_tach_latency_seconds, physics.physics_dt
    ) == 0
    star_immediate = _latency_steps(
        config.star_tracker_latency_seconds, physics.physics_dt
    ) == 0
    mag_immediate = _latency_steps(
        config.magnetometer_latency_seconds, physics.physics_dt
    ) == 0
    sun_immediate = _latency_steps(
        config.sun_sensor_latency_seconds, physics.physics_dt
    ) == 0
    gnss_immediate = _latency_steps(
        config.gnss_latency_seconds, physics.physics_dt
    ) == 0

    magnetic = _body_magnetic_field(physical.q, orbit.position_eci_m, orbit_config)
    sun_body = _body_sun_direction(physical.q, orbit_config)
    eclipse = earth_eclipse_mask(
        orbit.position_eci_m, orbit.sun_direction_eci, orbit_config
    )
    sun_available = (~eclipse)
    sun_valid_initial = sun_immediate & sun_available

    def scalar_valid(value: bool) -> jax.Array:
        return jnp.full((batch_size,), value, dtype=jnp.bool_)

    def scalar_step(value: bool) -> jax.Array:
        return jnp.full((batch_size,), 0 if value else -1, dtype=jnp.int32)

    return SensorState(
        gyro=jnp.where(gyro_immediate, physical.omega, jnp.zeros_like(physical.omega)),
        wheel_speed=jnp.where(
            wheel_immediate, physical.wheel_speed, jnp.zeros_like(physical.wheel_speed)
        ),
        star_tracker_q=jnp.where(
            star_immediate,
            quat_normalize(physical.q),
            jnp.zeros_like(physical.q),
        ),
        magnetometer_body_t=jnp.where(
            mag_immediate, magnetic, jnp.zeros_like(magnetic)
        ),
        sun_direction_body=jnp.where(
            sun_valid_initial[:, None], sun_body, jnp.zeros_like(sun_body)
        ),
        gnss_position_eci_m=jnp.where(
            gnss_immediate, orbit.position_eci_m, jnp.zeros_like(orbit.position_eci_m)
        ),
        gnss_velocity_eci_m_s=jnp.where(
            gnss_immediate, orbit.velocity_eci_m_s, jnp.zeros_like(orbit.velocity_eci_m_s)
        ),
        gnss_time_s=jnp.where(
            gnss_immediate, orbit.time_s, jnp.zeros_like(orbit.time_s)
        ),
        gyro_valid=scalar_valid(gyro_immediate),
        wheel_valid=jnp.full(physical.wheel_speed.shape, wheel_immediate, jnp.bool_),
        star_tracker_valid=scalar_valid(star_immediate),
        magnetometer_valid=scalar_valid(mag_immediate),
        sun_sensor_valid=sun_valid_initial,
        gnss_valid=scalar_valid(gnss_immediate),
        gyro_sample_step=scalar_step(gyro_immediate),
        wheel_sample_step=jnp.full(
            physical.wheel_speed.shape,
            0 if wheel_immediate else -1,
            dtype=jnp.int32,
        ),
        star_tracker_sample_step=scalar_step(star_immediate),
        magnetometer_sample_step=scalar_step(mag_immediate),
        sun_sensor_sample_step=scalar_step(sun_immediate),
        gnss_sample_step=scalar_step(gnss_immediate),
        q_history=q_history,
        omega_history=omega_history,
        wheel_speed_history=wheel_history,
        position_history=position_history,
        velocity_history=velocity_history,
    )


def _due(
    absolute_physics_step: jax.Array,
    rate_hz: float,
    latency_seconds: float,
    physics_dt: float,
) -> tuple[jax.Array, jax.Array, int]:
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
    """Advances histories and delivers every sensor packet due at this step"""
    q_history = jnp.concatenate(
        [state.q_history[1:], physical.q[None, ...]], axis=0
    )
    omega_history = jnp.concatenate(
        [state.omega_history[1:], physical.omega[None, ...]], axis=0
    )
    wheel_history = jnp.concatenate(
        [state.wheel_speed_history[1:], physical.wheel_speed[None, ...]], axis=0
    )
    position_history = jnp.concatenate(
        [state.position_history[1:], orbit.position_eci_m[None, ...]], axis=0
    )
    velocity_history = jnp.concatenate(
        [state.velocity_history[1:], orbit.velocity_eci_m_s[None, ...]], axis=0
    )

    gyro_due, gyro_source, gyro_latency = _due(
        absolute_physics_step,
        config.gyro_rate_hz,
        config.gyro_latency_seconds,
        physics.physics_dt,
    )
    wheel_due_scalar, wheel_source, wheel_latency = _due(
        absolute_physics_step,
        config.wheel_tach_rate_hz,
        config.wheel_tach_latency_seconds,
        physics.physics_dt,
    )
    star_due, star_source, star_latency = _due(
        absolute_physics_step,
        config.star_tracker_rate_hz,
        config.star_tracker_latency_seconds,
        physics.physics_dt,
    )
    mag_due, mag_source, mag_latency = _due(
        absolute_physics_step,
        config.magnetometer_rate_hz,
        config.magnetometer_latency_seconds,
        physics.physics_dt,
    )
    sun_due, sun_source, sun_latency = _due(
        absolute_physics_step,
        config.sun_sensor_rate_hz,
        config.sun_sensor_latency_seconds,
        physics.physics_dt,
    )
    gnss_due, gnss_source, gnss_latency = _due(
        absolute_physics_step,
        config.gnss_rate_hz,
        config.gnss_latency_seconds,
        physics.physics_dt,
    )

    gyro_candidate = omega_history[-1 - gyro_latency]
    wheel_candidate = wheel_history[-1 - wheel_latency]
    star_candidate = quat_normalize(q_history[-1 - star_latency])

    mag_q = q_history[-1 - mag_latency]
    mag_position = position_history[-1 - mag_latency]
    mag_candidate = _body_magnetic_field(mag_q, mag_position, orbit_config)

    sun_q = q_history[-1 - sun_latency]
    sun_position = position_history[-1 - sun_latency]
    sun_candidate = _body_sun_direction(sun_q, orbit_config)
    sun_eci = sun_direction_eci(orbit_config, sun_position.dtype)
    sun_eci = jnp.broadcast_to(sun_eci, sun_position.shape)
    eclipsed = earth_eclipse_mask(sun_position, sun_eci, orbit_config)
    sun_available = (~eclipsed) | (~config.sun_sensor_eclipse_enabled)
    sun_packet_valid = sun_due & sun_available

    gnss_position_candidate = position_history[-1 - gnss_latency]
    gnss_velocity_candidate = velocity_history[-1 - gnss_latency]
    gnss_time_candidate = gnss_source.astype(physical.omega.dtype) * physics.physics_dt

    wheel_due = wheel_due_scalar[:, None]
    gyro = jnp.where(gyro_due[:, None], gyro_candidate, state.gyro)
    wheel_speed = jnp.where(wheel_due, wheel_candidate, state.wheel_speed)
    star_q = jnp.where(star_due[:, None], star_candidate, state.star_tracker_q)
    magnetic = jnp.where(
        mag_due[:, None], mag_candidate, state.magnetometer_body_t
    )
    sun_body = jnp.where(
        sun_packet_valid[:, None], sun_candidate, state.sun_direction_body
    )
    gnss_position = jnp.where(
        gnss_due[:, None], gnss_position_candidate, state.gnss_position_eci_m
    )
    gnss_velocity = jnp.where(
        gnss_due[:, None], gnss_velocity_candidate, state.gnss_velocity_eci_m_s
    )
    gnss_time = jnp.where(gnss_due, gnss_time_candidate, state.gnss_time_s)

    return SensorState(
        gyro=gyro,
        wheel_speed=wheel_speed,
        star_tracker_q=star_q,
        magnetometer_body_t=magnetic,
        sun_direction_body=sun_body,
        gnss_position_eci_m=gnss_position,
        gnss_velocity_eci_m_s=gnss_velocity,
        gnss_time_s=gnss_time,
        gyro_valid=state.gyro_valid | gyro_due,
        wheel_valid=state.wheel_valid | wheel_due,
        star_tracker_valid=state.star_tracker_valid | star_due,
        magnetometer_valid=state.magnetometer_valid | mag_due,
        # An arriving invalid Sun packet explicitly marks the sensor unavailable.
        sun_sensor_valid=jnp.where(
            sun_due, sun_available, state.sun_sensor_valid
        ),
        gnss_valid=state.gnss_valid | gnss_due,
        gyro_sample_step=jnp.where(
            gyro_due, gyro_source, state.gyro_sample_step
        ),
        wheel_sample_step=jnp.where(
            wheel_due, wheel_source[:, None], state.wheel_sample_step
        ),
        star_tracker_sample_step=jnp.where(
            star_due, star_source, state.star_tracker_sample_step
        ),
        magnetometer_sample_step=jnp.where(
            mag_due, mag_source, state.magnetometer_sample_step
        ),
        sun_sensor_sample_step=jnp.where(
            sun_due, sun_source, state.sun_sensor_sample_step
        ),
        gnss_sample_step=jnp.where(
            gnss_due, gnss_source, state.gnss_sample_step
        ),
        q_history=q_history,
        omega_history=omega_history,
        wheel_speed_history=wheel_history,
        position_history=position_history,
        velocity_history=velocity_history,
    )


def _age(
    valid: jax.Array,
    sample_step: jax.Array,
    absolute_physics_step: jax.Array,
    physics_dt: float,
) -> jax.Array:
    invalid_age = jnp.asarray(jnp.inf, dtype=jnp.float32)
    while absolute_physics_step.ndim < sample_step.ndim:
        absolute_physics_step = absolute_physics_step[..., None]
    return jnp.where(
        valid,
        (absolute_physics_step - sample_step) * physics_dt,
        invalid_age,
    )


def sensor_age_seconds(
    state: SensorState,
    absolute_physics_step: jax.Array,
    physics: PhysicsConfig,
) -> SensorAges:
    """Returns age of each latest delivered packet in seconds"""
    return SensorAges(
        gyro=_age(
            state.gyro_valid,
            state.gyro_sample_step,
            absolute_physics_step,
            physics.physics_dt,
        ),
        wheel=_age(
            state.wheel_valid,
            state.wheel_sample_step,
            absolute_physics_step,
            physics.physics_dt,
        ),
        star_tracker=_age(
            state.star_tracker_valid,
            state.star_tracker_sample_step,
            absolute_physics_step,
            physics.physics_dt,
        ),
        magnetometer=_age(
            state.magnetometer_valid,
            state.magnetometer_sample_step,
            absolute_physics_step,
            physics.physics_dt,
        ),
        sun_sensor=_age(
            state.sun_sensor_valid,
            state.sun_sensor_sample_step,
            absolute_physics_step,
            physics.physics_dt,
        ),
        gnss=_age(
            state.gnss_valid,
            state.gnss_sample_step,
            absolute_physics_step,
            physics.physics_dt,
        ),
    )
