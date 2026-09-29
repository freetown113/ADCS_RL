from typing import NamedTuple

import jax
import jax.numpy as jnp

from simulators.fdir.config import OrbitConfig


class OrbitState(NamedTuple):
    position_eci_m: jax.Array
    velocity_eci_m_s: jax.Array
    time_s: jax.Array
    magnetic_field_eci_t: jax.Array
    sun_direction_eci: jax.Array
    sun_visible: jax.Array


def _normalize(vector: jax.Array, eps: float = 1.0e-12) -> jax.Array:
    return vector / (jnp.linalg.norm(vector, axis=-1, keepdims=True) + eps)


def _orbit_basis(config: OrbitConfig, dtype=jnp.float32) -> tuple[jax.Array, jax.Array]:
    """Returns orthonormal in-plane basis vectors p_hat and q_hat in ECI."""
    inclination = jnp.deg2rad(jnp.asarray(config.inclination_deg, dtype=dtype))
    raan = jnp.deg2rad(jnp.asarray(config.raan_deg, dtype=dtype))

    p_hat = jnp.asarray(
        [jnp.cos(raan), jnp.sin(raan), 0.0], dtype=dtype
    )
    q_hat = jnp.asarray(
        [
            -jnp.sin(raan) * jnp.cos(inclination),
            jnp.cos(raan) * jnp.cos(inclination),
            jnp.sin(inclination),
        ],
        dtype=dtype,
    )
    return p_hat, q_hat


def magnetic_dipole_axis_eci(config: OrbitConfig, dtype=jnp.float32) -> jax.Array:
    """Fixed unit magnetic dipole axis used by the centered-dipole model."""
    tilt = jnp.deg2rad(jnp.asarray(config.magnetic_dipole_tilt_deg, dtype=dtype))
    longitude = jnp.deg2rad(
        jnp.asarray(config.magnetic_dipole_longitude_deg, dtype=dtype)
    )
    axis = jnp.asarray(
        [
            jnp.sin(tilt) * jnp.cos(longitude),
            jnp.sin(tilt) * jnp.sin(longitude),
            jnp.cos(tilt),
        ],
        dtype=dtype,
    )
    return _normalize(axis)


def magnetic_field_eci(position_eci_m: jax.Array, config: OrbitConfig) -> jax.Array:
    """Centered-dipole magnetic field in ECI, ``magnetic_equator_field_t`` is 
    the surface equatorial magnitude. This is a deliberately compact reference 
    model suitable for estimator integration tests.
    """
    dtype = position_eci_m.dtype
    radius = jnp.linalg.norm(position_eci_m, axis=-1, keepdims=True)
    r_hat = position_eci_m / (radius + 1.0e-12)
    dipole = magnetic_dipole_axis_eci(config, dtype)
    projection = jnp.sum(r_hat * dipole, axis=-1, keepdims=True)
    scale = (
        jnp.asarray(config.magnetic_equator_field_t, dtype=dtype)
        * (jnp.asarray(config.earth_radius_m, dtype=dtype) / radius) ** 3
    )
    return scale * (3.0 * projection * r_hat - dipole)


def sun_direction_eci(config: OrbitConfig, dtype=jnp.float32) -> jax.Array:
    vector = jnp.asarray(config.sun_direction_eci, dtype=dtype)
    return _normalize(vector)


def earth_eclipse_mask(
    position_eci_m: jax.Array,
    sun_eci: jax.Array,
    config: OrbitConfig,
) -> jax.Array:
    """True where Earth geometrically blocks the Sun from direct view"""
    sun = _normalize(sun_eci)
    along_sun = jnp.sum(position_eci_m * sun, axis=-1)
    perpendicular = position_eci_m - along_sun[..., None] * sun
    distance_to_axis = jnp.linalg.norm(perpendicular, axis=-1)
    return (along_sun < 0.0) & (distance_to_axis < config.earth_radius_m)


def reset_orbit_state(batch_size: int, config: OrbitConfig) -> OrbitState:
    dtype = jnp.float32
    p_hat, q_hat = _orbit_basis(config, dtype)
    phase = jnp.deg2rad(
        jnp.asarray(config.initial_argument_of_latitude_deg, dtype=dtype)
    )
    radius = jnp.asarray(config.earth_radius_m + config.altitude_m, dtype=dtype)
    speed = jnp.sqrt(jnp.asarray(config.earth_mu_m3_s2, dtype=dtype) / radius)

    position_single = radius * (
        jnp.cos(phase) * p_hat + jnp.sin(phase) * q_hat
    )
    velocity_single = speed * (
        -jnp.sin(phase) * p_hat + jnp.cos(phase) * q_hat
    )
    position = jnp.broadcast_to(position_single, (batch_size, 3))
    velocity = jnp.broadcast_to(velocity_single, (batch_size, 3))
    sun_single = sun_direction_eci(config, dtype)
    sun = jnp.broadcast_to(sun_single, (batch_size, 3))
    field = magnetic_field_eci(position, config)
    visible = ~earth_eclipse_mask(position, sun, config)
    return OrbitState(
        position_eci_m=position,
        velocity_eci_m_s=velocity,
        time_s=jnp.zeros((batch_size,), dtype=dtype),
        magnetic_field_eci_t=field,
        sun_direction_eci=sun,
        sun_visible=visible,
    )


def _rotate_about_axis(vector: jax.Array, axis: jax.Array, angle: jax.Array) -> jax.Array:
    """Batch-safe Rodrigues rotation."""
    axis = _normalize(axis)
    cosine = jnp.cos(angle)[..., None]
    sine = jnp.sin(angle)[..., None]
    projection = jnp.sum(vector * axis, axis=-1, keepdims=True)
    return (
        vector * cosine
        + jnp.cross(axis, vector) * sine
        + axis * projection * (1.0 - cosine)
    )


def orbit_substep(state: OrbitState, dt: float, config: OrbitConfig) -> OrbitState:
    """Exact one-step propagation of the configured circular orbit."""
    dtype = state.position_eci_m.dtype
    radius_scalar = jnp.asarray(
        config.earth_radius_m + config.altitude_m, dtype=dtype
    )
    mean_motion = jnp.sqrt(
        jnp.asarray(config.earth_mu_m3_s2, dtype=dtype) / radius_scalar**3
    )
    orbit_normal = _normalize(
        jnp.cross(state.position_eci_m, state.velocity_eci_m_s)
    )
    angle = jnp.broadcast_to(mean_motion * dt, state.time_s.shape)
    rotated_position = _rotate_about_axis(
        state.position_eci_m, orbit_normal, angle
    )
    position = _normalize(rotated_position) * radius_scalar
    speed = jnp.sqrt(
        jnp.asarray(config.earth_mu_m3_s2, dtype=dtype) / radius_scalar
    )
    velocity_direction = _normalize(jnp.cross(orbit_normal, position))
    velocity = speed * velocity_direction
    field = magnetic_field_eci(position, config)
    visible = ~earth_eclipse_mask(position, state.sun_direction_eci, config)
    return OrbitState(
        position_eci_m=position,
        velocity_eci_m_s=velocity,
        time_s=state.time_s + dt,
        magnetic_field_eci_t=field,
        sun_direction_eci=state.sun_direction_eci,
        sun_visible=visible,
    )
