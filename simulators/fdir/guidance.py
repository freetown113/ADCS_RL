from typing import NamedTuple

import jax
import jax.numpy as jnp

from simulators.fdir.config import GuidanceConfig, OrbitConfig
from simulators.fdir.math3d import (
    canonicalize_quat,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_to_rotation_vector,
    rotate_body_to_inertial,
    rotate_inertial_to_body,
    rotation_matrix_to_quat,
    rotation_vector_to_quat,
)
from simulators.fdir.orbit import OrbitState


class GuidanceTarget(NamedTuple):
    q_body_to_inertial: jax.Array
    omega_inertial_rad_s: jax.Array
    # True when the mission reference is geometrically meaningful/available.
    # Earth-fixed targets become false while occulted by Earth; the returned q/ω
    # then follow the configured fallback reference.
    reference_valid: jax.Array
    # Primary pointing direction in ECI (LOS, Sun, nadir, or target +Z direction).
    reference_direction_eci: jax.Array


def _normalize(x: jax.Array, eps: float = 1.0e-12) -> jax.Array:
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _normalize_with_derivative(x: jax.Array, x_dot: jax.Array, eps: float = 1.0e-12) -> tuple[jax.Array, jax.Array]:
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    unit = x / (norm + eps)
    derivative = (x_dot - unit * jnp.sum(unit * x_dot, axis=-1, keepdims=True)) / (norm + eps)
    return unit, derivative


def body_axis_vector(axis_name: str, dtype=jnp.float32) -> jax.Array:
    axes = {
        "+X": (1.0, 0.0, 0.0), "-X": (-1.0, 0.0, 0.0),
        "+Y": (0.0, 1.0, 0.0), "-Y": (0.0, -1.0, 0.0),
        "+Z": (0.0, 0.0, 1.0), "-Z": (0.0, 0.0, -1.0),
    }
    return jnp.asarray(axes[axis_name], dtype=dtype)


def _principal_axis_post_rotation(axis_name: str, dtype) -> jax.Array:
    """P such that P @ requested_body_axis = +Z of a base pointing frame."""
    matrices = {
        "+Z": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        "-Z": ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
        "+X": ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
        "-X": ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
        "+Y": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
        "-Y": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    }
    return jnp.asarray(matrices[axis_name], dtype=dtype)


def _apply_pointing_axis(rotation_base: jax.Array, axis_name: str) -> jax.Array:
    return rotation_base @ _principal_axis_post_rotation(axis_name, rotation_base.dtype)


def _fixed_roll_pointing(direction_eci: jax.Array, axis_name: str) -> jax.Array:
    """Construct attitude pointing one principal body axis along direction_eci."""
    z_i = _normalize(direction_eci)
    dtype = z_i.dtype
    ref_z = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0], dtype=dtype), z_i.shape)
    ref_x = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0], dtype=dtype), z_i.shape)
    x_from_z = ref_z - jnp.sum(ref_z * z_i, axis=-1, keepdims=True) * z_i
    x_from_x = ref_x - jnp.sum(ref_x * z_i, axis=-1, keepdims=True) * z_i
    use_x = jnp.linalg.norm(x_from_z, axis=-1, keepdims=True) < 1.0e-5
    x_i = _normalize(jnp.where(use_x, x_from_x, x_from_z))
    y_i = _normalize(jnp.cross(z_i, x_i))
    x_i = _normalize(jnp.cross(y_i, z_i))
    base = jnp.stack([x_i, y_i, z_i], axis=-1)
    return rotation_matrix_to_quat(_apply_pointing_axis(base, axis_name))


def _inertial_target(orbit: OrbitState, config: GuidanceConfig) -> GuidanceTarget:
    batch = orbit.position_eci_m.shape[0]
    dtype = orbit.position_eci_m.dtype
    q_single = quat_normalize(jnp.asarray(config.inertial_target_q, dtype=dtype)[None, :])[0]
    q = jnp.broadcast_to(q_single, (batch, 4))
    omega = jnp.zeros((batch, 3), dtype=dtype)
    body_z = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0], dtype=dtype), (batch, 3))
    direction = _normalize(rotate_body_to_inertial(q, body_z))
    return GuidanceTarget(q, omega, jnp.ones((batch,), jnp.bool_), direction)


def _nadir_target(orbit: OrbitState) -> GuidanceTarget:
    r = orbit.position_eci_m
    v = orbit.velocity_eci_m_s
    r_hat = _normalize(r)
    z_i = -r_hat

    v_horizontal = v - jnp.sum(v * r_hat, axis=-1, keepdims=True) * r_hat
    x_i = _normalize(v_horizontal)
    y_i = _normalize(jnp.cross(z_i, x_i))
    x_i = _normalize(jnp.cross(y_i, z_i))

    rotation_b_to_i = jnp.stack([x_i, y_i, z_i], axis=-1)
    q = rotation_matrix_to_quat(rotation_b_to_i)

    h_i = jnp.cross(r, v)
    radius_sq = jnp.sum(r * r, axis=-1, keepdims=True)
    omega_i = h_i / (radius_sq + 1.0e-12)
    valid = jnp.ones(r.shape[:-1], jnp.bool_)
    return GuidanceTarget(q, omega_i, valid, z_i)


def earth_fixed_position_eci(
    time_s: jax.Array,
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
    orbit_config: OrbitConfig,
    dtype=jnp.float32,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Spherical-Earth fixed point position, velocity and acceleration in ECI."""
    lat = jnp.deg2rad(jnp.asarray(latitude_deg, dtype=dtype))
    lon = jnp.deg2rad(jnp.asarray(longitude_deg, dtype=dtype))
    radius = jnp.asarray(orbit_config.earth_radius_m + altitude_m, dtype=dtype)
    ecef = radius * jnp.asarray(
        [jnp.cos(lat) * jnp.cos(lon), jnp.cos(lat) * jnp.sin(lon), jnp.sin(lat)],
        dtype=dtype,
    )
    theta = (
        jnp.deg2rad(jnp.asarray(orbit_config.initial_greenwich_angle_deg, dtype=dtype))
        + jnp.asarray(orbit_config.earth_rotation_rate_rad_s, dtype=dtype) * time_s
    )
    c, s = jnp.cos(theta), jnp.sin(theta)
    x = c * ecef[0] - s * ecef[1]
    y = s * ecef[0] + c * ecef[1]
    z = jnp.broadcast_to(ecef[2], time_s.shape)
    position = jnp.stack([x, y, z], axis=-1)
    earth_omega = jnp.broadcast_to(
        jnp.asarray([0.0, 0.0, orbit_config.earth_rotation_rate_rad_s], dtype=dtype),
        position.shape,
    )
    velocity = jnp.cross(earth_omega, position)
    acceleration = jnp.cross(earth_omega, velocity)
    return position, velocity, acceleration


def earth_target_line_of_sight(
    orbit: OrbitState,
    config: GuidanceConfig,
    orbit_config: OrbitConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    target_r, target_v, target_a = earth_fixed_position_eci(
        orbit.time_s,
        config.earth_target_lat_deg,
        config.earth_target_lon_deg,
        config.earth_target_alt_m,
        orbit_config,
        orbit.position_eci_m.dtype,
    )
    los = target_r - orbit.position_eci_m
    los_hat = _normalize(los)
    target_up = _normalize(target_r)
    # A surface target sees the spacecraft iff it is above the local horizon.
    visible = jnp.sum((orbit.position_eci_m - target_r) * target_up, axis=-1) > 0.0
    return los_hat, visible, target_v, target_a


def _earth_tracking_target(
    orbit: OrbitState,
    config: GuidanceConfig,
    orbit_config: OrbitConfig,
) -> GuidanceTarget:
    r_s = orbit.position_eci_m
    v_s = orbit.velocity_eci_m_s
    target_r, target_v, target_a = earth_fixed_position_eci(
        orbit.time_s,
        config.earth_target_lat_deg,
        config.earth_target_lon_deg,
        config.earth_target_alt_m,
        orbit_config,
        r_s.dtype,
    )
    rel_r = target_r - r_s
    rel_v = target_v - v_s
    sat_radius = jnp.linalg.norm(r_s, axis=-1, keepdims=True)
    sat_a = -jnp.asarray(orbit_config.earth_mu_m3_s2, dtype=r_s.dtype) * r_s / (sat_radius**3 + 1.0e-12)
    rel_a = target_a - sat_a

    z_i, z_dot = _normalize_with_derivative(rel_r, rel_v)
    radial_rate = jnp.sum(rel_v * z_i, axis=-1, keepdims=True)
    tangent = rel_v - radial_rate * z_i
    radial_accel = jnp.sum(rel_a * z_i, axis=-1, keepdims=True) + jnp.sum(rel_v * z_dot, axis=-1, keepdims=True)
    tangent_dot = rel_a - radial_accel * z_i - radial_rate * z_dot

    # Near the instant at which LOS motion becomes tiny, use horizontal spacecraft
    # velocity as a deterministic roll reference instead of dividing by ~zero.
    r_hat = _normalize(r_s)
    fallback = v_s - jnp.sum(v_s * r_hat, axis=-1, keepdims=True) * r_hat
    fallback_dot = rel_a - jnp.sum(rel_a * z_i, axis=-1, keepdims=True) * z_i
    tangent_small = jnp.linalg.norm(tangent, axis=-1, keepdims=True) < 1.0e-4
    tangent = jnp.where(tangent_small, fallback, tangent)
    tangent_dot = jnp.where(tangent_small, fallback_dot, tangent_dot)

    x_i, x_dot = _normalize_with_derivative(tangent, tangent_dot)
    y_i = _normalize(jnp.cross(z_i, x_i))
    # Re-orthogonalize X. The derivative expression below is sufficiently accurate
    # for the smooth non-degenerate passes used by this scenario.
    x_i = _normalize(jnp.cross(y_i, z_i))
    y_dot = jnp.cross(z_dot, x_i) + jnp.cross(z_i, x_dot)

    omega_i = 0.5 * (
        jnp.cross(x_i, x_dot) + jnp.cross(y_i, y_dot) + jnp.cross(z_i, z_dot)
    )
    base_rotation = jnp.stack([x_i, y_i, z_i], axis=-1)
    q_track = rotation_matrix_to_quat(
        _apply_pointing_axis(base_rotation, config.earth_tracking_body_axis)
    )

    target_up = _normalize(target_r)
    visible = jnp.sum((r_s - target_r) * target_up, axis=-1) > 0.0

    fallback = _nadir_target(orbit) if config.earth_target_fallback_mode == "nadir_lvlh" else _inertial_target(orbit, config)
    q = jnp.where(visible[:, None], q_track, fallback.q_body_to_inertial)
    omega = jnp.where(visible[:, None], omega_i, fallback.omega_inertial_rad_s)
    direction = jnp.where(visible[:, None], z_i, fallback.reference_direction_eci)
    return GuidanceTarget(q, omega, visible, direction)


def _sun_target(orbit: OrbitState, config: GuidanceConfig) -> GuidanceTarget:
    sun = _normalize(orbit.sun_direction_eci)
    q = _fixed_roll_pointing(sun, config.sun_pointing_body_axis)
    omega = jnp.zeros_like(orbit.velocity_eci_m_s)
    # Guidance to the inertial Sun direction remains defined in eclipse. Sensor
    # availability is a separate FDIR/environment concept.
    valid = jnp.ones(orbit.time_s.shape, dtype=jnp.bool_)
    return GuidanceTarget(q, omega, valid, sun)


def _scheduled_slew_target(orbit: OrbitState, config: GuidanceConfig) -> GuidanceTarget:
    dtype = orbit.position_eci_m.dtype
    batch = orbit.position_eci_m.shape[0]
    q0 = quat_normalize(jnp.asarray(config.slew_start_q, dtype=dtype)[None, :])[0]
    q1 = quat_normalize(jnp.asarray(config.slew_end_q, dtype=dtype)[None, :])[0]
    q_rel = canonicalize_quat(quat_multiply(quat_conjugate(q0), q1))
    rotvec = quat_to_rotation_vector(q_rel)
    angle = jnp.linalg.norm(rotvec)
    axis = rotvec / (angle + 1.0e-12)

    vmax = jnp.deg2rad(jnp.asarray(config.slew_max_rate_deg_s, dtype=dtype))
    accel = jnp.deg2rad(jnp.asarray(config.slew_max_accel_deg_s2, dtype=dtype))
    triangular = angle <= (vmax * vmax / accel)
    t_accel_tri = jnp.sqrt(angle / accel)
    t_accel_trap = vmax / accel
    t_accel = jnp.where(triangular, t_accel_tri, t_accel_trap)
    v_peak = accel * t_accel
    t_flat = jnp.where(triangular, 0.0, (angle - v_peak * v_peak / accel) / v_peak)
    total = 2.0 * t_accel + t_flat

    t = orbit.time_s - jnp.asarray(config.slew_start_seconds, dtype=dtype)
    t_nonneg = jnp.maximum(t, 0.0)
    theta_acc = 0.5 * accel * t_nonneg**2
    theta_at_acc = 0.5 * accel * t_accel**2
    theta_flat = theta_at_acc + v_peak * (t_nonneg - t_accel)
    remaining = jnp.maximum(total - t_nonneg, 0.0)
    theta_dec = angle - 0.5 * accel * remaining**2
    theta = jnp.where(
        t <= 0.0,
        0.0,
        jnp.where(
            t_nonneg < t_accel,
            theta_acc,
            jnp.where(t_nonneg < t_accel + t_flat, theta_flat, jnp.where(t_nonneg < total, theta_dec, angle)),
        ),
    )
    rate = jnp.where(
        t <= 0.0,
        0.0,
        jnp.where(
            t_nonneg < t_accel,
            accel * t_nonneg,
            jnp.where(t_nonneg < t_accel + t_flat, v_peak, jnp.where(t_nonneg < total, accel * remaining, 0.0)),
        ),
    )

    q_rel_t = rotation_vector_to_quat(axis[None, :] * theta[:, None])
    q = quat_normalize(quat_multiply(jnp.broadcast_to(q0, (batch, 4)), q_rel_t))
    axis_i_single = rotate_body_to_inertial(q0, axis)
    omega = jnp.broadcast_to(axis_i_single, (batch, 3)) * rate[:, None]
    body_z = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0], dtype=dtype), (batch, 3))
    direction = _normalize(rotate_body_to_inertial(q, body_z))
    return GuidanceTarget(q, omega, jnp.ones((batch,), jnp.bool_), direction)


def guidance_target(
    orbit: OrbitState,
    config: GuidanceConfig,
    orbit_config: OrbitConfig | None = None,
) -> GuidanceTarget:
    """Return mission target attitude and its inertial angular velocity."""
    if config.mode == "inertial_hold":
        return _inertial_target(orbit, config)
    if config.mode == "nadir_lvlh":
        return _nadir_target(orbit)
    if config.mode in ("ground_target", "ground_station"):
        if orbit_config is None:
            raise ValueError("Earth-fixed guidance requires OrbitConfig")
        return _earth_tracking_target(orbit, config, orbit_config)
    if config.mode == "sun_pointing":
        return _sun_target(orbit, config)
    if config.mode == "scheduled_slew":
        return _scheduled_slew_target(orbit, config)
    raise ValueError(f"Unsupported guidance mode: {config.mode}")


def safe_sun_target(orbit: OrbitState, config: GuidanceConfig) -> GuidanceTarget:
    """Supervisor safe reference, separate from the configured mission mode."""
    return _sun_target(orbit, config)


def target_rate_in_estimated_body(
    estimated_q_body_to_inertial: jax.Array,
    target_omega_inertial_rad_s: jax.Array,
) -> jax.Array:
    return rotate_inertial_to_body(
        estimated_q_body_to_inertial,
        target_omega_inertial_rad_s,
    )


def tracking_rate_error_body(
    estimated_q_body_to_inertial: jax.Array,
    estimated_omega_body_rad_s: jax.Array,
    target_omega_inertial_rad_s: jax.Array,
) -> jax.Array:
    """Body-frame angular-rate error relative to a moving attitude reference."""
    return estimated_omega_body_rad_s - target_rate_in_estimated_body(
        estimated_q_body_to_inertial,
        target_omega_inertial_rad_s,
    )
