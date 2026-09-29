from typing import NamedTuple

import jax
import jax.numpy as jnp

from simulators.fdir.config import GuidanceConfig
from simulators.fdir.math3d import rotate_inertial_to_body, rotation_matrix_to_quat
from simulators.fdir.orbit import OrbitState


class GuidanceTarget(NamedTuple):
    q_body_to_inertial: jax.Array
    omega_inertial_rad_s: jax.Array


def _normalize(x: jax.Array, eps: float = 1.0e-12) -> jax.Array:
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)


def guidance_target(orbit: OrbitState, config: GuidanceConfig) -> GuidanceTarget:
    """Returns the desired body attitude and frame angular velocity in ECI.

    For ``nadir_lvlh`` the desired body axes expressed in ECI are:
      +Z: nadir (-r_hat)
      +X: horizontal velocity direction
      +Y: completes the right-handed triad

    The target-frame angular velocity for this circular-orbit reference is
    h/r^2, expressed in inertial coordinates. The implementation uses the
    instantaneous r,v values, so it remains well behaved if the orbit reference
    model is later upgraded beyond the current circular propagator.
    """
    batch = orbit.position_eci_m.shape[0]
    dtype = orbit.position_eci_m.dtype

    if config.mode == "inertial_hold":
        q = jnp.tile(jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=dtype), (batch, 1))
        omega_i = jnp.zeros((batch, 3), dtype=dtype)
        return GuidanceTarget(q, omega_i)

    if config.mode == "nadir_lvlh":
        r = orbit.position_eci_m
        v = orbit.velocity_eci_m_s
        r_hat = _normalize(r)
        z_i = -r_hat

        # Remove any radial velocity component so +X is strictly local-horizontal.
        v_horizontal = v - jnp.sum(v * r_hat, axis=-1, keepdims=True) * r_hat
        x_i = _normalize(v_horizontal)
        y_i = _normalize(jnp.cross(z_i, x_i))
        # Re-orthogonalize X to suppress numerical drift in the triad.
        x_i = _normalize(jnp.cross(y_i, z_i))

        # Columns are desired body basis vectors expressed in inertial coordinates.
        rotation_b_to_i = jnp.stack([x_i, y_i, z_i], axis=-1)
        q = rotation_matrix_to_quat(rotation_b_to_i)

        h_i = jnp.cross(r, v)
        radius_sq = jnp.sum(r * r, axis=-1, keepdims=True)
        omega_i = h_i / (radius_sq + 1.0e-12)
        return GuidanceTarget(q, omega_i)

    raise ValueError(f"Unsupported guidance mode: {config.mode}")


def target_rate_in_estimated_body(
    estimated_q_body_to_inertial: jax.Array,
    target_omega_inertial_rad_s: jax.Array,
) -> jax.Array:
    """Expresses the guidance frame angular velocity in the estimated body frame."""
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
