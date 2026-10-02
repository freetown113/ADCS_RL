from typing import NamedTuple

import jax
import jax.numpy as jnp

from simulators.fdir.config import MagnetorquerConfig, PhysicsConfig
from simulators.fdir.physics import WHEEL_AXES


class MagnetorquerState(NamedTuple):
    commanded_dipole_body_Am2: jax.Array
    actual_dipole_body_Am2: jax.Array


def reset_magnetorquer_state(batch_size: int, dtype=jnp.float32) -> MagnetorquerState:
    z = jnp.zeros((batch_size, 3), dtype=dtype)
    return MagnetorquerState(z, z)


def _quantize(x: jax.Array, quantum: float) -> jax.Array:
    if quantum <= 0.0:
        return x
    q = jnp.asarray(quantum, x.dtype)
    return q * jnp.round(x / q)


def condition_dipole(command: jax.Array, config: MagnetorquerConfig) -> jax.Array:
    command = _quantize(command, config.dipole_quantization_Am2)
    return jnp.clip(command, -config.max_dipole_Am2, config.max_dipole_Am2)


def magnetorquer_substep(
    state: MagnetorquerState,
    commanded_dipole_body_Am2: jax.Array,
    dt: float,
    config: MagnetorquerConfig,
) -> MagnetorquerState:
    cmd = condition_dipole(commanded_dipole_body_Am2, config) if config.enabled else jnp.zeros_like(commanded_dipole_body_Am2)
    alpha = 1.0 - jnp.exp(-jnp.asarray(dt, cmd.dtype) / jnp.asarray(config.time_constant_s, cmd.dtype))
    actual = state.actual_dipole_body_Am2 + alpha * (cmd - state.actual_dipole_body_Am2)
    return MagnetorquerState(cmd, actual)


def magnetic_torque_body(dipole_body_Am2: jax.Array, magnetic_field_body_t: jax.Array) -> jax.Array:
    return jnp.cross(dipole_body_Am2, magnetic_field_body_t)


def bdot_detumble_command(
    gyro_body_rad_s: jax.Array,
    magnetic_field_body_t: jax.Array,
    config: MagnetorquerConfig,
) -> jax.Array:
    # dB_body/dt ~= -omega x B, hence m=-k dB/dt ~= k(omega x B).
    command = config.bdot_gain_Am2_s_per_t * jnp.cross(gyro_body_rad_s, magnetic_field_body_t)
    valid = jnp.linalg.norm(magnetic_field_body_t, axis=-1) > config.minimum_field_t
    return jnp.where(valid[..., None], condition_dipole(command, config), 0.0)


def wheel_momentum_body(wheel_speed_rad_s: jax.Array, physics: PhysicsConfig) -> jax.Array:
    return jnp.einsum(
        "ij,...j->...i",
        WHEEL_AXES.astype(wheel_speed_rad_s.dtype),
        physics.wheel_inertia * wheel_speed_rad_s,
    )


def momentum_unload_command(
    wheel_speed_rad_s: jax.Array,
    magnetic_field_body_t: jax.Array,
    physics: PhysicsConfig,
    config: MagnetorquerConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return dipole and predicted achievable external unloading body torque."""
    h = wheel_momentum_body(wheel_speed_rad_s, physics)
    b2 = jnp.sum(magnetic_field_body_t * magnetic_field_body_t, axis=-1, keepdims=True)
    b_hat = magnetic_field_body_t / (jnp.sqrt(b2) + 1.0e-12)
    desired = -config.momentum_unload_gain_per_s * h
    achievable = desired - jnp.sum(desired * b_hat, axis=-1, keepdims=True) * b_hat
    dipole = jnp.cross(magnetic_field_body_t, achievable) / (b2 + 1.0e-16)
    valid = jnp.sqrt(b2[..., 0]) > config.minimum_field_t
    dipole = jnp.where(valid[..., None], condition_dipole(dipole, config), 0.0)
    predicted_torque = magnetic_torque_body(dipole, magnetic_field_body_t)
    return dipole, predicted_torque
