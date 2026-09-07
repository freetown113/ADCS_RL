from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from simulators.simplified.config import PhysicsConfig
from simulators.simplified.math3d import integrate_quaternion


WHEEL_AXES = (
    jnp.asarray(
        [
            [1.0, -1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
        ],
        dtype=jnp.float32,
    )
    / jnp.sqrt(jnp.asarray(3.0, dtype=jnp.float32))
)


class PhysicalState(NamedTuple):
    q: jax.Array
    omega: jax.Array
    wheel_speed: jax.Array


class ActuationInfo(NamedTuple):
    desired_body_torque: jax.Array
    achieved_body_torque: jax.Array
    motor_torque: jax.Array
    net_rotor_torque: jax.Array
    allocation_error: jax.Array


def body_inertia(config: PhysicsConfig, dtype=jnp.float32) -> jax.Array:
    return jnp.diag(jnp.asarray(config.body_inertia, dtype=dtype))


def body_inertia_inv(config: PhysicsConfig, dtype=jnp.float32) -> jax.Array:
    return jnp.diag(1.0 / jnp.asarray(config.body_inertia, dtype=dtype))


def _speed_limit_active_net_torque(
    net_rotor_torque: jax.Array,
    wheel_speed: jax.Array,
    wheel_control_mask: jax.Array,
    config: PhysicsConfig,
) -> jax.Array:
    """Limits *controlled* rotor torque so one Euler step cannot cross speed limits.

    Failed motors are not modified here: with motor torque forced to zero their
    net rotor torque is only viscous bearing friction, which always points toward
    zero speed and therefore cannot drive a valid wheel farther out of bounds.
    """
    dt = config.physics_dt
    lower = config.wheel_inertia * (-config.max_wheel_speed - wheel_speed) / dt
    upper = config.wheel_inertia * (config.max_wheel_speed - wheel_speed) / dt
    active_limited = jnp.clip(net_rotor_torque, lower, upper)
    return jnp.where(wheel_control_mask > 0.5, active_limited, net_rotor_torque)


def motor_torque_to_net_rotor_torque(
    commanded_motor_torque: jax.Array,
    wheel_speed: jax.Array,
    wheel_control_mask: jax.Array,
    config: PhysicsConfig,
) -> tuple[jax.Array, jax.Array]:
    """Applies motor availability, hardware torque limits, friction, and speed limits.

    A failed wheel motor receives zero motor torque, but its rotor is still part of
    the spacecraft and therefore still experiences bearing friction:
        tau_net = 0 - b * omega_w.
    """
    commanded_motor_torque = jnp.clip(
        commanded_motor_torque,
        -config.max_motor_torque,
        config.max_motor_torque,
    )
    motor_torque = commanded_motor_torque * wheel_control_mask
    net_rotor_torque = motor_torque - config.bearing_friction * wheel_speed
    net_rotor_torque = _speed_limit_active_net_torque(
        net_rotor_torque, wheel_speed, wheel_control_mask, config
    )

    # Reconstruct active motor torque after speed limiting. Failed motors remain 0.
    reconstructed_motor = net_rotor_torque + config.bearing_friction * wheel_speed
    motor_torque = jnp.where(
        wheel_control_mask > 0.5,
        jnp.clip(
            reconstructed_motor,
            -config.max_motor_torque,
            config.max_motor_torque,
        ),
        jnp.zeros_like(reconstructed_motor),
    )
    # Recompute net torque after the final motor clamp.
    net_rotor_torque = motor_torque - config.bearing_friction * wheel_speed
    net_rotor_torque = _speed_limit_active_net_torque(
        net_rotor_torque, wheel_speed, wheel_control_mask, config
    )
    return motor_torque, net_rotor_torque


def allocate_body_torque(
    desired_body_torque: jax.Array,
    wheel_speed: jax.Array,
    wheel_control_mask: jax.Array,
    config: PhysicsConfig,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Allocates desired body torque to available wheel motors.

    ``wheel_control_mask`` says which motors can be commanded. A failed rotor is
    *not* deleted: its bearing friction remains and can produce a passive body
    torque. The active wheels compensate that passive torque when possible.
    """
    dtype = desired_body_torque.dtype
    axes = WHEEL_AXES.astype(dtype)
    active_axes = axes[None, :, :] * wheel_control_mask[..., None, :]

    # Passive failed-wheel friction exists even though failed motor torque is zero.
    passive_net_rotor = (
        -config.bearing_friction
        * wheel_speed
        * (1.0 - wheel_control_mask)
    )
    passive_body_torque = -jnp.einsum(
        "ij,...j->...i", axes, passive_net_rotor
    )
    active_body_target = desired_body_torque - passive_body_torque

    gram = jnp.matmul(active_axes, jnp.swapaxes(active_axes, -1, -2))
    regularizer = config.allocation_regularization * jnp.eye(3, dtype=dtype)
    solved = jnp.linalg.solve(
        gram + regularizer,
        active_body_target[..., None],
    )
    desired_active_net_rotor = -jnp.matmul(
        jnp.swapaxes(active_axes, -1, -2), solved
    )[..., 0]
    desired_active_net_rotor = desired_active_net_rotor * wheel_control_mask

    # For active wheels, compensate viscous friction so the desired *net* rotor
    # torque is produced. Failed motors stay at zero via the helper below.
    commanded_motor = (
        desired_active_net_rotor + config.bearing_friction * wheel_speed
    ) * wheel_control_mask
    motor_torque, net_rotor_torque = motor_torque_to_net_rotor_torque(
        commanded_motor,
        wheel_speed,
        wheel_control_mask,
        config,
    )

    achieved_body_torque = -jnp.einsum(
        "ij,...j->...i", axes, net_rotor_torque
    )
    allocation_error = achieved_body_torque - desired_body_torque
    return motor_torque, net_rotor_torque, achieved_body_torque, allocation_error


def direct_motor_actuation(
    commanded_motor_torque: jax.Array,
    wheel_speed: jax.Array,
    wheel_control_mask: jax.Array,
    config: PhysicsConfig,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Applies four direct wheel-motor torque commands.

    Returns ``(motor_torque, net_rotor_torque, achieved_body_torque)``.
    No body-torque allocator is used.
    """
    motor_torque, net_rotor_torque = motor_torque_to_net_rotor_torque(
        commanded_motor_torque,
        wheel_speed,
        wheel_control_mask,
        config,
    )
    achieved_body_torque = -jnp.einsum(
        "ij,...j->...i", WHEEL_AXES.astype(commanded_motor_torque.dtype), net_rotor_torque
    )
    return motor_torque, net_rotor_torque, achieved_body_torque


def continuous_dynamics(
    state: PhysicalState,
    net_rotor_torque: jax.Array,
    external_body_torque: jax.Array,
    config: PhysicsConfig,
) -> Tuple[jax.Array, jax.Array]:
    """Gyrostat dynamics.

    Total body-frame angular momentum is
    ``h = I_body @ omega + A @ (J_wheel * wheel_speed)`` and obeys
    ``h_dot + omega x h = external_torque``.
    """
    dtype = state.omega.dtype
    inertia = body_inertia(config, dtype)
    inertia_inv = body_inertia_inv(config, dtype)
    axes = WHEEL_AXES.astype(dtype)

    wheel_momentum = jnp.einsum(
        "ij,...j->...i", axes, config.wheel_inertia * state.wheel_speed
    )
    total_momentum = jnp.einsum(
        "ij,...j->...i", inertia, state.omega
    ) + wheel_momentum

    body_reaction = jnp.einsum("ij,...j->...i", axes, net_rotor_torque)
    omega_dot = jnp.einsum(
        "ij,...j->...i",
        inertia_inv,
        external_body_torque
        - jnp.cross(state.omega, total_momentum)
        - body_reaction,
    )
    wheel_speed_dot = net_rotor_torque / config.wheel_inertia
    return omega_dot, wheel_speed_dot


def _integrate_from_net_rotor(
    state: PhysicalState,
    net_rotor_torque: jax.Array,
    external_body_torque: jax.Array,
    config: PhysicsConfig,
) -> PhysicalState:
    omega_dot, wheel_dot = continuous_dynamics(
        state, net_rotor_torque, external_body_torque, config
    )
    next_omega = state.omega + config.physics_dt * omega_dot
    next_wheel_speed = state.wheel_speed + config.physics_dt * wheel_dot
    midpoint_omega = 0.5 * (state.omega + next_omega)
    next_q = integrate_quaternion(state.q, midpoint_omega, config.physics_dt)
    return PhysicalState(next_q, next_omega, next_wheel_speed)


def physics_substep(
    state: PhysicalState,
    desired_body_torque: jax.Array,
    wheel_control_mask: jax.Array,
    external_body_torque: jax.Array,
    config: PhysicsConfig,
) -> Tuple[PhysicalState, ActuationInfo]:
    """One physics step for body-torque control modes."""
    motor, net_rotor, achieved, error = allocate_body_torque(
        desired_body_torque,
        state.wheel_speed,
        wheel_control_mask,
        config,
    )
    next_state = _integrate_from_net_rotor(
        state, net_rotor, external_body_torque, config
    )
    info = ActuationInfo(
        desired_body_torque=desired_body_torque,
        achieved_body_torque=achieved,
        motor_torque=motor,
        net_rotor_torque=net_rotor,
        allocation_error=error,
    )
    return next_state, info


def physics_substep_motor_direct(
    state: PhysicalState,
    commanded_motor_torque: jax.Array,
    wheel_control_mask: jax.Array,
    external_body_torque: jax.Array,
    config: PhysicsConfig,
) -> Tuple[PhysicalState, ActuationInfo]:
    """One physics step for direct four-motor torque control."""
    motor, net_rotor, achieved = direct_motor_actuation(
        commanded_motor_torque,
        state.wheel_speed,
        wheel_control_mask,
        config,
    )
    next_state = _integrate_from_net_rotor(
        state, net_rotor, external_body_torque, config
    )
    # In motor-direct mode there is no requested body torque; report the body
    # torque implied by the requested motor action before hardware/fault effects.
    requested_net = commanded_motor_torque - config.bearing_friction * state.wheel_speed
    implied_body = -jnp.einsum(
        "ij,...j->...i", WHEEL_AXES.astype(commanded_motor_torque.dtype), requested_net
    )
    error = achieved - implied_body
    info = ActuationInfo(
        desired_body_torque=implied_body,
        achieved_body_torque=achieved,
        motor_torque=motor,
        net_rotor_torque=net_rotor,
        allocation_error=error,
    )
    return next_state, info


def momentum_balance_residual(
    state: PhysicalState,
    omega_dot: jax.Array,
    wheel_speed_dot: jax.Array,
    external_body_torque: jax.Array,
    config: PhysicsConfig,
) -> jax.Array:
    """Continuous-time residual; should be approximately zero."""
    dtype = state.omega.dtype
    inertia = body_inertia(config, dtype)
    axes = WHEEL_AXES.astype(dtype)
    momentum = jnp.einsum("ij,...j->...i", inertia, state.omega) + jnp.einsum(
        "ij,...j->...i", axes, config.wheel_inertia * state.wheel_speed
    )
    return (
        jnp.einsum("ij,...j->...i", inertia, omega_dot)
        + jnp.einsum(
            "ij,...j->...i", axes, config.wheel_inertia * wheel_speed_dot
        )
        + jnp.cross(state.omega, momentum)
        - external_body_torque
    )
