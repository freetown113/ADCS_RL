from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from simulators.fdir.config import PhysicsConfig
from simulators.fdir.math3d import integrate_quaternion

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
    motor_torque: jax.Array


class ActuationInfo(NamedTuple):
    desired_body_torque: jax.Array
    achieved_body_torque: jax.Array
    commanded_motor_torque: jax.Array
    motor_torque: jax.Array
    net_rotor_torque: jax.Array
    allocation_error: jax.Array


def body_inertia(config: PhysicsConfig, dtype=jnp.float32) -> jax.Array:
    return jnp.diag(jnp.asarray(config.body_inertia, dtype=dtype))


def body_inertia_inv(config: PhysicsConfig, dtype=jnp.float32) -> jax.Array:
    return jnp.diag(1.0 / jnp.asarray(config.body_inertia, dtype=dtype))


def _quantize(value: jax.Array, quantum: float) -> jax.Array:
    if quantum <= 0.0:
        return value
    q = jnp.asarray(quantum, dtype=value.dtype)
    return q * jnp.round(value / q)


def _command_conditioning(command: jax.Array, config: PhysicsConfig) -> jax.Array:
    command = _quantize(command, config.motor_command_quantization_torque)
    command = jnp.where(
        jnp.abs(command) < config.motor_dead_zone_torque,
        jnp.zeros_like(command),
        command,
    )
    return jnp.clip(command, -config.max_motor_torque, config.max_motor_torque)


def wheel_friction_torque(wheel_speed: jax.Array, motor_torque: jax.Array,
                          config: PhysicsConfig) -> jax.Array:
    """Torque opposing rotor motion, including viscous, Coulomb and stiction."""
    viscous = config.bearing_friction * wheel_speed
    speed_scale = max(config.stiction_speed_rad_s, 1.0e-6)
    dynamic_coulomb = config.coulomb_friction_torque * jnp.tanh(wheel_speed / speed_scale)
    dynamic = viscous + dynamic_coulomb

    near_zero = jnp.abs(wheel_speed) < config.stiction_speed_rad_s
    below_breakaway = jnp.abs(motor_torque) <= config.stiction_torque
    static = jnp.where(
        below_breakaway,
        motor_torque,
        config.stiction_torque * jnp.sign(motor_torque),
    )
    return jnp.where(near_zero, static, dynamic)


def predicted_wheel_friction_torque(wheel_speed: jax.Array, config: PhysicsConfig) -> jax.Array:
    """Controller-side friction estimate based only on wheel telemetry."""
    speed_scale = max(config.stiction_speed_rad_s, 1.0e-6)
    return (
        config.bearing_friction * wheel_speed
        + config.coulomb_friction_torque * jnp.tanh(wheel_speed / speed_scale)
    )


def _speed_limit_net_torque(net_rotor_torque: jax.Array, wheel_speed: jax.Array,
                            config: PhysicsConfig) -> jax.Array:
    dt = config.physics_dt
    lower = config.wheel_inertia * (-config.max_wheel_speed - wheel_speed) / dt
    upper = config.wheel_inertia * (config.max_wheel_speed - wheel_speed) / dt
    return jnp.clip(net_rotor_torque, lower, upper)


def motor_torque_to_net_rotor_torque(
    commanded_motor_torque: jax.Array,
    wheel_speed: jax.Array,
    wheel_control_mask: jax.Array,
    config: PhysicsConfig,
    previous_motor_torque: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Applies electronics, partial authority, motor lag, friction and speed limits.
    ``wheel_control_mask`` is an actual authority factor in [0, 1], so a value of
    0.5 models a degraded wheel and 0 models loss of motor torque. The rotor is
    never deleted: stored momentum and passive bearing friction remain physical.
    """
    command = _command_conditioning(commanded_motor_torque, config)
    target_motor = (
        command
        * wheel_control_mask
        * jnp.asarray(config.motor_torque_scale, dtype=command.dtype)
    )
    target_motor = jnp.clip(target_motor, -config.max_motor_torque, config.max_motor_torque)

    if previous_motor_torque is None:
        # Diagnostic/helper calls that do not carry actuator state get the steady
        # motor response. The environment always supplies the previous torque.
        previous_motor_torque = target_motor

    alpha = 1.0 - jnp.exp(-config.physics_dt / config.motor_time_constant_s)
    motor_torque = previous_motor_torque + alpha * (target_motor - previous_motor_torque)
    motor_torque = jnp.clip(motor_torque, -config.max_motor_torque, config.max_motor_torque)

    friction = wheel_friction_torque(wheel_speed, motor_torque, config)
    net_rotor_torque = _speed_limit_net_torque(motor_torque - friction, wheel_speed, config)

    # When a speed limit clips the net torque, reconstruct the motor torque that
    # is physically consistent with the clipped rotor acceleration.
    reconstructed = net_rotor_torque + friction
    motor_torque = jnp.clip(reconstructed, -config.max_motor_torque, config.max_motor_torque)
    friction = wheel_friction_torque(wheel_speed, motor_torque, config)
    net_rotor_torque = _speed_limit_net_torque(motor_torque - friction, wheel_speed, config)
    return motor_torque, net_rotor_torque


def allocate_body_torque_command(
    desired_body_torque: jax.Array,
    measured_wheel_speed: jax.Array,
    wheel_control_mask: jax.Array,
    config: PhysicsConfig,
) -> jax.Array:
    """Authority-aware bounded controller-side wheel allocation.

    ``wheel_control_mask`` is FDIR-estimated authority, not simulator truth.  The
    allocator solves directly for motor command using the effective wheel matrix
    A*diag(authority), then performs a second residual-allocation pass after torque
    clipping.  This avoids the old behavior in which a 10%-authority wheel was first
    treated as fully capable and only compensated by asking for 10x more command.
    """
    dtype = desired_body_torque.dtype
    axes = WHEEL_AXES.astype(dtype)
    authority = jnp.clip(wheel_control_mask, 0.0, 1.0)
    predicted_friction = predicted_wheel_friction_torque(measured_wheel_speed, config)

    # body torque = -A (authority*u - friction)
    friction_body = jnp.einsum("ij,...j->...i", axes, predicted_friction)
    motor_body_target = desired_body_torque - friction_body
    effective_axes = axes[None, :, :] * authority[..., None, :]
    regularizer = config.allocation_regularization * jnp.eye(3, dtype=dtype)

    def solve(target: jax.Array, available: jax.Array) -> jax.Array:
        matrix = effective_axes * available[..., None, :]
        gram = matrix @ jnp.swapaxes(matrix, -1, -2)
        solved = jnp.linalg.solve(gram + regularizer, target[..., None])
        return -jnp.matmul(jnp.swapaxes(matrix, -1, -2), solved)[..., 0]

    available0 = (authority > 1.0e-4).astype(dtype)
    command0 = solve(motor_body_target, available0) * available0
    command0 = jnp.clip(command0, -config.max_motor_torque, config.max_motor_torque)

    predicted_net0 = authority * command0 - predicted_friction
    predicted_body0 = -jnp.einsum("ij,...j->...i", axes, predicted_net0)
    residual = desired_body_torque - predicted_body0

    # Do not ask already saturated channels to absorb the residual.
    unsaturated = (jnp.abs(command0) < (config.max_motor_torque - 1.0e-6)).astype(dtype) * available0
    correction = solve(residual, unsaturated) * unsaturated
    command = jnp.clip(command0 + correction, -config.max_motor_torque, config.max_motor_torque)
    return jnp.where(available0 > 0.5, command, 0.0)


def allocate_body_torque(
    desired_body_torque: jax.Array,
    wheel_speed: jax.Array,
    wheel_control_mask: jax.Array,
    config: PhysicsConfig,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    commanded = allocate_body_torque_command(desired_body_torque, wheel_speed, wheel_control_mask, config)
    motor, net = motor_torque_to_net_rotor_torque(
        commanded, wheel_speed, wheel_control_mask, config, previous_motor_torque=None
    )
    achieved = -jnp.einsum("ij,...j->...i", WHEEL_AXES.astype(desired_body_torque.dtype), net)
    return motor, net, achieved, achieved - desired_body_torque


def direct_motor_actuation(
    commanded_motor_torque: jax.Array,
    wheel_speed: jax.Array,
    wheel_control_mask: jax.Array,
    config: PhysicsConfig,
    previous_motor_torque: jax.Array | None = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    motor, net = motor_torque_to_net_rotor_torque(
        commanded_motor_torque, wheel_speed, wheel_control_mask, config,
        previous_motor_torque=previous_motor_torque,
    )
    achieved = -jnp.einsum("ij,...j->...i", WHEEL_AXES.astype(commanded_motor_torque.dtype), net)
    return motor, net, achieved


def continuous_dynamics(
    state: PhysicalState,
    net_rotor_torque: jax.Array,
    external_body_torque: jax.Array,
    config: PhysicsConfig,
) -> Tuple[jax.Array, jax.Array]:
    dtype = state.omega.dtype
    inertia = body_inertia(config, dtype)
    inertia_inv = body_inertia_inv(config, dtype)
    axes = WHEEL_AXES.astype(dtype)
    wheel_momentum = jnp.einsum("ij,...j->...i", axes, config.wheel_inertia * state.wheel_speed)
    total_momentum = jnp.einsum("ij,...j->...i", inertia, state.omega) + wheel_momentum
    body_reaction = jnp.einsum("ij,...j->...i", axes, net_rotor_torque)
    omega_dot = jnp.einsum(
        "ij,...j->...i",
        inertia_inv,
        external_body_torque - jnp.cross(state.omega, total_momentum) - body_reaction,
    )
    return omega_dot, net_rotor_torque / config.wheel_inertia


def _integrate_from_net_rotor(
    state: PhysicalState,
    motor_torque: jax.Array,
    net_rotor_torque: jax.Array,
    external_body_torque: jax.Array,
    config: PhysicsConfig,
) -> PhysicalState:
    omega_dot, wheel_dot = continuous_dynamics(state, net_rotor_torque, external_body_torque, config)
    next_omega = state.omega + config.physics_dt * omega_dot
    next_wheel_speed = state.wheel_speed + config.physics_dt * wheel_dot
    midpoint_omega = 0.5 * (state.omega + next_omega)
    next_q = integrate_quaternion(state.q, midpoint_omega, config.physics_dt)
    return PhysicalState(next_q, next_omega, next_wheel_speed, motor_torque)


def physics_substep(
    state: PhysicalState,
    desired_body_torque: jax.Array,
    wheel_control_mask: jax.Array,
    external_body_torque: jax.Array,
    config: PhysicsConfig,
    measured_wheel_speed: jax.Array | None = None,
    physical_wheel_authority: jax.Array | None = None,
) -> Tuple[PhysicalState, ActuationInfo]:
    if measured_wheel_speed is None:
        measured_wheel_speed = state.wheel_speed
    commanded = allocate_body_torque_command(
        desired_body_torque, measured_wheel_speed, wheel_control_mask, config
    )
    if physical_wheel_authority is None:
        physical_wheel_authority = wheel_control_mask
    motor, net = motor_torque_to_net_rotor_torque(
        commanded, state.wheel_speed, physical_wheel_authority, config,
        previous_motor_torque=state.motor_torque,
    )
    achieved = -jnp.einsum("ij,...j->...i", WHEEL_AXES.astype(desired_body_torque.dtype), net)
    next_state = _integrate_from_net_rotor(state, motor, net, external_body_torque, config)
    return next_state, ActuationInfo(
        desired_body_torque=desired_body_torque,
        achieved_body_torque=achieved,
        commanded_motor_torque=commanded,
        motor_torque=motor,
        net_rotor_torque=net,
        allocation_error=achieved - desired_body_torque,
    )


def physics_substep_motor_direct(
    state: PhysicalState,
    commanded_motor_torque: jax.Array,
    wheel_control_mask: jax.Array,
    external_body_torque: jax.Array,
    config: PhysicsConfig,
) -> Tuple[PhysicalState, ActuationInfo]:
    motor, net, achieved = direct_motor_actuation(
        commanded_motor_torque, state.wheel_speed, wheel_control_mask, config,
        previous_motor_torque=state.motor_torque,
    )
    next_state = _integrate_from_net_rotor(state, motor, net, external_body_torque, config)
    predicted_friction = predicted_wheel_friction_torque(state.wheel_speed, config)
    implied_net = commanded_motor_torque * wheel_control_mask - predicted_friction
    implied_body = -jnp.einsum("ij,...j->...i", WHEEL_AXES.astype(commanded_motor_torque.dtype), implied_net)
    return next_state, ActuationInfo(
        desired_body_torque=implied_body,
        achieved_body_torque=achieved,
        commanded_motor_torque=commanded_motor_torque,
        motor_torque=motor,
        net_rotor_torque=net,
        allocation_error=achieved - implied_body,
    )


def momentum_balance_residual(
    state: PhysicalState,
    omega_dot: jax.Array,
    wheel_speed_dot: jax.Array,
    external_body_torque: jax.Array,
    config: PhysicsConfig,
) -> jax.Array:
    dtype = state.omega.dtype
    inertia = body_inertia(config, dtype)
    axes = WHEEL_AXES.astype(dtype)
    momentum = jnp.einsum("ij,...j->...i", inertia, state.omega) + jnp.einsum(
        "ij,...j->...i", axes, config.wheel_inertia * state.wheel_speed
    )
    return (
        jnp.einsum("ij,...j->...i", inertia, omega_dot)
        + jnp.einsum("ij,...j->...i", axes, config.wheel_inertia * wheel_speed_dot)
        + jnp.cross(state.omega, momentum)
        - external_body_torque
    )
