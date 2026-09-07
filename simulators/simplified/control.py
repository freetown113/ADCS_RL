import jax
import jax.numpy as jnp

from simulators.simplified.env import EnvState, SatelliteEnv
from simulators.simplified.math3d import attitude_error
from simulators.simplified.physics import allocate_body_torque


def pd_gains(env: SatelliteEnv) -> tuple[jax.Array, jax.Array]:
    control = env.config.control
    inertia = jnp.asarray(env.config.physics.body_inertia, dtype=jnp.float32)
    wn = control.pd_natural_frequency
    zeta = control.pd_damping
    # q_error.xyz ~= theta/2 near the target.
    kp = 2.0 * inertia * wn**2
    kd = 2.0 * zeta * inertia * wn
    return kp, kd


def normalized_pd_body_action(env: SatelliteEnv, state: EnvState) -> jax.Array:
    """Three normalized body-torque commands produced by the PD controller."""
    kp, kd = pd_gains(env)
    error_q = attitude_error(state.target_q, state.physical.q)
    desired_torque = -kp * error_q[..., 1:] - kd * state.physical.omega
    torque_limit = jnp.asarray(
        env.config.physics.body_torque_limit, dtype=desired_torque.dtype
    )
    return jnp.clip(desired_torque / torque_limit, -1.0, 1.0)


def body_action_to_motor_action(
    env: SatelliteEnv,
    state: EnvState,
    body_action: jax.Array,
) -> jax.Array:
    """Uses the deterministic allocator as a teacher for four motor commands"""
    p = env.config.physics
    desired_body_torque = jnp.clip(body_action, -1.0, 1.0) * jnp.asarray(
        p.body_torque_limit, dtype=body_action.dtype
    )
    motor_torque, _, _, _ = allocate_body_torque(
        desired_body_torque,
        state.physical.wheel_speed,
        state.wheel_mask,
        p,
    )
    return jnp.clip(motor_torque / p.motor_control_limit, -1.0, 1.0)


def pd_command_action(env: SatelliteEnv, state: EnvState) -> jax.Array:
    """Returns a valid environment command for the current control mode."""
    body_action = normalized_pd_body_action(env, state)
    if env.config.control.mode == "motor_direct":
        return body_action_to_motor_action(env, state, body_action)
    return body_action


def motor_teacher_action(
    env: SatelliteEnv,
    state: EnvState,
    teacher_body_action: jax.Array | None = None,
) -> jax.Array:
    if teacher_body_action is None:
        teacher_body_action = normalized_pd_body_action(env, state)
    return body_action_to_motor_action(env, state, teacher_body_action)


def policy_to_command(
    env: SatelliteEnv,
    state: EnvState,
    policy_action: jax.Array,
) -> jax.Array:
    control = env.config.control
    if control.mode in ("direct", "motor_direct"):
        return jnp.clip(policy_action, -1.0, 1.0)
    if control.mode == "residual_pd":
        nominal = normalized_pd_body_action(env, state)
        return jnp.clip(
            nominal + control.residual_scale * policy_action,
            -1.0,
            1.0,
        )
    raise ValueError(f"Unsupported control mode: {control.mode}")
