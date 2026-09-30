
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import FDIRConfig, PhysicsConfig, SupervisorConfig
from .estimator import EstimatorState, attitude_sigma_rad
from .fdir import FAILED, FDIRState


DETUMBLE = 0
ATTITUDE_ACQUIRE = 1
FINE_POINTING = 2
DEGRADED_POINTING = 3
SAFE_SUN = 4
MOMENTUM_UNLOAD = 5
NUM_MODES = 6

MODE_NAMES = (
    "DETUMBLE",
    "ATTITUDE_ACQUIRE",
    "FINE_POINTING",
    "DEGRADED_POINTING",
    "SAFE_SUN",
    "MOMENTUM_UNLOAD",
)


class SupervisorState(NamedTuple):
    mode: jax.Array
    bad_dwell_steps: jax.Array
    good_dwell_steps: jax.Array
    mode_elapsed_steps: jax.Array


def _steps(seconds: float, physics: PhysicsConfig) -> int:
    return max(1, int(round(seconds / physics.control_dt)))


def reset_supervisor_state(
    estimator: EstimatorState,
    fdir: FDIRState,
    physics: PhysicsConfig,
    fdir_config: FDIRConfig,
    config: SupervisorConfig,
) -> SupervisorState:
    acquired = estimator.attitude_acquired
    sigma_deg = jnp.rad2deg(attitude_sigma_rad(estimator))
    enough_wheels = jnp.sum(fdir.wheel_authority_estimate > config.failed_wheel_authority, axis=-1) >= config.minimum_control_wheels
    fine = acquired & enough_wheels & (sigma_deg <= fdir_config.estimator_fine_sigma_deg)
    mode = jnp.where(fine, FINE_POINTING, ATTITUDE_ACQUIRE).astype(jnp.int32)
    zeros = jnp.zeros_like(mode)
    return SupervisorState(mode, zeros, zeros, zeros)


def update_supervisor(
    state: SupervisorState,
    estimator: EstimatorState,
    fdir: FDIRState,
    physics: PhysicsConfig,
    fdir_config: FDIRConfig,
    config: SupervisorConfig,
) -> SupervisorState:
    if not config.enabled:
        return state._replace(mode_elapsed_steps=state.mode_elapsed_steps + 1)

    sigma_deg = jnp.rad2deg(attitude_sigma_rad(estimator))
    acquired = estimator.attitude_acquired
    wheel_authority = fdir.wheel_authority_estimate
    available_wheels = jnp.sum(wheel_authority > config.failed_wheel_authority, axis=-1)
    enough_wheels = available_wheels >= config.minimum_control_wheels
    speed_fraction = 1.0 - fdir.wheel_speed_margin
    wheel_degraded = (
        jnp.any(wheel_authority < config.degraded_wheel_authority, axis=-1)
        | jnp.any(speed_fraction >= config.wheel_speed_degraded_fraction, axis=-1)
    )
    wheel_speed_critical = jnp.any(speed_fraction >= config.wheel_speed_critical_fraction, axis=-1)
    abs_sensor_failed = (
        (fdir.star_health == FAILED)
        | (fdir.magnetometer_health == FAILED)
        | (fdir.sun_health == FAILED)
    )
    gyro_failed = jnp.any(fdir.gyro_health == FAILED, axis=-1)

    estimator_lost = (~acquired) | (sigma_deg >= fdir_config.estimator_lost_sigma_deg)
    severe = acquired & (
        (~enough_wheels)
        | wheel_speed_critical
        | (sigma_deg >= fdir_config.estimator_degraded_sigma_deg)
    )
    degraded = (
        acquired
        & enough_wheels
        & (
            wheel_degraded
            | abs_sensor_failed
            | gyro_failed
            | (sigma_deg > fdir_config.estimator_fine_sigma_deg)
        )
    )
    fine = (
        acquired
        & enough_wheels
        & (~wheel_degraded)
        & (~gyro_failed)
        & (sigma_deg <= fdir_config.estimator_fine_sigma_deg)
    )

    enter_deg_steps = _steps(config.enter_degraded_dwell_seconds, physics)
    enter_safe_steps = _steps(config.enter_safe_dwell_seconds, physics)
    recover_steps = _steps(config.recovery_dwell_seconds, physics)
    acquire_steps = _steps(config.acquisition_dwell_seconds, physics)

    current = state.mode
    bad_condition = jnp.where(
        current == FINE_POINTING,
        degraded | severe | estimator_lost,
        jnp.where(current == DEGRADED_POINTING, severe | estimator_lost, estimator_lost),
    )
    good_condition = jnp.where(
        current == ATTITUDE_ACQUIRE,
        acquired & (fine | degraded),
        jnp.where(current == SAFE_SUN, acquired & enough_wheels & (sigma_deg < fdir_config.estimator_degraded_sigma_deg), fine),
    )
    bad_dwell = jnp.where(bad_condition, state.bad_dwell_steps + 1, 0)
    good_dwell = jnp.where(good_condition, state.good_dwell_steps + 1, 0)

    next_mode = current

    # Lost global attitude belongs in acquisition rather than pretending fine
    # pointing can continue from an unobservable state.
    next_mode = jnp.where(
        ((current == FINE_POINTING) | (current == DEGRADED_POINTING))
        & estimator_lost
        & (bad_dwell >= enter_safe_steps),
        ATTITUDE_ACQUIRE,
        next_mode,
    )
    next_mode = jnp.where(
        (current == FINE_POINTING) & severe & (bad_dwell >= enter_safe_steps),
        SAFE_SUN,
        next_mode,
    )
    next_mode = jnp.where(
        (current == FINE_POINTING) & degraded & (bad_dwell >= enter_deg_steps),
        DEGRADED_POINTING,
        next_mode,
    )
    next_mode = jnp.where(
        (current == DEGRADED_POINTING) & severe & (bad_dwell >= enter_safe_steps),
        SAFE_SUN,
        next_mode,
    )
    next_mode = jnp.where(
        (current == DEGRADED_POINTING) & fine & (good_dwell >= recover_steps),
        FINE_POINTING,
        next_mode,
    )
    next_mode = jnp.where(
        (current == ATTITUDE_ACQUIRE)
        & acquired
        & fine
        & (good_dwell >= acquire_steps),
        FINE_POINTING,
        next_mode,
    )
    next_mode = jnp.where(
        (current == ATTITUDE_ACQUIRE)
        & acquired
        & degraded
        & (good_dwell >= acquire_steps),
        DEGRADED_POINTING,
        next_mode,
    )
    next_mode = jnp.where(
        (current == SAFE_SUN)
        & acquired
        & enough_wheels
        & (sigma_deg < fdir_config.estimator_degraded_sigma_deg)
        & (good_dwell >= recover_steps),
        ATTITUDE_ACQUIRE,
        next_mode,
    )

    # DETUMBLE and MOMENTUM_UNLOAD are intentionally not entered automatically in
    # this milestone: genuine robust versions require external-torque actuators.
    changed = next_mode != current
    return SupervisorState(
        mode=next_mode.astype(jnp.int32),
        bad_dwell_steps=jnp.where(changed, 0, bad_dwell),
        good_dwell_steps=jnp.where(changed, 0, good_dwell),
        mode_elapsed_steps=jnp.where(changed, 0, state.mode_elapsed_steps + 1),
    )


def mode_one_hot(mode: jax.Array, dtype=jnp.float32) -> jax.Array:
    return jax.nn.one_hot(mode, NUM_MODES, dtype=dtype)
