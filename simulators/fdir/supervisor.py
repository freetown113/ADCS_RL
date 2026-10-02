from typing import NamedTuple

import jax
import jax.numpy as jnp

from simulators.fdir.config import FDIRConfig, PhysicsConfig, SupervisorConfig
from simulators.fdir.estimator import EstimatorState, attitude_sigma_rad
from simulators.fdir.fdir import FAILED, FDIRState


DETUMBLE = 0
ATTITUDE_ACQUIRE = 1
FINE_POINTING = 2
DEGRADED_POINTING = 3
SAFE_SUN = 4
MOMENTUM_UNLOAD = 5
NUM_MODES = 6
MODE_NAMES = (
    "DETUMBLE", "ATTITUDE_ACQUIRE", "FINE_POINTING",
    "DEGRADED_POINTING", "SAFE_SUN", "MOMENTUM_UNLOAD",
)


class SupervisorState(NamedTuple):
    mode: jax.Array
    bad_dwell_steps: jax.Array
    good_dwell_steps: jax.Array
    detumble_high_dwell_steps: jax.Array
    detumble_low_dwell_steps: jax.Array
    momentum_high_dwell_steps: jax.Array
    momentum_low_dwell_steps: jax.Array
    mode_elapsed_steps: jax.Array
    resume_mode: jax.Array


def _steps(seconds: float, physics: PhysicsConfig) -> int:
    return max(1, int(round(seconds / physics.control_dt)))


def _pointing_conditions(estimator, fdir, fdir_config, config):
    sigma_deg = jnp.rad2deg(attitude_sigma_rad(estimator))
    acquired = estimator.attitude_acquired
    authority = fdir.wheel_authority_estimate
    available = jnp.sum(authority > config.failed_wheel_authority, axis=-1)
    enough = available >= config.minimum_control_wheels
    speed_fraction = 1.0 - fdir.wheel_speed_margin
    degraded_wheel = (
        jnp.any(authority < config.degraded_wheel_authority, axis=-1)
        | jnp.any(speed_fraction >= config.wheel_speed_degraded_fraction, axis=-1)
    )
    critical_speed = jnp.any(speed_fraction >= config.wheel_speed_critical_fraction, axis=-1)
    abs_sensor_failed = (
        (fdir.star_health == FAILED)
        | (fdir.magnetometer_health == FAILED)
        | (fdir.sun_health == FAILED)
    )
    gyro_failed = jnp.any(fdir.gyro_health == FAILED, axis=-1)
    lost = (~acquired) | (sigma_deg >= fdir_config.estimator_lost_sigma_deg)
    severe = acquired & ((~enough) | critical_speed | (sigma_deg >= fdir_config.estimator_degraded_sigma_deg))
    degraded = acquired & enough & (
        degraded_wheel | abs_sensor_failed | gyro_failed | (sigma_deg > fdir_config.estimator_fine_sigma_deg)
    )
    fine = acquired & enough & (~degraded_wheel) & (~gyro_failed) & (sigma_deg <= fdir_config.estimator_fine_sigma_deg)
    return sigma_deg, acquired, enough, fine, degraded, severe, lost, speed_fraction


def reset_supervisor_state(
    estimator: EstimatorState,
    fdir: FDIRState,
    physics: PhysicsConfig,
    fdir_config: FDIRConfig,
    config: SupervisorConfig,
) -> SupervisorState:
    _, acquired, _, fine, _, _, _, _ = _pointing_conditions(estimator, fdir, fdir_config, config)
    rate_deg_s = jnp.rad2deg(jnp.linalg.norm(estimator.omega, axis=-1))
    detumble = config.autonomous_detumble_enabled & (rate_deg_s >= config.detumble_enter_rate_deg_s)
    mode = jnp.where(detumble, DETUMBLE, jnp.where(fine, FINE_POINTING, ATTITUDE_ACQUIRE)).astype(jnp.int32)
    z = jnp.zeros_like(mode)
    return SupervisorState(mode, z, z, z, z, z, z, z, jnp.where(fine, FINE_POINTING, ATTITUDE_ACQUIRE).astype(jnp.int32))


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

    sigma_deg, acquired, enough, fine, degraded, severe, lost, speed_fraction = _pointing_conditions(
        estimator, fdir, fdir_config, config
    )
    rate_deg_s = jnp.rad2deg(jnp.linalg.norm(estimator.omega, axis=-1))
    max_speed_fraction = jnp.max(speed_fraction, axis=-1)

    det_high = config.autonomous_detumble_enabled & (rate_deg_s >= config.detumble_enter_rate_deg_s)
    det_low = rate_deg_s <= config.detumble_exit_rate_deg_s
    mom_high = config.autonomous_momentum_unload_enabled & (max_speed_fraction >= config.momentum_unload_enter_fraction)
    mom_low = max_speed_fraction <= config.momentum_unload_exit_fraction

    dh = jnp.where(det_high, state.detumble_high_dwell_steps + 1, 0)
    dl = jnp.where(det_low, state.detumble_low_dwell_steps + 1, 0)
    mh = jnp.where(mom_high, state.momentum_high_dwell_steps + 1, 0)
    ml = jnp.where(mom_low, state.momentum_low_dwell_steps + 1, 0)

    enter_det = _steps(config.detumble_enter_dwell_seconds, physics)
    exit_det = _steps(config.detumble_exit_dwell_seconds, physics)
    enter_mom = _steps(config.momentum_unload_enter_dwell_seconds, physics)
    exit_mom = _steps(config.momentum_unload_exit_dwell_seconds, physics)
    enter_deg = _steps(config.enter_degraded_dwell_seconds, physics)
    enter_safe = _steps(config.enter_safe_dwell_seconds, physics)
    recover = _steps(config.recovery_dwell_seconds, physics)
    acquire = _steps(config.acquisition_dwell_seconds, physics)

    current = state.mode
    bad_condition = jnp.where(
        current == FINE_POINTING,
        degraded | severe | lost,
        jnp.where(current == DEGRADED_POINTING, severe | lost, lost),
    )
    good_condition = jnp.where(
        current == ATTITUDE_ACQUIRE,
        acquired & (fine | degraded),
        jnp.where(current == SAFE_SUN, acquired & enough & (sigma_deg < fdir_config.estimator_degraded_sigma_deg), fine),
    )
    bad = jnp.where(bad_condition, state.bad_dwell_steps + 1, 0)
    good = jnp.where(good_condition, state.good_dwell_steps + 1, 0)

    next_mode = current
    resume_mode = state.resume_mode

    # Highest-priority dynamic protection: high-rate detumble when explicitly enabled.
    may_enter_det = (current != DETUMBLE) & (current != SAFE_SUN) & (dh >= enter_det)
    resume_mode = jnp.where(may_enter_det, current, resume_mode)
    next_mode = jnp.where(may_enter_det, DETUMBLE, next_mode)
    leave_det = (current == DETUMBLE) & (dl >= exit_det)
    next_mode = jnp.where(leave_det, ATTITUDE_ACQUIRE, next_mode)

    # Severe loss overrides pointing/unloading.
    next_mode = jnp.where(
        ((current == FINE_POINTING) | (current == DEGRADED_POINTING) | (current == MOMENTUM_UNLOAD))
        & severe & (bad >= enter_safe),
        SAFE_SUN,
        next_mode,
    )
    next_mode = jnp.where(
        ((current == FINE_POINTING) | (current == DEGRADED_POINTING) | (current == MOMENTUM_UNLOAD))
        & lost & (bad >= enter_safe),
        ATTITUDE_ACQUIRE,
        next_mode,
    )

    # Momentum management may coexist with a mission reference
    may_enter_mom = (
        ((current == FINE_POINTING) | (current == DEGRADED_POINTING))
        & (mh >= enter_mom)
        & (~severe) & (~lost)
    )
    resume_mode = jnp.where(may_enter_mom, current, resume_mode)
    next_mode = jnp.where(may_enter_mom, MOMENTUM_UNLOAD, next_mode)
    leave_mom = (current == MOMENTUM_UNLOAD) & (ml >= exit_mom)
    resume_after = jnp.where(fine, FINE_POINTING, jnp.where(degraded, DEGRADED_POINTING, ATTITUDE_ACQUIRE))
    next_mode = jnp.where(leave_mom, resume_after, next_mode)

    # Nominal health transitions.
    next_mode = jnp.where((current == FINE_POINTING) & degraded & (bad >= enter_deg), DEGRADED_POINTING, next_mode)
    next_mode = jnp.where((current == DEGRADED_POINTING) & fine & (good >= recover), FINE_POINTING, next_mode)
    next_mode = jnp.where((current == ATTITUDE_ACQUIRE) & fine & (good >= acquire), FINE_POINTING, next_mode)
    next_mode = jnp.where((current == ATTITUDE_ACQUIRE) & degraded & (good >= acquire), DEGRADED_POINTING, next_mode)
    next_mode = jnp.where(
        (current == SAFE_SUN) & acquired & enough
        & (sigma_deg < fdir_config.estimator_degraded_sigma_deg)
        & (good >= recover),
        ATTITUDE_ACQUIRE,
        next_mode,
    )

    changed = next_mode != current
    return SupervisorState(
        mode=next_mode.astype(jnp.int32),
        bad_dwell_steps=jnp.where(changed, 0, bad),
        good_dwell_steps=jnp.where(changed, 0, good),
        detumble_high_dwell_steps=jnp.where(changed, 0, dh),
        detumble_low_dwell_steps=jnp.where(changed, 0, dl),
        momentum_high_dwell_steps=jnp.where(changed, 0, mh),
        momentum_low_dwell_steps=jnp.where(changed, 0, ml),
        mode_elapsed_steps=jnp.where(changed, 0, state.mode_elapsed_steps + 1),
        resume_mode=resume_mode.astype(jnp.int32),
    )


def mode_one_hot(mode: jax.Array, dtype=jnp.float32) -> jax.Array:
    return jax.nn.one_hot(mode, NUM_MODES, dtype=dtype)
