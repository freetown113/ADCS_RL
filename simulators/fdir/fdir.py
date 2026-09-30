from typing import NamedTuple

import jax
import jax.numpy as jnp

from simulators.fdir.config import EstimatorConfig, FDIRConfig, PhysicsConfig, SensorConfig
from simulators.fdir.estimator import EstimatorState, attitude_sigma_rad
from simulators.fdir.physics import predicted_wheel_friction_torque
from simulators.fdir.sensors import SensorState, sensor_age_seconds


HEALTHY = 0
SUSPECT = 1
DEGRADED = 2
FAILED = 3
RECOVERING = 4

REASON_NONE = 0
REASON_STALE = 1 << 0
REASON_RANGE = 1 << 1
REASON_FROZEN = 1 << 2
REASON_RATE = 1 << 3
REASON_NIS = 1 << 4
REASON_RESPONSE = 1 << 5
REASON_SATURATION = 1 << 6


class FDIRState(NamedTuple):
    gyro_health: jax.Array              # [B,3]
    wheel_motor_health: jax.Array       # [B,4]
    wheel_tach_health: jax.Array        # [B,4]
    star_health: jax.Array              # [B]
    magnetometer_health: jax.Array      # [B]
    sun_health: jax.Array               # [B]
    gnss_health: jax.Array              # [B]

    wheel_authority_estimate: jax.Array # [B,4], controller belief only
    wheel_speed_margin: jax.Array       # [B,4], 1 at zero speed -> 0 at limit
    estimator_confidence: jax.Array     # [B], 0..1

    wheel_authority_ewma: jax.Array
    wheel_bad_count: jax.Array
    wheel_good_count: jax.Array
    wheel_tach_bad_count: jax.Array
    wheel_tach_good_count: jax.Array
    previous_wheel_speed: jax.Array
    previous_wheel_sample_step: jax.Array
    predicted_motor_torque: jax.Array

    star_nis_ewma: jax.Array
    mag_nis_ewma: jax.Array
    sun_nis_ewma: jax.Array
    star_bad_count: jax.Array
    mag_bad_count: jax.Array
    sun_bad_count: jax.Array
    star_good_count: jax.Array
    mag_good_count: jax.Array
    sun_good_count: jax.Array
    previous_star_sample_step: jax.Array
    previous_mag_sample_step: jax.Array
    previous_sun_sample_step: jax.Array

    gyro_reason: jax.Array
    wheel_motor_reason: jax.Array
    wheel_tach_reason: jax.Array
    star_reason: jax.Array
    magnetometer_reason: jax.Array
    sun_reason: jax.Array
    gnss_reason: jax.Array


def _condition_motor_command(command: jax.Array, physics: PhysicsConfig) -> jax.Array:
    q = jnp.asarray(physics.motor_command_quantization_torque, dtype=command.dtype)
    conditioned = jnp.where(q > 0.0, q * jnp.round(command / jnp.maximum(q, 1.0e-12)), command)
    conditioned = jnp.where(
        jnp.abs(conditioned) < physics.motor_dead_zone_torque,
        jnp.zeros_like(conditioned),
        conditioned,
    )
    return jnp.clip(conditioned, -physics.max_motor_torque, physics.max_motor_torque)


def _estimator_confidence(estimator: EstimatorState, config: FDIRConfig) -> jax.Array:
    sigma = jnp.rad2deg(attitude_sigma_rad(estimator))
    scale = jnp.maximum(jnp.asarray(config.estimator_lost_sigma_deg, sigma.dtype), 1.0e-6)
    confidence = jnp.clip(1.0 - sigma / scale, 0.0, 1.0)
    return jnp.where(estimator.attitude_acquired, confidence, 0.0)


def reset_fdir_state(
    sensors: SensorState,
    estimator: EstimatorState,
    config: FDIRConfig,
) -> FDIRState:
    batch = sensors.gyro.shape[0]
    h3 = jnp.full((batch, 3), HEALTHY, dtype=jnp.int32)
    h4 = jnp.full((batch, 4), HEALTHY, dtype=jnp.int32)
    hs = jnp.full((batch,), HEALTHY, dtype=jnp.int32)
    z4i = jnp.zeros((batch, 4), dtype=jnp.int32)
    zsi = jnp.zeros((batch,), dtype=jnp.int32)
    z4 = jnp.zeros((batch, 4), dtype=sensors.wheel_speed.dtype)
    zs = jnp.zeros((batch,), dtype=sensors.gyro.dtype)
    return FDIRState(
        gyro_health=h3,
        wheel_motor_health=h4,
        wheel_tach_health=h4,
        star_health=hs,
        magnetometer_health=hs,
        sun_health=hs,
        gnss_health=hs,
        wheel_authority_estimate=jnp.ones((batch, 4), dtype=sensors.wheel_speed.dtype),
        wheel_speed_margin=jnp.ones((batch, 4), dtype=sensors.wheel_speed.dtype),
        estimator_confidence=_estimator_confidence(estimator, config),
        wheel_authority_ewma=jnp.ones((batch, 4), dtype=sensors.wheel_speed.dtype),
        wheel_bad_count=z4i,
        wheel_good_count=z4i,
        wheel_tach_bad_count=z4i,
        wheel_tach_good_count=z4i,
        previous_wheel_speed=sensors.wheel_speed,
        previous_wheel_sample_step=sensors.wheel_sample_step,
        predicted_motor_torque=z4,
        star_nis_ewma=zs,
        mag_nis_ewma=zs,
        sun_nis_ewma=zs,
        star_bad_count=zsi,
        mag_bad_count=zsi,
        sun_bad_count=zsi,
        star_good_count=zsi,
        mag_good_count=zsi,
        sun_good_count=zsi,
        previous_star_sample_step=sensors.star_tracker_sample_step,
        previous_mag_sample_step=sensors.magnetometer_sample_step,
        previous_sun_sample_step=sensors.sun_sensor_sample_step,
        gyro_reason=jnp.zeros((batch, 3), dtype=jnp.int32),
        wheel_motor_reason=jnp.zeros((batch, 4), dtype=jnp.int32),
        wheel_tach_reason=jnp.zeros((batch, 4), dtype=jnp.int32),
        star_reason=zsi,
        magnetometer_reason=zsi,
        sun_reason=zsi,
        gnss_reason=zsi,
    )


def _persistent_health(
    health: jax.Array,
    bad: jax.Array,
    good: jax.Array,
    bad_count: jax.Array,
    good_count: jax.Array,
    suspect_samples: int,
    fail_samples: int,
    recovery_samples: int,
    terminal_health: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    bad_count_next = jnp.where(bad, bad_count + 1, jnp.where(good, 0, bad_count))
    good_count_next = jnp.where(good, good_count + 1, jnp.where(bad, 0, good_count))

    next_health = health
    next_health = jnp.where(
        (health == HEALTHY) & (bad_count_next >= suspect_samples), SUSPECT, next_health
    )
    next_health = jnp.where(
        ((health == HEALTHY) | (health == SUSPECT)) & (bad_count_next >= fail_samples),
        terminal_health,
        next_health,
    )
    # A partially degraded channel may continue worsening into a hard failure.
    next_health = jnp.where(
        (health == DEGRADED)
        & (terminal_health == FAILED)
        & (bad_count_next >= fail_samples),
        FAILED,
        next_health,
    )
    enter_recovery = (
        ((health == SUSPECT) | (health == DEGRADED) | (health == FAILED))
        & (good_count_next >= recovery_samples)
    )
    next_health = jnp.where(enter_recovery, RECOVERING, next_health)
    # Require another full recovery dwell before returning healthy.
    good_count_next = jnp.where(enter_recovery, 0, good_count_next)
    next_health = jnp.where(
        (health == RECOVERING) & bad, FAILED, next_health
    )
    next_health = jnp.where(
        (health == RECOVERING) & (good_count_next >= recovery_samples), HEALTHY, next_health
    )
    return next_health.astype(jnp.int32), bad_count_next, good_count_next


def _watchdog_health(
    current: jax.Array,
    age: jax.Array,
    stale_seconds: float,
    recovery_samples: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    stale = age > stale_seconds
    good = jnp.isfinite(age) & (~stale)
    bad_count = jnp.where(stale, jnp.ones_like(current), jnp.zeros_like(current))
    good_count = jnp.where(good, jnp.full_like(current, recovery_samples), jnp.zeros_like(current))
    terminal = jnp.full_like(current, FAILED)
    # Watchdog threshold already includes multiple nominal sample periods, so stale
    # can directly fail; recovery still passes through RECOVERING.
    health = jnp.where(stale, FAILED, current)
    health = jnp.where((current == FAILED) & good, RECOVERING, health)
    health = jnp.where((current == RECOVERING) & good, HEALTHY, health)
    return health.astype(jnp.int32), bad_count, good_count


def filter_sensors_for_estimator(sensors: SensorState, state: FDIRState) -> SensorState:
    """Reject only measurements already isolated as failed.
    Environmental unavailability (eclipse, star-lock geometry) remains represented by
    the sensor's own validity flag and is not turned into a hardware failure here.
    """
    return sensors._replace(
        star_tracker_valid=sensors.star_tracker_valid & (state.star_health != FAILED),
        magnetometer_valid=sensors.magnetometer_valid & (state.magnetometer_health != FAILED),
        sun_sensor_valid=sensors.sun_sensor_valid & (state.sun_health != FAILED),
        gnss_valid=sensors.gnss_valid & (state.gnss_health != FAILED),
    )


def update_fdir(
    state: FDIRState,
    sensors: SensorState,
    estimator: EstimatorState,
    commanded_motor_torque: jax.Array,
    absolute_physics_step: jax.Array,
    physics: PhysicsConfig,
    sensor_config: SensorConfig,
    estimator_config: EstimatorConfig,
    config: FDIRConfig,
) -> FDIRState:
    if not config.enabled:
        return state._replace(estimator_confidence=_estimator_confidence(estimator, config))

    dtype = sensors.wheel_speed.dtype
    # --- Reaction-wheel motor authority ---------------------------------------
    conditioned = _condition_motor_command(commanded_motor_torque, physics)
    nominal_target = conditioned * jnp.asarray(physics.motor_torque_scale, dtype=dtype)
    alpha_motor = 1.0 - jnp.exp(-physics.physics_dt / physics.motor_time_constant_s)
    predicted_motor = state.predicted_motor_torque + alpha_motor * (nominal_target - state.predicted_motor_torque)

    new_wheel_sample = sensors.wheel_sample_step > state.previous_wheel_sample_step
    sample_dt = (sensors.wheel_sample_step - state.previous_wheel_sample_step).astype(dtype) * physics.physics_dt
    sample_dt = jnp.maximum(sample_dt, physics.physics_dt)
    measured_accel = (sensors.wheel_speed - state.previous_wheel_speed) / sample_dt
    predicted_friction = predicted_wheel_friction_torque(sensors.wheel_speed, physics)
    observed_motor_equivalent = physics.wheel_inertia * measured_accel + predicted_friction
    authority_sample = (observed_motor_equivalent * predicted_motor) / (predicted_motor**2 + 1.0e-8)
    authority_sample = jnp.clip(authority_sample, 0.0, 1.10)

    excitation = (
        new_wheel_sample
        & sensors.wheel_valid
        & (jnp.abs(predicted_motor) >= config.wheel_min_excitation_torque)
        & (jnp.abs(sensors.wheel_speed) <= config.wheel_speed_monitor_fraction * physics.max_wheel_speed)
        & (state.wheel_tach_health != FAILED)
    )
    authority_ewma = jnp.where(
        excitation,
        (1.0 - config.wheel_authority_ewma_alpha) * state.wheel_authority_ewma
        + config.wheel_authority_ewma_alpha * authority_sample,
        state.wheel_authority_ewma,
    )
    authority_est = jnp.clip(authority_ewma, 0.0, 1.0)
    wheel_bad = excitation & (authority_est < config.wheel_degraded_authority)
    wheel_good = excitation & (authority_est >= config.wheel_recovery_authority)
    wheel_terminal = jnp.where(
        authority_est <= config.wheel_failed_authority,
        jnp.full_like(state.wheel_motor_health, FAILED),
        jnp.full_like(state.wheel_motor_health, DEGRADED),
    )
    wheel_health, wheel_bad_count, wheel_good_count = _persistent_health(
        state.wheel_motor_health,
        wheel_bad,
        wheel_good,
        state.wheel_bad_count,
        state.wheel_good_count,
        config.wheel_suspect_samples,
        config.wheel_fail_samples,
        config.wheel_recovery_samples,
        wheel_terminal,
    )
    wheel_reason = jnp.where(
        wheel_bad,
        state.wheel_motor_reason | REASON_RESPONSE,
        jnp.where(wheel_health == HEALTHY, REASON_NONE, state.wheel_motor_reason),
    )
    saturation = jnp.abs(sensors.wheel_speed) > config.wheel_speed_monitor_fraction * physics.max_wheel_speed
    wheel_reason = jnp.where(saturation, wheel_reason | REASON_SATURATION, wheel_reason)

    # --- Tachometer watchdog.  Motor-vs-encoder isolation is intentionally not
    # inferred from the motor residual alone; independent tach faults can be added
    # to the simulator without contaminating the motor-health state.
    ages = sensor_age_seconds(sensors, absolute_physics_step, physics)
    wheel_stale = ages.wheel > (config.fast_sensor_stale_periods / sensor_config.wheel_tach_rate_hz)
    tach_health = jnp.where(wheel_stale, FAILED, state.wheel_tach_health)
    tach_health = jnp.where((state.wheel_tach_health == FAILED) & (~wheel_stale), RECOVERING, tach_health)
    tach_health = jnp.where((state.wheel_tach_health == RECOVERING) & (~wheel_stale), HEALTHY, tach_health).astype(jnp.int32)
    tach_bad_count = jnp.where(wheel_stale, state.wheel_tach_bad_count + 1, 0)
    tach_good_count = jnp.where(~wheel_stale, state.wheel_tach_good_count + 1, 0)
    tach_reason = jnp.where(wheel_stale, state.wheel_tach_reason | REASON_STALE, jnp.where(tach_health == HEALTHY, 0, state.wheel_tach_reason))

    # --- Packet watchdogs ------------------------------------------------------
    gyro_stale = ages.gyro > (config.fast_sensor_stale_periods / sensor_config.gyro_rate_hz)
    gyro_bad_values = ~jnp.all(jnp.isfinite(sensors.gyro), axis=-1)
    gyro_failed = gyro_stale | gyro_bad_values
    gyro_health = jnp.where(gyro_failed[:, None], FAILED, state.gyro_health)
    gyro_health = jnp.where((state.gyro_health == FAILED) & (~gyro_failed[:, None]), RECOVERING, gyro_health)
    gyro_health = jnp.where((state.gyro_health == RECOVERING) & (~gyro_failed[:, None]), HEALTHY, gyro_health).astype(jnp.int32)
    gyro_reason_scalar = jnp.where(gyro_stale, REASON_STALE, 0) | jnp.where(gyro_bad_values, REASON_RANGE, 0)
    gyro_reason = jnp.where(gyro_failed[:, None], state.gyro_reason | gyro_reason_scalar[:, None], jnp.where(gyro_health == HEALTHY, 0, state.gyro_reason))

    gnss_stale = ages.gnss > (config.slow_sensor_stale_periods / sensor_config.gnss_rate_hz)
    gnss_health = jnp.where(gnss_stale, FAILED, state.gnss_health)
    gnss_health = jnp.where((state.gnss_health == FAILED) & (~gnss_stale), RECOVERING, gnss_health)
    gnss_health = jnp.where((state.gnss_health == RECOVERING) & (~gnss_stale), HEALTHY, gnss_health).astype(jnp.int32)
    gnss_reason = jnp.where(gnss_stale, state.gnss_reason | REASON_STALE, jnp.where(gnss_health == HEALTHY, 0, state.gnss_reason))

    # --- MEKF innovation health -----------------------------------------------
    def sensor_innovation_update(
        health, nis_ewma, bad_count, good_count, reason,
        nis, gate, processed, accepted, measurement_valid, age, rate_hz,
    ):
        alpha = config.nis_ewma_alpha
        stale = age > (config.slow_sensor_stale_periods / rate_hz)
        # FAILED measurements were blocked before this estimator update. A fresh
        # packet therefore starts a probation state; subsequent RECOVERING packets
        # are admitted and judged by the MEKF innovation gate.
        tested = processed & measurement_valid & (health != FAILED)
        probe_ready = processed & measurement_valid & (health == FAILED) & (~stale)
        nis_ewma_next = jnp.where(tested, (1.0 - alpha) * nis_ewma + alpha * nis, nis_ewma)
        innovation_bad = tested & ((~accepted) | (nis_ewma_next > config.nis_suspect_ratio * gate))
        innovation_good = tested & accepted & (nis_ewma_next <= config.nis_suspect_ratio * gate)
        terminal = jnp.full_like(health, FAILED)
        health_next, bad_next, good_next = _persistent_health(
            health, innovation_bad, innovation_good, bad_count, good_count,
            config.innovation_suspect_samples, config.innovation_fail_samples,
            config.sensor_recovery_samples, terminal,
        )
        health_next = jnp.where(probe_ready, RECOVERING, health_next)
        bad_next = jnp.where(probe_ready, 0, bad_next)
        good_next = jnp.where(probe_ready, 0, good_next)
        health_next = jnp.where(stale, FAILED, health_next)
        reason_next = jnp.where(innovation_bad, reason | REASON_NIS, reason)
        reason_next = jnp.where(stale, reason_next | REASON_STALE, reason_next)
        reason_next = jnp.where(health_next == HEALTHY, 0, reason_next)
        return health_next.astype(jnp.int32), nis_ewma_next, bad_next, good_next, reason_next

    star_processed = sensors.star_tracker_sample_step > state.previous_star_sample_step
    mag_processed = sensors.magnetometer_sample_step > state.previous_mag_sample_step
    sun_processed = sensors.sun_sensor_sample_step > state.previous_sun_sample_step

    star_health, star_ewma, star_bad, star_good, star_reason = sensor_innovation_update(
        state.star_health, state.star_nis_ewma, state.star_bad_count, state.star_good_count,
        state.star_reason, estimator.star_tracker_nis, estimator_config.star_tracker_nis_gate,
        star_processed, estimator.star_tracker_update_accepted, sensors.star_tracker_valid,
        ages.star_tracker, sensor_config.star_tracker_rate_hz,
    )
    mag_health, mag_ewma, mag_bad, mag_good, mag_reason = sensor_innovation_update(
        state.magnetometer_health, state.mag_nis_ewma, state.mag_bad_count, state.mag_good_count,
        state.magnetometer_reason, estimator.magnetometer_nis, estimator_config.magnetometer_nis_gate,
        mag_processed, estimator.magnetometer_update_accepted, sensors.magnetometer_valid,
        ages.magnetometer, sensor_config.magnetometer_rate_hz,
    )
    sun_health, sun_ewma, sun_bad, sun_good, sun_reason = sensor_innovation_update(
        state.sun_health, state.sun_nis_ewma, state.sun_bad_count, state.sun_good_count,
        state.sun_reason, estimator.sun_sensor_nis, estimator_config.sun_sensor_nis_gate,
        sun_processed, estimator.sun_sensor_update_accepted, sensors.sun_sensor_valid,
        ages.sun_sensor, sensor_config.sun_sensor_rate_hz,
    )

    return FDIRState(
        gyro_health=gyro_health,
        wheel_motor_health=wheel_health,
        wheel_tach_health=tach_health,
        star_health=star_health,
        magnetometer_health=mag_health,
        sun_health=sun_health,
        gnss_health=gnss_health,
        wheel_authority_estimate=authority_est,
        wheel_speed_margin=jnp.clip(1.0 - jnp.abs(sensors.wheel_speed) / physics.max_wheel_speed, 0.0, 1.0),
        estimator_confidence=_estimator_confidence(estimator, config),
        wheel_authority_ewma=authority_ewma,
        wheel_bad_count=wheel_bad_count,
        wheel_good_count=wheel_good_count,
        wheel_tach_bad_count=tach_bad_count,
        wheel_tach_good_count=tach_good_count,
        previous_wheel_speed=jnp.where(new_wheel_sample, sensors.wheel_speed, state.previous_wheel_speed),
        previous_wheel_sample_step=jnp.where(new_wheel_sample, sensors.wheel_sample_step, state.previous_wheel_sample_step),
        predicted_motor_torque=predicted_motor,
        star_nis_ewma=star_ewma,
        mag_nis_ewma=mag_ewma,
        sun_nis_ewma=sun_ewma,
        star_bad_count=star_bad,
        mag_bad_count=mag_bad,
        sun_bad_count=sun_bad,
        star_good_count=star_good,
        mag_good_count=mag_good,
        sun_good_count=sun_good,
        previous_star_sample_step=jnp.where(star_processed, sensors.star_tracker_sample_step, state.previous_star_sample_step),
        previous_mag_sample_step=jnp.where(mag_processed, sensors.magnetometer_sample_step, state.previous_mag_sample_step),
        previous_sun_sample_step=jnp.where(sun_processed, sensors.sun_sensor_sample_step, state.previous_sun_sample_step),
        gyro_reason=gyro_reason,
        wheel_motor_reason=wheel_reason,
        wheel_tach_reason=tach_reason,
        star_reason=star_reason,
        magnetometer_reason=mag_reason,
        sun_reason=sun_reason,
        gnss_reason=gnss_reason,
    )
