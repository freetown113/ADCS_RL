from dataclasses import asdict, replace

import jax
import jax.numpy as jnp

from simulators.fdir.config import config_from_dict, default_config
from simulators.fdir.env import SatelliteEnv
from simulators.fdir.math3d import attitude_angle, attitude_error, rotate_body_to_inertial, rotate_inertial_to_body
from simulators.fdir.guidance import guidance_target, tracking_rate_error_body
from simulators.fdir.physics import motor_torque_to_net_rotor_torque


def _clean_sensors(s):
    return replace(
        s,
        gyro_noise_density_rad_s_sqrt_hz=0.0,
        gyro_initial_bias_std_rad_s=0.0,
        gyro_bias_random_walk_rad_s_per_sqrt_s=0.0,
        gyro_scale_factor_std=0.0,
        gyro_misalignment_std_deg=0.0,
        gyro_quantization_rad_s=0.0,
        gyro_packet_loss_probability=0.0,
        wheel_tach_noise_std_rad_s=0.0,
        wheel_tach_scale_factor_std=0.0,
        wheel_tach_quantization_rad_s=0.0,
        wheel_tach_packet_loss_probability=0.0,
        star_tracker_noise_std_deg=0.0,
        star_tracker_alignment_std_deg=0.0,
        star_tracker_packet_loss_probability=0.0,
        star_tracker_outlier_probability=0.0,
        star_tracker_initially_locked=True,
        magnetometer_noise_std_t=0.0,
        magnetometer_bias_std_t=0.0,
        magnetometer_scale_factor_std=0.0,
        magnetometer_misalignment_std_deg=0.0,
        magnetometer_quantization_t=0.0,
        magnetometer_packet_loss_probability=0.0,
        sun_sensor_noise_std_deg=0.0,
        sun_sensor_misalignment_std_deg=0.0,
        sun_sensor_packet_loss_probability=0.0,
        gnss_position_noise_std_m=0.0,
        gnss_velocity_noise_std_m_s=0.0,
        gnss_packet_loss_probability=0.0,
        sun_sensor_eclipse_enabled=False,
    )


def _error_deg(state):
    return jnp.rad2deg(attitude_angle(attitude_error(state.estimator.q, state.physical.q)))


def test_legacy_checkpoint_loads():
    data = asdict(default_config())
    data["estimator"]["compensate_fixed_latency"] = True
    restored = config_from_dict(data)
    assert restored.estimator.use_fixed_lag_replay


def test_fixed_lag_replays_delayed_star_solution():
    cfg = default_config()
    sensors = _clean_sensors(cfg.sensors)
    sensors = replace(
        sensors,
        gyro_rate_hz=100.0,
        gyro_latency_seconds=0.0,
        star_tracker_rate_hz=2.0,
        star_tracker_latency_seconds=0.08,
        magnetometer_rate_hz=10.0,
        magnetometer_latency_seconds=0.02,
        sun_sensor_rate_hz=10.0,
        sun_sensor_latency_seconds=0.01,
        gnss_rate_hz=1.0,
        gnss_latency_seconds=0.10,
    )
    cfg = replace(
        cfg,
        sensors=sensors,
        control=replace(cfg.control, mode="motor_direct"),
        task=replace(cfg.task, fixed_angle_deg=120.0, max_initial_rate=0.1),
    )
    env = SatelliteEnv(cfg)
    state = env.reset(jax.random.PRNGKey(100), 1)
    step = jax.jit(env.step)
    zero = jnp.zeros((1, 4), jnp.float32)
    for _ in range(4):
        state, _, _, _ = step(state, zero)
    if float(_error_deg(state)[0]) > 0.1:
        raise AssertionError(f"fixed-lag acquisition/repropagation error: {_error_deg(state)}")


def test_fixed_lag_handles_gyro_transport_delay():
    cfg = default_config()
    sensors = _clean_sensors(cfg.sensors)
    sensors = replace(
        sensors,
        gyro_rate_hz=20.0,
        gyro_latency_seconds=0.02,
        star_tracker_rate_hz=2.0,
        star_tracker_latency_seconds=0.08,
        magnetometer_latency_seconds=0.02,
        sun_sensor_latency_seconds=0.01,
        gnss_latency_seconds=0.10,
    )
    cfg = replace(
        cfg,
        sensors=sensors,
        control=replace(cfg.control, mode="motor_direct"),
        task=replace(cfg.task, fixed_angle_deg=60.0, max_initial_rate=0.1),
    )
    env = SatelliteEnv(cfg)
    state = env.reset(jax.random.PRNGKey(101), 1)
    step = jax.jit(env.step)
    zero = jnp.zeros((1, 4), jnp.float32)
    for _ in range(12):
        state, _, _, _ = step(state, zero)
    if float(_error_deg(state)[0]) > 0.2:
        raise AssertionError(f"delayed gyro replay error: {_error_deg(state)}")


def test_sensor_model_is_not_ground_truth():
    cfg = default_config()
    env = SatelliteEnv(cfg)
    state = env.reset(jax.random.PRNGKey(102), 16)
    gyro_difference = jnp.linalg.norm(state.sensors.gyro - state.physical.omega, axis=-1)
    mag_bias = jnp.linalg.norm(state.sensors.magnetometer_bias_true_t, axis=-1)
    if not bool(jnp.any(gyro_difference > 1.0e-6)):
        raise AssertionError("gyro model unexpectedly equals truth")
    if not bool(jnp.any(mag_bias > 0.0)):
        raise AssertionError("magnetometer calibration bias was not sampled")


def test_star_tracker_loses_lock_above_tracking_rate():
    cfg = default_config()
    cfg = replace(
        cfg,
        sensors=replace(
            _clean_sensors(cfg.sensors),
            star_tracker_initially_locked=True,
            star_tracker_max_tracking_rate_deg_s=3.0,
        ),
        task=replace(cfg.task, max_initial_rate=0.0),
    )
    env = SatelliteEnv(cfg)
    state = env.reset(jax.random.PRNGKey(103), 1)
    state = state._replace(
        physical=state.physical._replace(
            omega=jnp.asarray([[0.0, 0.0, jnp.deg2rad(10.0)]], dtype=jnp.float32)
        )
    )
    state, _, _, _ = jax.jit(env.step)(state, jnp.zeros((1, env.action_size), jnp.float32))
    if bool(state.sensors.star_tracker_locked[0]):
        raise AssertionError("star tracker stayed locked above configured tracking rate")


def test_motor_lag_deadzone_and_degradation():
    cfg = default_config().physics
    speed = jnp.zeros((1, 4), jnp.float32)
    previous = jnp.zeros((1, 4), jnp.float32)
    small = jnp.full((1, 4), 0.5 * cfg.motor_dead_zone_torque, jnp.float32)
    motor, net = motor_torque_to_net_rotor_torque(small, speed, jnp.ones((1, 4)), cfg, previous)
    if float(jnp.max(jnp.abs(net))) > 1.0e-8:
        raise AssertionError("dead-zone/stiction did not suppress tiny motor command")

    command = jnp.full((1, 4), 0.02, jnp.float32)
    motor_full, _ = motor_torque_to_net_rotor_torque(command, speed, jnp.ones((1, 4)), cfg, previous)
    motor_half, _ = motor_torque_to_net_rotor_torque(command, speed, 0.5 * jnp.ones((1, 4)), cfg, previous)
    if not float(jnp.mean(motor_full)) < 0.02:
        raise AssertionError("motor lag was not applied")
    ratio = float(jnp.mean(motor_half) / jnp.mean(motor_full))
    if not 0.45 < ratio < 0.55:
        raise AssertionError(f"partial wheel authority ratio incorrect: {ratio}")


def test_covariance_stays_well_formed():
    env = SatelliteEnv(default_config())
    state = env.reset(jax.random.PRNGKey(104), 4)
    state, _, _, _ = jax.jit(env.step)(
        state, jnp.zeros((4, env.action_size), jnp.float32)
    )
    p = state.estimator.covariance
    if not bool(jnp.all(jnp.isfinite(p))):
        raise AssertionError("non-finite estimator covariance")
    if not bool(jnp.allclose(p, jnp.swapaxes(p, -1, -2), atol=1e-6, rtol=0.0)):
        raise AssertionError("estimator covariance lost symmetry")
    if not bool(jnp.all(jnp.diagonal(p, axis1=-2, axis2=-1) > 0.0)):
        raise AssertionError("estimator covariance has non-positive diagonal")



def test_inertial_guidance_preserves_legacy_reference():
    cfg = default_config()
    env = SatelliteEnv(cfg)
    state = env.reset(jax.random.PRNGKey(105), 3)
    expected_q = jnp.tile(jnp.asarray([1.0, 0.0, 0.0, 0.0], jnp.float32), (3, 1))
    if not bool(jnp.allclose(state.target_q, expected_q, atol=1e-7, rtol=0.0)):
        raise AssertionError(f"inertial guidance target changed: {state.target_q}")
    if not bool(jnp.allclose(state.target_omega_inertial, 0.0, atol=1e-9, rtol=0.0)):
        raise AssertionError("inertial hold unexpectedly has non-zero target rate")


def test_nadir_lvlh_guidance_geometry_and_rate():
    cfg = default_config()
    cfg = replace(cfg, guidance=replace(cfg.guidance, mode="nadir_lvlh"))
    env = SatelliteEnv(cfg)
    state = env.reset(jax.random.PRNGKey(106), 4)

    z_body = jnp.tile(jnp.asarray([0.0, 0.0, 1.0], jnp.float32), (4, 1))
    x_body = jnp.tile(jnp.asarray([1.0, 0.0, 0.0], jnp.float32), (4, 1))
    z_i = rotate_body_to_inertial(state.target_q, z_body)
    x_i = rotate_body_to_inertial(state.target_q, x_body)
    r_hat = state.orbit.position_eci_m / jnp.linalg.norm(
        state.orbit.position_eci_m, axis=-1, keepdims=True
    )
    v = state.orbit.velocity_eci_m_s
    v_horizontal = v - jnp.sum(v * r_hat, axis=-1, keepdims=True) * r_hat
    v_hat = v_horizontal / jnp.linalg.norm(v_horizontal, axis=-1, keepdims=True)

    if not bool(jnp.allclose(z_i, -r_hat, atol=2e-6, rtol=0.0)):
        raise AssertionError("nadir target +Z does not point toward Earth center")
    if not bool(jnp.allclose(x_i, v_hat, atol=2e-6, rtol=0.0)):
        raise AssertionError("nadir target +X does not follow horizontal velocity")
    if not bool(jnp.all(jnp.linalg.norm(state.target_omega_inertial, axis=-1) > 1e-5)):
        raise AssertionError("LVLH target frame did not receive orbital angular rate")

    omega_body = rotate_inertial_to_body(state.target_q, state.target_omega_inertial)
    error = tracking_rate_error_body(state.target_q, omega_body, state.target_omega_inertial)
    if not bool(jnp.allclose(error, 0.0, atol=1e-8, rtol=0.0)):
        raise AssertionError(f"perfect LVLH tracking has nonzero rate error: {error}")



def test_nadir_reset_distribution_is_target_relative():
    cfg = default_config()
    cfg = replace(
        cfg,
        guidance=replace(cfg.guidance, mode="nadir_lvlh"),
        task=replace(
            cfg.task, reset_mode="fixed", fixed_angle_deg=20.0,
            max_initial_rate=0.0,
        ),
    )
    env = SatelliteEnv(cfg)
    state = env.reset(jax.random.PRNGKey(107), 2)
    angle = jnp.rad2deg(attitude_angle(attitude_error(state.target_q, state.physical.q)))
    if not bool(jnp.allclose(angle, 20.0, atol=2e-4, rtol=0.0)):
        raise AssertionError(f"nadir reset angle is not target-relative: {angle}")
    target_rate_body = rotate_inertial_to_body(
        state.physical.q, state.target_omega_inertial
    )
    if not bool(jnp.allclose(state.physical.omega, target_rate_body, atol=1e-8, rtol=0.0)):
        raise AssertionError("zero initial tracking-rate error did not follow LVLH frame rate")


def run_all():
    tests = [
        test_legacy_checkpoint_loads,
        test_fixed_lag_replays_delayed_star_solution,
        test_fixed_lag_handles_gyro_transport_delay,
        test_sensor_model_is_not_ground_truth,
        test_star_tracker_loses_lock_above_tracking_rate,
        test_motor_lag_deadzone_and_degradation,
        test_covariance_stays_well_formed,
        test_inertial_guidance_preserves_legacy_reference,
        test_nadir_lvlh_guidance_geometry_and_rate,
        test_nadir_reset_distribution_is_target_relative,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("All flight-like hardening diagnostics passed.")


if __name__ == "__main__":
    run_all()
