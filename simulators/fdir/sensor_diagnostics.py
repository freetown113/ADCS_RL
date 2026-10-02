from dataclasses import replace

import jax
import jax.numpy as jnp

from simulators.fdir.config import SensorConfig, default_config
from simulators.fdir.control import normalized_pd_body_action
from simulators.fdir.env import SatelliteEnv


def _a2_config():
    config = default_config()
    return replace(
        config, estimator=replace(config.estimator, enabled=False), sensors=_ideal_sensors()
    )

def _ideal_sensors(**kwargs):
    base = default_config().sensors
    clean = replace(
        base,
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
    )
    return replace(clean, **kwargs)

def _assert_allclose(actual, expected, atol=1.0e-6, message=""):
    if not bool(jnp.allclose(actual, expected, atol=atol, rtol=0.0)):
        raise AssertionError(f"{message}\nactual={actual}\nexpected={expected}")


def test_perfect_sensors_match_truth():
    config = _a2_config()
    config = replace(
        config,
        control=replace(config.control, mode="direct"),
        task=replace(config.task, fixed_angle_deg=0.0, max_initial_rate=0.05),
    )
    env = SatelliteEnv(config)
    state = env.reset(jax.random.PRNGKey(700), 8)
    _assert_allclose(state.sensors.gyro, state.physical.omega, 1.0e-8)
    _assert_allclose(state.sensors.wheel_speed, state.physical.wheel_speed, 1.0e-8)
    action = jnp.tile(jnp.asarray([[0.3, -0.2, 0.1]], jnp.float32), (8, 1))
    next_state, _, _, info = env.step(state, action)
    _assert_allclose(next_state.sensors.gyro, next_state.physical.omega, 1.0e-7)
    _assert_allclose(
        next_state.sensors.wheel_speed, next_state.physical.wheel_speed, 1.0e-6
    )
    _assert_allclose(info.gyro_age_seconds, jnp.zeros(8), 1.0e-8)


def test_low_rate_sensor_sample_and_hold():
    config = _a2_config()
    config = replace(
        config,
        control=replace(config.control, mode="direct"),
        task=replace(config.task, fixed_angle_deg=0.0, max_initial_rate=0.0),
        sensors=_ideal_sensors(gyro_rate_hz=10.0, gyro_latency_seconds=0.0, wheel_tach_rate_hz=10.0, wheel_tach_latency_seconds=0.0),
    )
    env = SatelliteEnv(config)
    state = env.reset(jax.random.PRNGKey(701), 1)
    action = jnp.asarray([[0.5, 0.0, 0.0]], dtype=jnp.float32)
    state, _, _, info = env.step(state, action)
    if not float(jnp.linalg.norm(state.physical.omega[0])) > 0.0:
        raise AssertionError("Plant did not move")
    _assert_allclose(state.sensors.gyro, jnp.zeros((1, 3)), 1.0e-8)
    _assert_allclose(info.gyro_age_seconds, jnp.asarray([0.05]), 1.0e-7)
    state, _, _, info = env.step(state, action)
    _assert_allclose(state.sensors.gyro, state.physical.omega, 1.0e-7)
    _assert_allclose(info.gyro_age_seconds, jnp.zeros(1), 1.0e-8)


def test_fixed_sensor_latency():
    config = _a2_config()
    config = replace(
        config,
        control=replace(config.control, mode="direct"),
        task=replace(config.task, fixed_angle_deg=0.0, max_initial_rate=0.0),
        sensors=_ideal_sensors(gyro_rate_hz=100.0, gyro_latency_seconds=0.02, wheel_tach_rate_hz=100.0, wheel_tach_latency_seconds=0.02),
    )
    env = SatelliteEnv(config)
    state = env.reset(jax.random.PRNGKey(702), 1)
    if bool(state.sensors.gyro_valid[0]):
        raise AssertionError("Positive-latency gyro should be invalid at reset")
    action = jnp.asarray([[0.5, 0.0, 0.0]], dtype=jnp.float32)
    state, _, _, info = env.step(state, action)
    if not bool(state.sensors.gyro_valid[0]):
        raise AssertionError("Gyro packet did not arrive")
    _assert_allclose(info.gyro_age_seconds, jnp.asarray([0.02]), 1.0e-7)
    _assert_allclose(info.wheel_age_seconds, jnp.full((1, 4), 0.02), 1.0e-7)
    if float(jnp.linalg.norm(state.physical.omega - state.sensors.gyro)) <= 0.0:
        raise AssertionError("Delayed measurement unexpectedly equals current truth")


def test_pd_uses_gyro_measurement():
    config = _a2_config()
    config = replace(config, task=replace(config.task, fixed_angle_deg=0.0))
    env = SatelliteEnv(config)
    state = env.reset(jax.random.PRNGKey(703), 1)
    physical = state.physical._replace(
        omega=jnp.asarray([[0.2, 0.0, 0.0]], dtype=jnp.float32)
    )
    sensors = state.sensors._replace(gyro=jnp.zeros((1, 3), dtype=jnp.float32))
    state = state._replace(physical=physical, sensors=sensors)
    _assert_allclose(
        normalized_pd_body_action(env, state), jnp.zeros((1, 3)), 1.0e-8
    )


def test_allocator_uses_tachometer_measurement():
    config = _a2_config()
    config = replace(
        config,
        control=replace(config.control, mode="direct"),
        task=replace(config.task, fixed_angle_deg=0.0),
    )
    env = SatelliteEnv(config)
    base = env.reset(jax.random.PRNGKey(704), 1)
    physical = base.physical._replace(
        wheel_speed=jnp.asarray([[100.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    )
    perfect = base._replace(
        physical=physical,
        sensors=base.sensors._replace(wheel_speed=physical.wheel_speed),
    )
    stale = base._replace(
        physical=physical,
        sensors=base.sensors._replace(wheel_speed=jnp.zeros((1, 4), jnp.float32)),
    )
    zero = jnp.zeros((1, 3), dtype=jnp.float32)
    _, _, _, perfect_info = env.step(perfect, zero)
    _, _, _, stale_info = env.step(stale, zero)
    if not float(stale_info.allocation_error_norm[0]) > float(
        perfect_info.allocation_error_norm[0] + 1.0e-4
    ):
        raise AssertionError("Allocator did not use tachometer telemetry")


def test_jitted_batched_step():
    env = SatelliteEnv(_a2_config())
    state = env.reset(jax.random.PRNGKey(705), 32)
    action = jnp.zeros((32, env.action_size), dtype=jnp.float32)
    step = jax.jit(env.step)
    next_state, reward, done, info = step(state, action)
    leaves = jax.tree_util.tree_leaves((next_state, reward, done, info))
    if not all(bool(jnp.all(jnp.isfinite(x))) for x in leaves if x.dtype != jnp.bool_):
        raise AssertionError("Non-finite values from jitted sensorized step")


def run_all():
    tests = [
        test_perfect_sensors_match_truth,
        test_low_rate_sensor_sample_and_hold,
        test_fixed_sensor_latency,
        test_pd_uses_gyro_measurement,
        test_allocator_uses_tachometer_measurement,
        test_jitted_batched_step,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("All sensor diagnostics passed.")


if __name__ == "__main__":
    run_all()
