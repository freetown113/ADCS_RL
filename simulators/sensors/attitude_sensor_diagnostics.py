from dataclasses import replace

import jax
import jax.numpy as jnp

from simulators.sensors.config import OrbitConfig, SensorConfig, default_config
from simulators.sensors.env import SatelliteEnv
from simulators.sensors.math3d import rotate_inertial_to_body


def _a2_config():
    config = default_config()
    return replace(
        config, estimator=replace(config.estimator, enabled=False)
    )


def _assert_allclose(actual, expected, atol=1.0e-6, message=""):
    if not bool(jnp.allclose(actual, expected, atol=atol, rtol=0.0)):
        raise AssertionError(f"{message}\nactual={actual}\nexpected={expected}")


def test_clean_measurements_match_truth_at_reset():
    env = SatelliteEnv(_a2_config())
    state = env.reset(jax.random.PRNGKey(800), 4)

    _assert_allclose(state.sensors.star_tracker_q, state.physical.q, 1.0e-7)
    expected_mag = rotate_inertial_to_body(
        state.physical.q, state.orbit.magnetic_field_eci_t
    )
    expected_sun = rotate_inertial_to_body(
        state.physical.q, state.orbit.sun_direction_eci
    )
    expected_sun = expected_sun / jnp.linalg.norm(
        expected_sun, axis=-1, keepdims=True
    )
    _assert_allclose(state.sensors.magnetometer_body_t, expected_mag, 1.0e-10)
    _assert_allclose(state.sensors.sun_direction_body, expected_sun, 1.0e-7)
    _assert_allclose(
        state.sensors.gnss_position_eci_m, state.orbit.position_eci_m, 0.5
    )
    _assert_allclose(
        state.sensors.gnss_velocity_eci_m_s, state.orbit.velocity_eci_m_s, 1.0e-3
    )
    _assert_allclose(state.sensors.gnss_time_s, state.orbit.time_s, 1.0e-8)


def test_orbit_propagation_preserves_circular_geometry():
    env = SatelliteEnv(_a2_config())
    state = env.reset(jax.random.PRNGKey(801), 8)
    initial_radius = jnp.linalg.norm(state.orbit.position_eci_m, axis=-1)
    initial_speed = jnp.linalg.norm(state.orbit.velocity_eci_m_s, axis=-1)
    zero = jnp.zeros((8, env.action_size), dtype=jnp.float32)
    step = jax.jit(env.step)
    for _ in range(40):
        state, _, _, _ = step(state, zero)
    _assert_allclose(
        jnp.linalg.norm(state.orbit.position_eci_m, axis=-1),
        initial_radius,
        2.0,
        "Circular orbit radius drifted",
    )
    _assert_allclose(
        jnp.linalg.norm(state.orbit.velocity_eci_m_s, axis=-1),
        initial_speed,
        2.0e-3,
        "Circular orbit speed drifted",
    )
    _assert_allclose(state.orbit.time_s, jnp.full((8,), 2.0), 2.0e-6)


def test_star_tracker_sample_and_hold():
    config = _a2_config()
    config = replace(
        config,
        control=replace(config.control, mode="direct"),
        task=replace(config.task, fixed_angle_deg=0.0),
        sensors=replace(
            config.sensors,
            star_tracker_rate_hz=2.0,
            star_tracker_latency_seconds=0.0,
        ),
    )
    env = SatelliteEnv(config)
    state = env.reset(jax.random.PRNGKey(802), 1)
    initial_measurement = state.sensors.star_tracker_q
    action = jnp.asarray([[0.8, 0.0, 0.0]], dtype=jnp.float32)
    state, _, _, info = env.step(state, action)
    if float(jnp.linalg.norm(state.physical.q - initial_measurement)) <= 0.0:
        raise AssertionError("Plant attitude did not change")
    _assert_allclose(state.sensors.star_tracker_q, initial_measurement, 1.0e-8)
    _assert_allclose(info.star_tracker_age_seconds, jnp.asarray([0.05]), 1.0e-7)

    # At 2 Hz the next attitude packet is sampled at t=0.5 s.
    for _ in range(9):
        state, _, _, info = env.step(state, action)
    _assert_allclose(state.sensors.star_tracker_q, state.physical.q, 1.0e-6)
    _assert_allclose(info.star_tracker_age_seconds, jnp.zeros(1), 1.0e-7)


def test_attitude_sensor_fixed_latency():
    config = _a2_config()
    config = replace(
        config,
        control=replace(config.control, mode="direct"),
        task=replace(config.task, fixed_angle_deg=0.0),
        sensors=SensorConfig(
            gyro_rate_hz=100.0,
            gyro_latency_seconds=0.0,
            wheel_tach_rate_hz=100.0,
            wheel_tach_latency_seconds=0.0,
            star_tracker_rate_hz=100.0,
            star_tracker_latency_seconds=0.02,
            magnetometer_rate_hz=100.0,
            magnetometer_latency_seconds=0.02,
            sun_sensor_rate_hz=100.0,
            sun_sensor_latency_seconds=0.02,
            gnss_rate_hz=100.0,
            gnss_latency_seconds=0.02,
        ),
    )
    env = SatelliteEnv(config)
    state = env.reset(jax.random.PRNGKey(803), 1)
    if bool(state.sensors.star_tracker_valid[0]):
        raise AssertionError("Positive-latency star tracker should be invalid at reset")
    action = jnp.asarray([[0.8, 0.0, 0.0]], dtype=jnp.float32)
    state, _, _, info = env.step(state, action)

    _assert_allclose(
        state.sensors.star_tracker_q,
        state.sensors.q_history[-3],
        1.0e-7,
        "Star tracker did not deliver the two-step-old attitude",
    )
    _assert_allclose(info.star_tracker_age_seconds, jnp.asarray([0.02]), 1.0e-7)
    _assert_allclose(info.magnetometer_age_seconds, jnp.asarray([0.02]), 1.0e-7)
    _assert_allclose(info.sun_sensor_age_seconds, jnp.asarray([0.02]), 1.0e-7)
    _assert_allclose(info.gnss_age_seconds, jnp.asarray([0.02]), 1.0e-7)


def test_optional_eclipse_invalidates_sun_sensor():
    config = _a2_config()
    config = replace(
        config,
        orbit=OrbitConfig(initial_argument_of_latitude_deg=180.0),
        sensors=replace(config.sensors, sun_sensor_eclipse_enabled=True),
    )
    env = SatelliteEnv(config)
    state = env.reset(jax.random.PRNGKey(804), 1)
    if bool(state.orbit.sun_visible[0]):
        raise AssertionError("Configured reset was expected to be in eclipse")
    if bool(state.sensors.sun_sensor_valid[0]):
        raise AssertionError("Sun sensor remained valid in eclipse")


def test_observation_shape_unchanged():
    env = SatelliteEnv(_a2_config())
    state = env.reset(jax.random.PRNGKey(805), 3)
    observation = env.observe(state)
    if observation.shape != (3, env.observation_size):
        raise AssertionError(
            f"Observation shape changed: {observation.shape}, {env.observation_size}"
        )


def run_all():
    tests = [
        test_clean_measurements_match_truth_at_reset,
        test_orbit_propagation_preserves_circular_geometry,
        test_star_tracker_sample_and_hold,
        test_attitude_sensor_fixed_latency,
        test_optional_eclipse_invalidates_sun_sensor,
        test_observation_shape_unchanged,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("All Milestone A.2 attitude/navigation sensor diagnostics passed.")


if __name__ == "__main__":
    run_all()
