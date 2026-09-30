import argparse
from dataclasses import replace

import jax
import jax.numpy as jnp

from simulators.fdir.config import SensorConfig, WheelFaultConfig, default_config, config_from_dict
from simulators.fdir.env import SatelliteEnv
from simulators.fdir.control import normalized_pd_body_action
from simulators.fdir.evaluation import compare_baselines, evaluate_policy
from simulators.fdir.network import make_network
from simulators.fdir.ppo import (
    TrainState, collect_rollout, compute_gae, flatten_rollout,
    make_optimizer, make_ppo_update,
)
from simulators.fdir.math3d import integrate_quaternion
from simulators.fdir.physics import (
    WHEEL_AXES,
    PhysicalState,
    allocate_body_torque,
    continuous_dynamics,
    direct_motor_actuation,
    momentum_balance_residual,
)
from simulators.fdir.guidance import earth_fixed_position_eci, guidance_target
from simulators.fdir.fdir import FAILED, update_fdir
from simulators.fdir.orbit import reset_orbit_state


def _assert_allclose(actual, expected, atol=1.0e-6, message=""):
    if not bool(jnp.allclose(actual, expected, atol=atol, rtol=0.0)):
        raise AssertionError(f"{message}\nactual={actual}\nexpected={expected}")



def test_perfect_sensors_match_truth():
    """Default 100 Hz zero-latency sensors reproduce the 100 Hz plant state."""
    config = default_config()
    config = replace(
        config,
        control=replace(config.control, mode="direct"),
        task=replace(config.task, fixed_angle_deg=0.0, max_initial_rate=0.05),
    )
    env = SatelliteEnv(config)
    state = env.reset(jax.random.PRNGKey(700), 8)
    _assert_allclose(state.sensors.gyro, state.physical.omega, 1.0e-8)
    _assert_allclose(
        state.sensors.wheel_speed, state.physical.wheel_speed, 1.0e-8
    )
    action = jnp.tile(
        jnp.asarray([[0.3, -0.2, 0.1]], dtype=jnp.float32), (8, 1)
    )
    next_state, _, _, info = env.step(state, action)
    _assert_allclose(next_state.sensors.gyro, next_state.physical.omega, 1.0e-7)
    _assert_allclose(
        next_state.sensors.wheel_speed,
        next_state.physical.wheel_speed,
        1.0e-6,
    )
    _assert_allclose(info.gyro_age_seconds, jnp.zeros(8), 1.0e-8)


def test_low_rate_sensor_sample_and_hold():
    """A 10 Hz sensor holds its reset sample during the first 50 ms control step."""
    config = default_config()
    config = replace(
        config,
        control=replace(config.control, mode="direct"),
        task=replace(config.task, fixed_angle_deg=0.0, max_initial_rate=0.0),
        sensors=SensorConfig(
            gyro_rate_hz=10.0,
            gyro_latency_seconds=0.0,
            wheel_tach_rate_hz=10.0,
            wheel_tach_latency_seconds=0.0,
        ),
    )
    env = SatelliteEnv(config)
    state = env.reset(jax.random.PRNGKey(701), 1)
    action = jnp.asarray([[0.5, 0.0, 0.0]], dtype=jnp.float32)

    state, _, _, info = env.step(state, action)
    if not float(jnp.linalg.norm(state.physical.omega[0])) > 0.0:
        raise AssertionError("Plant did not move during sample-and-hold test")
    _assert_allclose(
        state.sensors.gyro, jnp.zeros((1, 3)), 1.0e-8,
        "10 Hz gyro did not hold its t=0 sample through t=0.05 s",
    )
    _assert_allclose(info.gyro_age_seconds, jnp.asarray([0.05]), 1.0e-7)

    state, _, _, info = env.step(state, action)
    _assert_allclose(state.sensors.gyro, state.physical.omega, 1.0e-7)
    _assert_allclose(info.gyro_age_seconds, jnp.zeros(1), 1.0e-8)


def test_fixed_sensor_latency():
    """A 20 ms latency delivers a two-physics-step-old measurement."""
    config = default_config()
    config = replace(
        config,
        control=replace(config.control, mode="direct"),
        task=replace(config.task, fixed_angle_deg=0.0, max_initial_rate=0.0),
        sensors=SensorConfig(
            gyro_rate_hz=100.0,
            gyro_latency_seconds=0.02,
            wheel_tach_rate_hz=100.0,
            wheel_tach_latency_seconds=0.02,
        ),
    )
    env = SatelliteEnv(config)
    state = env.reset(jax.random.PRNGKey(702), 1)
    if bool(state.sensors.gyro_valid[0]):
        raise AssertionError("Positive-latency gyro should be invalid at reset")
    action = jnp.asarray([[0.5, 0.0, 0.0]], dtype=jnp.float32)
    state, _, _, info = env.step(state, action)
    if not bool(state.sensors.gyro_valid[0]):
        raise AssertionError("Gyro packet did not arrive after configured latency")
    _assert_allclose(info.gyro_age_seconds, jnp.asarray([0.02]), 1.0e-7)
    _assert_allclose(
        info.wheel_age_seconds, jnp.full((1, 4), 0.02), 1.0e-7
    )
    if float(jnp.linalg.norm(state.physical.omega - state.sensors.gyro)) <= 0.0:
        raise AssertionError("Delayed gyro unexpectedly equals current truth")


def test_pd_uses_gyro_measurement():
    """Residual/classical PD damping must not read true body rate."""
    config = default_config()
    config = replace(config, task=replace(config.task, fixed_angle_deg=0.0))
    env = SatelliteEnv(config)
    state = env.reset(jax.random.PRNGKey(703), 1)
    physical = state.physical._replace(
        omega=jnp.asarray([[0.2, 0.0, 0.0]], dtype=jnp.float32)
    )
    sensors = state.sensors._replace(gyro=jnp.zeros((1, 3), dtype=jnp.float32))
    state = state._replace(physical=physical, sensors=sensors)
    action = normalized_pd_body_action(env, state)
    _assert_allclose(
        action, jnp.zeros((1, 3)), 1.0e-8,
        "PD damping still appears to use true angular velocity",
    )


def test_allocator_uses_tachometer_measurement():
    """Stale wheel telemetry must create a real controller/plant mismatch."""
    config = default_config()
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
        sensors=base.sensors._replace(
            wheel_speed=jnp.zeros((1, 4), dtype=jnp.float32)
        ),
    )
    zero = jnp.zeros((1, 3), dtype=jnp.float32)
    _, _, _, perfect_info = env.step(perfect, zero)
    _, _, _, stale_info = env.step(stale, zero)
    if not float(stale_info.allocation_error_norm[0]) > float(
        perfect_info.allocation_error_norm[0] + 1.0e-4
    ):
        raise AssertionError(
            "Allocator output did not respond to stale tachometer telemetry"
        )


def test_wheel_geometry():
    norms = jnp.linalg.norm(WHEEL_AXES, axis=0)
    gram = WHEEL_AXES @ WHEEL_AXES.T
    _assert_allclose(norms, jnp.ones(4), 1.0e-6, "Wheel axes are not unit vectors")
    _assert_allclose(
        gram,
        (4.0 / 3.0) * jnp.eye(3),
        1.0e-6,
        "Wheel geometry is not isotropic",
    )


def test_quaternion_integration():
    q0 = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    omega = jnp.asarray([[0.0, 0.0, 0.5 * jnp.pi]], dtype=jnp.float32)
    q1 = integrate_quaternion(q0, omega, 1.0)
    expected = jnp.asarray(
        [[jnp.sqrt(0.5), 0.0, 0.0, jnp.sqrt(0.5)]], dtype=jnp.float32
    )
    _assert_allclose(q1, expected, 2.0e-6, "Quaternion integration convention mismatch")
    _assert_allclose(jnp.linalg.norm(q1, axis=-1), jnp.ones(1), 1.0e-6)


def test_allocation_healthy_and_failed():
    cfg = default_config().physics
    desired = jnp.asarray([[0.01, -0.008, 0.006]], dtype=jnp.float32)
    wheel_speed = jnp.zeros((1, 4), dtype=jnp.float32)

    for mask in (
        jnp.ones((1, 4), dtype=jnp.float32),
        jnp.asarray([[0.0, 1.0, 1.0, 1.0]], dtype=jnp.float32),
    ):
        _, _, achieved, error = allocate_body_torque(
            desired, wheel_speed, mask, cfg
        )
        _assert_allclose(achieved, desired, 2.0e-6, "Body torque allocation failed")
        if float(jnp.max(jnp.linalg.norm(error, axis=-1))) > 2.0e-6:
            raise AssertionError(f"Allocation error too large: {error}")


def test_continuous_momentum_balance():
    cfg = default_config().physics
    state = PhysicalState(
        q=jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
        omega=jnp.asarray([[0.2, -0.1, 0.3]], dtype=jnp.float32),
        wheel_speed=jnp.asarray([[20.0, -40.0, 10.0, 30.0]], dtype=jnp.float32),
        motor_torque=jnp.zeros((1, 4), dtype=jnp.float32),
    )
    net_rotor = jnp.asarray([[0.1, -0.2, 0.05, 0.15]], dtype=jnp.float32)
    external = jnp.asarray([[0.01, -0.02, 0.03]], dtype=jnp.float32)
    omega_dot, wheel_dot = continuous_dynamics(state, net_rotor, external, cfg)
    residual = momentum_balance_residual(
        state, omega_dot, wheel_dot, external, cfg
    )
    if float(jnp.max(jnp.linalg.norm(residual, axis=-1))) > 2.0e-6:
        raise AssertionError(f"Momentum balance residual too large: {residual}")


def test_zero_action_stationary():
    config = default_config()
    config = replace(
        config,
        task=replace(config.task, fixed_angle_deg=0.0, max_initial_rate=0.0),
    )
    env = SatelliteEnv(config)
    state = env.reset(jax.random.PRNGKey(0), 4)
    zero = jnp.zeros((4, env.action_size), dtype=jnp.float32)
    for _ in range(20):
        state, _, _, _ = env.step(state, zero)
    _assert_allclose(state.physical.omega, jnp.zeros((4, 3)), 1.0e-8)
    _assert_allclose(state.physical.wheel_speed, jnp.zeros((4, 4)), 1.0e-8)
    _assert_allclose(
        state.physical.q,
        jnp.tile(jnp.asarray([1.0, 0.0, 0.0, 0.0]), (4, 1)),
        1.0e-7,
    )


def test_action_sign():
    config = default_config()
    config = replace(config, task=replace(config.task, fixed_angle_deg=0.0))
    env = SatelliteEnv(config)
    state = env.reset(jax.random.PRNGKey(1), 1)
    action = jnp.asarray([[0.5, 0.0, 0.0]], dtype=jnp.float32)
    next_state, _, _, info = env.step(state, action)
    if not float(next_state.physical.omega[0, 0]) > 0.0:
        raise AssertionError(
            "Positive desired X body torque did not produce positive X body rate"
        )
    if not float(info.achieved_body_torque[0, 0]) > 0.0:
        raise AssertionError("Allocator/body torque sign mismatch")


def test_pd_baseline_and_reward_ordering():
    env = SatelliteEnv(default_config())
    initial = env.reset(jax.random.PRNGKey(2), 32)
    results = compare_baselines(env, initial)

    pd = results["pd"]
    zero = results["zero"]
    wrong = results["opposite_pd"]

    if float(pd["final_angle_deg_mean"]) >= 0.5:
        raise AssertionError(f"PD final angle is too large: {pd}")
    if float(pd["final_rate_mean"]) >= 0.01:
        raise AssertionError(f"PD final rate is too large: {pd}")
    if float(pd["success_rate"]) < 0.99:
        raise AssertionError(f"PD did not satisfy dwell success: {pd}")
    if not float(pd["episode_return_mean"]) > float(zero["episode_return_mean"]):
        raise AssertionError(f"Reward does not prefer PD over zero: {results}")
    if not float(zero["episode_return_mean"]) > float(wrong["episode_return_mean"]):
        raise AssertionError(f"Reward does not penalize wrong-sign PD: {results}")



def test_failed_motor_is_not_deleted():
    """Power loss removes motor authority but keeps rotor friction/momentum."""
    cfg = default_config().physics
    wheel_speed = jnp.asarray([[100.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    mask = jnp.asarray([[0.0, 1.0, 1.0, 1.0]], dtype=jnp.float32)
    desired = jnp.zeros((1, 3), dtype=jnp.float32)
    motor, net, achieved, error = allocate_body_torque(
        desired, wheel_speed, mask, cfg
    )
    _assert_allclose(motor[:, 0], jnp.zeros(1), 1.0e-8, "Failed motor is not off")
    expected_passive = -cfg.bearing_friction * wheel_speed[:, 0]
    _assert_allclose(
        net[:, 0], expected_passive, 1.0e-6,
        "Failed rotor lost its physical bearing-friction torque",
    )
    if float(jnp.max(jnp.linalg.norm(error, axis=-1))) > 2.0e-5:
        raise AssertionError(
            f"Healthy wheels did not compensate failed-wheel passive torque: {achieved}"
        )


def test_motor_direct_action_sign():
    cfg = default_config().physics
    wheel_speed = jnp.zeros((1, 4), dtype=jnp.float32)
    mask = jnp.ones((1, 4), dtype=jnp.float32)
    # Wheel 0 has positive X axis; positive motor torque gives negative body X torque.
    command = jnp.asarray([[0.01, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    _, _, body = direct_motor_actuation(command, wheel_speed, mask, cfg)
    if not float(body[0, 0]) < 0.0:
        raise AssertionError(f"Direct motor reaction-torque sign is wrong: {body}")


def test_fixed_interval_fault_schedule():
    config = default_config()
    config = replace(
        config,
        faults=WheelFaultConfig(
            mode="fixed_interval",
            wheel_selection="fixed",
            wheel_index=2,
            start_seconds=0.10,
            duration_seconds=0.10,
        ),
    )
    env = SatelliteEnv(config)
    state = env.reset(jax.random.PRNGKey(99), 2)
    _assert_allclose(state.wheel_mask, jnp.ones((2, 4)), 1.0e-8)
    zero = jnp.zeros((2, env.action_size), dtype=jnp.float32)
    # After two 0.05 s control steps, next state is at t=0.10 s: wheel 2 is off.
    state, _, _, _ = env.step(state, zero)
    state, _, _, _ = env.step(state, zero)
    if not bool(jnp.all(state.wheel_mask[:, 2] == 0.0)):
        raise AssertionError(f"Fixed fault did not start on schedule: {state.wheel_mask}")
    # At t=0.20 s the fixed interval has ended.
    state, _, _, _ = env.step(state, zero)
    state, _, _, _ = env.step(state, zero)
    _assert_allclose(state.wheel_mask, jnp.ones((2, 4)), 1.0e-8)


def test_end_to_end_residual_policy():
    """The untrained residual policy must inherit the stable PD behavior."""
    config = default_config()
    env = SatelliteEnv(config)
    network = make_network(env.observation_size, env.action_size, config.network)
    init_key, reset_key = jax.random.split(jax.random.PRNGKey(123))
    state = env.reset(reset_key, 32)
    dummy_state = env.reset(reset_key, 1)
    params = network.init(init_key, env.observe(dummy_state))
    _, _, metrics = evaluate_policy(env, params, network.apply, state)
    if float(metrics["success_rate"]) < 0.99:
        raise AssertionError(f"Residual policy lost nominal PD stability: {metrics}")
    if float(metrics["final_angle_deg_mean"]) >= 0.5:
        raise AssertionError(f"End-to-end final angle too large: {metrics}")


def run_ppo_smoke():
    """Runs one complete rollout and one PPO update; all outputs must be finite."""
    config = default_config()
    config = replace(
        config,
        ppo=replace(config.ppo, num_envs=16, update_epochs=1, num_minibatches=2),
    )
    env = SatelliteEnv(config)
    network = make_network(env.observation_size, env.action_size, config.network)
    key = jax.random.PRNGKey(456)
    key, init_key, reset_key, rollout_key, update_key = jax.random.split(key, 5)
    initial = env.reset(reset_key, config.ppo.num_envs)
    params = network.init(init_key, env.observe(env.reset(reset_key, 1)))
    optimizer = make_optimizer(config)
    train_state = TrainState(params, optimizer.init(params))
    _, rollout = collect_rollout(
        env, params, network.apply, initial, rollout_key, config.network.fixed_log_std
    )
    advantage, returns = compute_gae(
        rollout.reward, rollout.done, rollout.old_value,
        config.ppo.gamma, config.ppo.gae_lambda,
    )
    normalized = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)
    batch = flatten_rollout(rollout, normalized, returns)
    update = make_ppo_update(network.apply, optimizer, config)
    next_state, metrics = update(train_state, batch, update_key)
    leaves = jax.tree_util.tree_leaves((next_state, metrics))
    if not all(bool(jnp.all(jnp.isfinite(x))) for x in leaves):
        raise AssertionError(f"Non-finite PPO smoke output: {metrics}")
    if float(metrics["action_reconstruction_error"]) > 1e-6:
        raise AssertionError(f"Tanh action/log-prob mismatch: {metrics}")
    print("PASS run_ppo_smoke")


def _assert_close(a, b, tol=1e-5):
    if not bool(jnp.all(jnp.abs(a - b) <= tol)):
        raise AssertionError(f"not close: {a} vs {b}")


def b2_diagnostics() -> None:
    cfg = default_config()
    orbit = reset_orbit_state(2, cfg.orbit)

    # Improved inertial hold: target is configurable rather than hard-coded identity.
    q_custom = (0.9238795, 0.0, 0.3826834, 0.0)
    g = replace(cfg.guidance, mode="inertial_hold", inertial_target_q=q_custom)
    tgt = guidance_target(orbit, g, cfg.orbit)
    _assert_close(tgt.q_body_to_inertial[0], jnp.asarray(q_custom), 2e-5)
    _assert_close(tgt.omega_inertial_rad_s, jnp.zeros((2, 3)))

    # Earth-fixed target moves in ECI because Earth rotates.
    p0, _, _ = earth_fixed_position_eci(jnp.asarray([0.0]), 0.0, 0.0, 0.0, cfg.orbit)
    p1, _, _ = earth_fixed_position_eci(jnp.asarray([100.0]), 0.0, 0.0, 0.0, cfg.orbit)
    if not float(jnp.linalg.norm(p1 - p0)) > 1_000.0:
        raise AssertionError("Earth-fixed point did not rotate in ECI")

    # Ground tracking has a moving reference even though the target is fixed on Earth.
    gt = guidance_target(orbit, replace(cfg.guidance, mode="ground_target"), cfg.orbit)
    if not bool(jnp.all(gt.reference_valid)):
        raise AssertionError("initial equatorial target should be visible below initial spacecraft")
    if not float(jnp.linalg.norm(gt.omega_inertial_rad_s[0])) > 1e-5:
        raise AssertionError("ground-target reference rate incorrectly zero")

    # Scheduled slew is stationary before start, moving during the slew, stationary after.
    gs = replace(
        cfg.guidance,
        mode="scheduled_slew",
        slew_start_seconds=1.0,
        slew_max_rate_deg_s=2.0,
        slew_max_accel_deg_s2=1.0,
    )
    before = guidance_target(orbit._replace(time_s=jnp.asarray([0.0, 0.0])), gs, cfg.orbit)
    during = guidance_target(orbit._replace(time_s=jnp.asarray([5.0, 5.0])), gs, cfg.orbit)
    after = guidance_target(orbit._replace(time_s=jnp.asarray([60.0, 60.0])), gs, cfg.orbit)
    _assert_close(before.omega_inertial_rad_s, jnp.zeros((2, 3)))
    if not float(jnp.linalg.norm(during.omega_inertial_rad_s[0])) > 1e-5:
        raise AssertionError("scheduled slew has zero rate during maneuver")
    _assert_close(after.omega_inertial_rad_s, jnp.zeros((2, 3)))

    # Actor observation is invariant to simulator wheel-fault truth when FDIR belief is fixed.
    env = SatelliteEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0), 2)
    obs_a = env.observe(state)
    fake_truth = state._replace(wheel_mask=jnp.zeros_like(state.wheel_mask))
    obs_b = env.observe(fake_truth)
    _assert_close(obs_a, obs_b, 0.0)

    # Synthetic analytical-redundancy test: commanded wheel 0 has zero tach
    # acceleration while other wheels respond. FDIR must infer wheel-0 authority
    # loss without reading env.wheel_mask.
    f = state.fdir
    sensors = state.sensors
    estimator = state.estimator
    command = jnp.full((2, 4), 0.015, jnp.float32)
    speed = jnp.zeros((2, 4), jnp.float32)
    for k in range(1, 45):
        speed = speed.at[:, 1:].add(
            (0.015 / cfg.physics.wheel_inertia) * cfg.physics.physics_dt
        )
        sensors = sensors._replace(
            wheel_speed=speed,
            wheel_valid=jnp.ones_like(sensors.wheel_valid),
            wheel_sample_step=jnp.full_like(sensors.wheel_sample_step, k),
        )
        f = update_fdir(
            f, sensors, estimator, command, jnp.full((2,), k, jnp.int32),
            cfg.physics, cfg.sensors, cfg.estimator, cfg.fdir,
        )
    if not bool(jnp.all(f.wheel_motor_health[:, 0] == FAILED)):
        raise AssertionError("wheel-response FDIR failed to isolate synthetic dead motor")
    if not bool(jnp.all(f.wheel_authority_estimate[:, 0] < 0.10)):
        raise AssertionError("wheel authority estimate did not converge toward zero")

    # Old checkpoint configs that said `truth` are migrated to FDIR semantics while
    # retaining the same 4-value observation width.
    import dataclasses
    data = dataclasses.asdict(cfg)
    data["observation"]["wheel_mask_source"] = "truth"
    migrated = config_from_dict(data)
    if migrated.observation.wheel_mask_source != "fdir":
        raise AssertionError("legacy truth mask was not migrated to FDIR authority")

    print("B.2 diagnostics passed")


def run_all():
    tests = [
        test_perfect_sensors_match_truth,
        test_low_rate_sensor_sample_and_hold,
        test_fixed_sensor_latency,
        test_pd_uses_gyro_measurement,
        test_allocator_uses_tachometer_measurement,
        test_wheel_geometry,
        test_quaternion_integration,
        test_allocation_healthy_and_failed,
        test_continuous_momentum_balance,
        test_zero_action_stationary,
        test_action_sign,
        test_failed_motor_is_not_deleted,
        test_motor_direct_action_sign,
        test_fixed_interval_fault_schedule,
        test_pd_baseline_and_reward_ordering,
        test_end_to_end_residual_policy,
        b2_diagnostics,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("All mandatory diagnostics passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-smoke", action="store_true")
    args = parser.parse_args()
    run_all()
    if args.ppo_smoke:
        run_ppo_smoke()


if __name__ == "__main__":
    main()
