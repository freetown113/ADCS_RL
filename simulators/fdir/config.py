from dataclasses import dataclass, field
from typing import Literal, Tuple


ResetMode = Literal["fixed", "cone"]
ControlMode = Literal["residual_pd", "direct", "motor_direct"]
FaultMode = Literal[
    "none",
    "permanent",
    "fixed_interval",
    "random_interval",
    "stochastic",
]
WheelSelection = Literal["fixed", "random"]
GuidanceMode = Literal[
    "inertial_hold",
    "nadir_lvlh",
    "ground_target",
    "ground_station",
    "sun_pointing",
    "scheduled_slew",
]


@dataclass(frozen=True)
class PhysicsConfig:
    """Physical parameters and integration settings."""

    body_inertia: Tuple[float, float, float] = (2.0, 2.0, 2.5)
    wheel_inertia: float = 9.0e-4

    bearing_friction: float = 3.0e-6
    coulomb_friction_torque: float = 2.0e-4
    stiction_torque: float = 3.0e-4
    stiction_speed_rad_s: float = 0.5

    max_wheel_speed: float = 628.0
    max_motor_torque: float = 0.05

    motor_time_constant_s: float = 0.03
    motor_dead_zone_torque: float = 2.0e-4
    motor_command_quantization_torque: float = 5.0e-5
    motor_torque_scale: float = 1.0

    body_torque_limit: Tuple[float, float, float] = (0.02, 0.02, 0.02)

    motor_control_limit: float = 0.03

    physics_dt: float = 0.01
    control_dt: float = 0.05
    allocation_regularization: float = 1.0e-7

    @property
    def substeps(self) -> int:
        ratio = self.control_dt / self.physics_dt
        rounded = int(round(ratio))
        if abs(ratio - rounded) > 1.0e-9:
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        if rounded < 1:
            raise ValueError("control_dt must be >= physics_dt")
        return rounded


@dataclass(frozen=True)
class TaskConfig:
    reset_mode: ResetMode = "cone"
    fixed_axis: Tuple[float, float, float] = (1.0, 0.0, 1.0)
    fixed_angle_deg: float = 90.0
    max_initial_angle_deg: float = 180.0
    max_initial_rate: float = 0.2

    episode_seconds: float = 90.0
    success_angle_deg: float = 1.0
    success_rate: float = 0.01
    success_dwell_seconds: float = 1.0


@dataclass(frozen=True)
class ControlConfig:
    """
    residual_pd:    3 policy outputs correct a PD body-torque command.
    direct:         3 policy outputs command body torque directly.
    motor_direct:   4 policy outputs command the four wheel motors directly.
    """
    mode: ControlMode = "motor_direct"
    pd_natural_frequency: float = 0.50
    pd_damping: float = 1.0
    residual_scale: float = 0.20


@dataclass(frozen=True)
class WheelFaultConfig:
    """
    Modes:
      none              no failures.
      permanent         selected wheel fails at ``start_seconds`` until episode end.
      fixed_interval    selected wheel fails for a fixed time window.
      random_interval   start/duration are sampled independently per environment.
      stochastic        healthy<->failed transitions follow per-second hazard rates.
    """

    mode: FaultMode = "stochastic"
    wheel_selection: WheelSelection = "random"
    wheel_index: int = 0

    start_seconds: float = 2.0
    duration_seconds: float = 2.0

    random_start_min_seconds: float = 0.5
    random_start_max_seconds: float = 5.0
    random_duration_min_seconds: float = 0.5
    random_duration_max_seconds: float = 3.0

    failure_rate_per_second: float = 0.10
    recovery_rate_per_second: float = 0.50

    fault_torque_fraction: float = 0.0


@dataclass(frozen=True)
class OrbitConfig:
    earth_mu_m3_s2: float = 3.986004418e14
    earth_radius_m: float = 6_378_137.0
    altitude_m: float = 500_000.0
    inclination_deg: float = 51.6
    raan_deg: float = 0.0
    initial_argument_of_latitude_deg: float = 0.0

    magnetic_equator_field_t: float = 3.12e-5
    magnetic_dipole_tilt_deg: float = 11.0
    magnetic_dipole_longitude_deg: float = 0.0

    # Earth-to-Sun unit direction in ECI, treated as constant over one episode.
    sun_direction_eci: Tuple[float, float, float] = (1.0, 0.0, 0.0) #fall


@dataclass(frozen=True)
class SensorConfig:
    gyro_rate_hz: float = 100.0
    gyro_latency_seconds: float = 0.2
    gyro_noise_density_rad_s_sqrt_hz: float = 1.0e-4
    gyro_initial_bias_std_rad_s: float = 5.0e-4
    gyro_bias_random_walk_rad_s_per_sqrt_s: float = 5.0e-6
    gyro_scale_factor_std: float = 5.0e-4
    gyro_misalignment_std_deg: float = 0.02
    gyro_quantization_rad_s: float = 1.0e-5
    gyro_clip_rad_s: float = 4.0
    gyro_packet_loss_probability: float = 0.0

    wheel_tach_rate_hz: float = 100.0
    wheel_tach_latency_seconds: float = 0.2
    wheel_tach_noise_std_rad_s: float = 0.10
    wheel_tach_scale_factor_std: float = 5.0e-4
    wheel_tach_quantization_rad_s: float = 0.05
    wheel_tach_packet_loss_probability: float = 0.0

    star_tracker_rate_hz: float = 2.0
    star_tracker_latency_seconds: float = 0.08
    star_tracker_noise_std_deg: float = 0.01
    star_tracker_alignment_std_deg: float = 0.02
    star_tracker_packet_loss_probability: float = 0.001
    star_tracker_outlier_probability: float = 0.0005
    star_tracker_outlier_std_deg: float = 2.0
    star_tracker_acquisition_time_s: float = 1.0
    star_tracker_max_acquisition_rate_deg_s: float = 1.0
    star_tracker_max_tracking_rate_deg_s: float = 3.0
    star_tracker_boresight_body: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    star_tracker_sun_exclusion_deg: float = 30.0
    star_tracker_earth_limb_exclusion_deg: float = 10.0
    # Fine-pointing experiments may start with a previously acquired solution.
    # Full mission-mode simulations should set this False.
    star_tracker_initially_locked: bool = True

    magnetometer_rate_hz: float = 10.0
    magnetometer_latency_seconds: float = 0.02
    magnetometer_noise_std_t: float = 1.0e-7
    magnetometer_bias_std_t: float = 5.0e-7
    magnetometer_scale_factor_std: float = 2.0e-3
    magnetometer_misalignment_std_deg: float = 0.10
    magnetometer_quantization_t: float = 1.0e-8
    magnetometer_clip_t: float = 1.0e-4
    magnetometer_packet_loss_probability: float = 0.0

    sun_sensor_rate_hz: float = 10.0
    sun_sensor_latency_seconds: float = 0.02
    sun_sensor_noise_std_deg: float = 0.30
    sun_sensor_misalignment_std_deg: float = 0.10
    sun_sensor_packet_loss_probability: float = 0.001
    sun_sensor_eclipse_enabled: bool = True

    gnss_rate_hz: float = 1.0
    gnss_latency_seconds: float = 0.02
    gnss_position_noise_std_m: float = 2.0
    gnss_velocity_noise_std_m_s: float = 0.05
    gnss_packet_loss_probability: float = 0.001


@dataclass(frozen=True)
class EstimatorConfig:
    enabled: bool = True
    initialize_from_star_tracker: bool = True
    hard_acquire_first_star_tracker: bool = True
    use_fixed_lag_replay: bool = True
    fixed_lag_history_seconds: float = 0.25

    use_star_tracker: bool = True
    use_magnetometer: bool = True
    use_sun_sensor: bool = True

    initial_attitude_sigma_deg: float = 20.0
    initial_gyro_bias_sigma_rad_s: float = 0.002

    gyro_process_noise_rad_s_sqrt_hz: float = 1.0e-4
    gyro_bias_random_walk_rad_s2_sqrt_hz: float = 5.0e-6

    star_tracker_noise_std_deg: float = 0.015
    star_hard_acquisition_sigma_deg: float = 2.0
    magnetometer_noise_std_deg: float = 0.50
    sun_sensor_noise_std_deg: float = 0.40

    star_tracker_nis_gate: float = 16.3
    magnetometer_nis_gate: float = 13.8
    sun_sensor_nis_gate: float = 13.8

    covariance_floor: float = 1.0e-12
    innovation_regularization: float = 1.0e-10


@dataclass(frozen=True)
class GuidanceConfig:
    """Reference-attitude generator.
    ``inertial_hold`` preserves the original fixed inertial identity target.
    ``nadir_lvlh`` tracks a local orbital frame with body +Z toward nadir and
    body +X along the horizontal velocity direction.
    """
    mode: GuidanceMode = "inertial_hold"

    # Arbitrary inertial attitude. The legacy identity hold is recovered by the
    # default quaternion. Quaternion convention is [w, x, y, z], body -> ECI.
    inertial_target_q: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # Earth-fixed target/station coordinates. These are mission data, not sensor
    # truth: in flight they would normally come from a target/station catalogue or
    # an uploaded mission timeline.
    earth_target_lat_deg: float = 0.0
    earth_target_lon_deg: float = 0.0
    earth_target_alt_m: float = 0.0
    earth_tracking_body_axis: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"] = "+Z"
    earth_target_fallback_mode: Literal["nadir_lvlh", "inertial_hold"] = "nadir_lvlh"
    antenna_half_beamwidth_deg: float = 10.0

    # Sun-pointing/safe-attitude reference. The configured body axis is aligned
    # with the inertial Sun direction; roll is fixed deterministically.
    sun_pointing_body_axis: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"] = "+Z"

    # Rest-to-rest scheduled slew between two inertial attitudes. The trajectory
    # is generated with a triangular/trapezoidal angular-rate profile constrained
    # by the requested maximum rate and acceleration.
    slew_start_q: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    slew_end_q: Tuple[float, float, float, float] = (0.9238795, 0.0, 0.3826834, 0.0)
    slew_start_seconds: float = 1.0
    slew_max_rate_deg_s: float = 2.0
    slew_max_accel_deg_s2: float = 0.5


@dataclass(frozen=True)
class FDIRConfig:
    """Telemetry-derived health estimation thresholds.
    No member of this config references injected fault identity. FDIR receives
    commands, telemetry and estimator diagnostics only.
    """

    enabled: bool = True
    wheel_authority_ewma_alpha: float = 0.08
    wheel_min_excitation_torque: float = 0.003
    wheel_degraded_authority: float = 0.80
    wheel_failed_authority: float = 0.15
    wheel_recovery_authority: float = 0.90
    wheel_suspect_samples: int = 5
    wheel_fail_samples: int = 20
    wheel_recovery_samples: int = 50
    wheel_speed_monitor_fraction: float = 0.95

    # Watchdog multipliers are relative to nominal sensor sample period.
    fast_sensor_stale_periods: float = 4.0
    slow_sensor_stale_periods: float = 4.0
    nis_ewma_alpha: float = 0.15
    nis_suspect_ratio: float = 0.80
    innovation_suspect_samples: int = 3
    innovation_fail_samples: int = 8
    sensor_recovery_samples: int = 10

    estimator_fine_sigma_deg: float = 1.0
    estimator_degraded_sigma_deg: float = 5.0
    estimator_lost_sigma_deg: float = 20.0


@dataclass(frozen=True)
class SupervisorConfig:
    """Supervisory-GNC mode transition thresholds and hysteresis."""

    enabled: bool = True
    enter_degraded_dwell_seconds: float = 0.25
    enter_safe_dwell_seconds: float = 0.50
    recovery_dwell_seconds: float = 3.0
    acquisition_dwell_seconds: float = 0.50
    minimum_control_wheels: int = 3
    degraded_wheel_authority: float = 0.80
    failed_wheel_authority: float = 0.15
    wheel_speed_degraded_fraction: float = 0.90
    wheel_speed_critical_fraction: float = 0.98


@dataclass(frozen=True)
class ObservationConfig:
    omega_scale: float = 0.20
    include_previous_action: bool = True
    include_wheel_mask: bool = True
    wheel_mask_source: Literal["fdir", "unknown"] = "fdir"
    include_estimator_confidence: bool = False
    include_supervisory_mode: bool = False

@dataclass(frozen=True)
class RewardConfig:
    progress_weight: float = 10.0
    attitude_weight: float = 1.0
    rate_weight: float = 0.20
    action_weight: float = 0.002
    smoothness_weight: float = 0.001
    wheel_speed_weight: float = 0.0005
    success_bonus_per_second: float = 0.10

    rate_scale: float = 0.20
    huber_delta: float = 1.0


@dataclass(frozen=True)
class NetworkConfig:
    hidden_sizes: Tuple[int, ...] = (128, 128)
    activation: Literal["tanh", "relu"] = "tanh"
    policy_mean_limit: float = 2.0
    fixed_log_std: float = -1.5


@dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 512
    learning_rate: float = 3.0e-5
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.10
    value_clip_epsilon: float = 0.20
    value_coef: float = 0.50
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.50
    update_epochs: int = 4
    num_minibatches: int = 8
    adam_eps: float = 1.0e-5


@dataclass(frozen=True)
class RunConfig:
    seed: int = 1337
    total_updates: int = 5_000
    log_every: int = 10
    eval_every: int = 50
    video_every: int = 250
    eval_envs: int = 512
    output_dir: str = "output/satellite_reference"


@dataclass(frozen=True)
class ExperimentConfig:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    faults: WheelFaultConfig = field(default_factory=WheelFaultConfig)
    orbit: OrbitConfig = field(default_factory=OrbitConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    fdir: FDIRConfig = field(default_factory=FDIRConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    run: RunConfig = field(default_factory=RunConfig)


def default_config() -> ExperimentConfig:
    return ExperimentConfig()


def config_from_dict(data: dict) -> ExperimentConfig:
    estimator_data = data.get("estimator")
    legacy_without_estimator = estimator_data is None
    if estimator_data is not None:
        estimator_data = dict(estimator_data)
        estimator_data.pop("compensate_fixed_latency", None)
        estimator = EstimatorConfig(**estimator_data)
    else:
        estimator = EstimatorConfig(enabled=False)
    return ExperimentConfig(
        physics=PhysicsConfig(**data["physics"]),
        task=TaskConfig(**data["task"]),
        control=ControlConfig(**data["control"]),
        faults=WheelFaultConfig(**data.get("faults", {})),
        orbit=OrbitConfig(**data.get("orbit", {})),
        sensors=SensorConfig(**data.get("sensors", {})),
        estimator=estimator,
        guidance=GuidanceConfig(**data.get("guidance", {})),
        fdir=FDIRConfig(**data.get("fdir", ({"enabled": False} if legacy_without_estimator else {}))),
        supervisor=SupervisorConfig(**data.get("supervisor", ({"enabled": False} if legacy_without_estimator else {}))),
        observation=ObservationConfig(**({
            **data["observation"],
            "wheel_mask_source": (
                "fdir" if data["observation"].get("wheel_mask_source") == "truth"
                else data["observation"].get("wheel_mask_source", "fdir")
            ),
        })),
        reward=RewardConfig(**data["reward"]),
        network=NetworkConfig(**data["network"]),
        ppo=PPOConfig(**data["ppo"]),
        run=RunConfig(**data["run"]),
    )


def validate_config(config: ExperimentConfig) -> None:
    p, t, o, r, c, f, orbit, s, ppo, e, g, fd, sup = (
        config.physics,
        config.task,
        config.observation,
        config.reward,
        config.control,
        config.faults,
        config.orbit,
        config.sensors,
        config.ppo,
        config.estimator,
        config.guidance,
        config.fdir,
        config.supervisor,
    )
    if any(value <= 0.0 for value in p.body_inertia):
        raise ValueError("body_inertia entries must be positive")
    if p.wheel_inertia <= 0.0:
        raise ValueError("wheel_inertia must be positive")
    if p.bearing_friction < 0.0 or p.coulomb_friction_torque < 0.0 or p.stiction_torque < 0.0:
        raise ValueError("wheel friction coefficients must be non-negative")
    if p.stiction_speed_rad_s <= 0.0 or p.motor_time_constant_s <= 0.0:
        raise ValueError("stiction speed and motor time constant must be positive")
    if p.motor_dead_zone_torque < 0.0 or p.motor_command_quantization_torque < 0.0:
        raise ValueError("motor dead-zone/quantization must be non-negative")
    if p.motor_torque_scale <= 0.0:
        raise ValueError("motor_torque_scale must be positive")
    if p.max_wheel_speed <= 0.0 or p.max_motor_torque <= 0.0:
        raise ValueError("wheel and motor limits must be positive")
    if p.motor_control_limit <= 0.0 or p.motor_control_limit > p.max_motor_torque:
        raise ValueError("motor_control_limit must be in (0, max_motor_torque]")
    if any(value <= 0.0 for value in p.body_torque_limit):
        raise ValueError("body_torque_limit entries must be positive")
    _ = p.substeps
    episode_ratio = t.episode_seconds / p.control_dt
    if abs(episode_ratio - round(episode_ratio)) > 1.0e-9:
        raise ValueError("episode_seconds must be an integer multiple of control_dt")
    if t.success_dwell_seconds <= 0.0 or t.success_dwell_seconds > t.episode_seconds:
        raise ValueError("success_dwell_seconds must be in (0, episode_seconds]")
    if o.omega_scale <= 0.0 or r.rate_scale <= 0.0:
        raise ValueError("observation and reward rate scales must be positive")
    if c.mode not in ("residual_pd", "direct", "motor_direct"):
        raise ValueError(f"unsupported control mode: {c.mode}")
    if not 0.0 <= c.residual_scale <= 1.0:
        raise ValueError("residual_scale must be in [0, 1]")
    if o.wheel_mask_source not in ("fdir", "unknown"):
        raise ValueError("wheel_mask_source must be 'fdir' or 'unknown'")
    if g.mode not in (
        "inertial_hold", "nadir_lvlh", "ground_target", "ground_station",
        "sun_pointing", "scheduled_slew",
    ):
        raise ValueError(f"unsupported guidance mode: {g.mode}")
    if g.earth_tracking_body_axis not in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
        raise ValueError("unsupported earth_tracking_body_axis")
    if g.sun_pointing_body_axis not in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
        raise ValueError("unsupported sun_pointing_body_axis")
    if g.antenna_half_beamwidth_deg <= 0.0 or g.antenna_half_beamwidth_deg >= 90.0:
        raise ValueError("antenna_half_beamwidth_deg must be in (0, 90)")
    if g.slew_start_seconds < 0.0 or g.slew_max_rate_deg_s <= 0.0 or g.slew_max_accel_deg_s2 <= 0.0:
        raise ValueError("scheduled slew start/rate/acceleration are invalid")
    for name, quat in (("inertial_target_q", g.inertial_target_q), ("slew_start_q", g.slew_start_q), ("slew_end_q", g.slew_end_q)):
        norm2 = sum(value * value for value in quat)
        if norm2 <= 1.0e-12:
            raise ValueError(f"{name} must be non-zero")
    if orbit.earth_rotation_rate_rad_s < 0.0:
        raise ValueError("earth_rotation_rate_rad_s must be non-negative")
    if not 0.0 < fd.wheel_authority_ewma_alpha <= 1.0 or not 0.0 < fd.nis_ewma_alpha <= 1.0:
        raise ValueError("FDIR EWMA alphas must be in (0, 1]")
    if not 0.0 <= fd.wheel_failed_authority < fd.wheel_degraded_authority <= fd.wheel_recovery_authority <= 1.0:
        raise ValueError("FDIR wheel authority thresholds are inconsistent")
    if min(fd.wheel_suspect_samples, fd.wheel_fail_samples, fd.wheel_recovery_samples, fd.innovation_suspect_samples, fd.innovation_fail_samples, fd.sensor_recovery_samples) < 1:
        raise ValueError("FDIR persistence counters must be positive")
    if sup.minimum_control_wheels < 1 or sup.minimum_control_wheels > 4:
        raise ValueError("minimum_control_wheels must be in [1,4]")
    if not 0.0 < sup.wheel_speed_degraded_fraction < sup.wheel_speed_critical_fraction <= 1.0:
        raise ValueError("supervisor wheel-speed fractions are inconsistent")
    if min(sup.enter_degraded_dwell_seconds, sup.enter_safe_dwell_seconds, sup.recovery_dwell_seconds, sup.acquisition_dwell_seconds) < 0.0:
        raise ValueError("supervisor dwell times must be non-negative")
    if ppo.num_envs < 1 or ppo.num_minibatches < 1 or ppo.update_epochs < 1:
        raise ValueError("PPO counts must be positive")


    if orbit.earth_mu_m3_s2 <= 0.0 or orbit.earth_radius_m <= 0.0:
        raise ValueError("Earth gravitational parameter and radius must be positive")
    if orbit.altitude_m <= 0.0:
        raise ValueError("orbit altitude must be positive")
    if orbit.magnetic_equator_field_t <= 0.0:
        raise ValueError("magnetic_equator_field_t must be positive")
    if sum(value * value for value in orbit.sun_direction_eci) <= 0.0:
        raise ValueError("sun_direction_eci must be non-zero")


    def validate_sensor_clock(name: str, rate_hz: float, latency_seconds: float) -> None:
        if rate_hz <= 0.0:
            raise ValueError(f"{name} rate must be positive")
        period_ratio = 1.0 / (rate_hz * p.physics_dt)
        period_steps = int(round(period_ratio))
        if period_steps < 1 or abs(period_ratio - period_steps) > 1.0e-9:
            raise ValueError(
                f"{name} rate must divide the physics rate exactly; "
                f"got rate_hz={rate_hz}, physics_dt={p.physics_dt}"
            )
        if latency_seconds < 0.0:
            raise ValueError(f"{name} latency must be non-negative")
        latency_ratio = latency_seconds / p.physics_dt
        if abs(latency_ratio - round(latency_ratio)) > 1.0e-9:
            raise ValueError(
                f"{name} latency must be an integer multiple of physics_dt"
            )

    validate_sensor_clock(
        "gyro", s.gyro_rate_hz, s.gyro_latency_seconds
    )
    validate_sensor_clock(
        "wheel tachometer",
        s.wheel_tach_rate_hz,
        s.wheel_tach_latency_seconds,
    )
    validate_sensor_clock(
        "star tracker",
        s.star_tracker_rate_hz,
        s.star_tracker_latency_seconds,
    )
    validate_sensor_clock(
        "magnetometer",
        s.magnetometer_rate_hz,
        s.magnetometer_latency_seconds,
    )
    validate_sensor_clock(
        "Sun sensor",
        s.sun_sensor_rate_hz,
        s.sun_sensor_latency_seconds,
    )
    validate_sensor_clock(
        "GNSS", s.gnss_rate_hz, s.gnss_latency_seconds
    )

    if f.mode not in (
        "none",
        "permanent",
        "fixed_interval",
        "random_interval",
        "stochastic",
    ):
        raise ValueError(f"unsupported fault mode: {f.mode}")
    if f.wheel_selection not in ("fixed", "random"):
        raise ValueError(f"unsupported wheel_selection: {f.wheel_selection}")
    if not 0 <= f.wheel_index < 4:
        raise ValueError("wheel_index must be in [0, 3]")
    if f.start_seconds < 0.0 or f.duration_seconds < 0.0:
        raise ValueError("fault start/duration must be non-negative")
    if f.random_start_min_seconds < 0.0:
        raise ValueError("random_start_min_seconds must be non-negative")
    if f.random_start_max_seconds < f.random_start_min_seconds:
        raise ValueError("random fault start range is invalid")
    if f.random_duration_min_seconds < 0.0:
        raise ValueError("random_duration_min_seconds must be non-negative")
    if f.random_duration_max_seconds < f.random_duration_min_seconds:
        raise ValueError("random fault duration range is invalid")
    if f.failure_rate_per_second < 0.0 or f.recovery_rate_per_second < 0.0:
        raise ValueError("stochastic fault rates must be non-negative")
    if not 0.0 <= f.fault_torque_fraction <= 1.0:
        raise ValueError("fault_torque_fraction must be in [0, 1]")

    nonnegative_estimator_values = (
        e.initial_gyro_bias_sigma_rad_s,
        e.gyro_process_noise_rad_s_sqrt_hz,
        e.gyro_bias_random_walk_rad_s2_sqrt_hz,
    )
    if any(value < 0.0 for value in nonnegative_estimator_values):
        raise ValueError("estimator process noise and bias sigma must be non-negative")

    if e.fixed_lag_history_seconds <= 0.0:
        raise ValueError("fixed_lag_history_seconds must be positive")
    max_sensor_latency = max(
        s.gyro_latency_seconds, s.star_tracker_latency_seconds,
        s.magnetometer_latency_seconds, s.sun_sensor_latency_seconds
    )
    if e.use_fixed_lag_replay and e.fixed_lag_history_seconds < max_sensor_latency + 2.0 * p.physics_dt:
        raise ValueError("fixed_lag_history_seconds must exceed max attitude/gyro latency by at least two physics steps")

    probability_fields = (
        s.gyro_packet_loss_probability, s.wheel_tach_packet_loss_probability,
        s.star_tracker_packet_loss_probability, s.star_tracker_outlier_probability,
        s.magnetometer_packet_loss_probability, s.sun_sensor_packet_loss_probability,
        s.gnss_packet_loss_probability,
    )
    if any(value < 0.0 or value > 1.0 for value in probability_fields):
        raise ValueError("sensor loss/outlier probabilities must be in [0, 1]")

    positive_estimator_values = (
        e.initial_attitude_sigma_deg,
        e.star_tracker_noise_std_deg,
        e.star_hard_acquisition_sigma_deg,
        e.magnetometer_noise_std_deg,
        e.sun_sensor_noise_std_deg,
        e.star_tracker_nis_gate,
        e.magnetometer_nis_gate,
        e.sun_sensor_nis_gate,
        e.covariance_floor,
        e.innovation_regularization,
    )
    if any(value <= 0.0 for value in positive_estimator_values):
        raise ValueError("estimator covariance, measurement noise, and gates must be positive")
