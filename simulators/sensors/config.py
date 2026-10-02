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


@dataclass(frozen=True)
class PhysicsConfig:
    """Physical parameters and integration settings."""

    body_inertia: Tuple[float, float, float] = (2.0, 2.0, 2.5)
    wheel_inertia: float = 0.05
    bearing_friction: float = 1.0e-3
    max_wheel_speed: float = 628.0
    max_motor_torque: float = 0.5

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
    fixed_axis: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    fixed_angle_deg: float = 60.0
    max_initial_angle_deg: float = 120.0
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
      none            all 4 wheels work during the whole episode
      permanent       selected wheel is unavilable during the whole episode
      fixed_interval  selected wheel out of order for a fixed time interval
      random_interval same as previos but interval start/end set independently per environment.
      stochastic      healthy<->failed intervals follow stochastic rates.
    """

    mode: FaultMode = "stochastic"
    wheel_selection: WheelSelection = "random"
    wheel_index: int = 0

    start_seconds: float = 2.0
    duration_seconds: float = 4.0

    random_start_min_seconds: float = 0.5
    random_start_max_seconds: float = 5.0
    random_duration_min_seconds: float = 0.5
    random_duration_max_seconds: float = 3.0

    failure_rate_per_second: float = 0.10
    recovery_rate_per_second: float = 0.50




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

    sun_direction_eci: Tuple[float, float, float] = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class SensorConfig:

    gyro_rate_hz: float = 100.0
    gyro_latency_seconds: float = 0.2
    wheel_tach_rate_hz: float = 100.0
    wheel_tach_latency_seconds: float = 0.2

    star_tracker_rate_hz: float = 2.0
    star_tracker_latency_seconds: float = 0.08
    magnetometer_rate_hz: float = 10.0
    magnetometer_latency_seconds: float = 0.2
    sun_sensor_rate_hz: float = 10.0
    sun_sensor_latency_seconds: float = 0.1
    gnss_rate_hz: float = 1.0
    gnss_latency_seconds: float = 0.0

    sun_sensor_eclipse_enabled: bool = False


@dataclass(frozen=True)
class EstimatorConfig:
    """Six-state quaternion error-state EKF configuration.

    The filter estimates body-to-inertial attitude and additive gyro bias.
    Measurement standard deviations are tuning values for the clean A.3 sensor
    model; Milestone B will make them consistent with injected sensor errors.
    """

    enabled: bool = True
    initialize_from_star_tracker: bool = True
    hard_acquire_first_star_tracker: bool = True
    compensate_fixed_latency: bool = True

    use_star_tracker: bool = True
    use_magnetometer: bool = True
    use_sun_sensor: bool = True

    initial_attitude_sigma_deg: float = 10.0
    initial_gyro_bias_sigma_rad_s: float = 0.02

    gyro_process_noise_rad_s_sqrt_hz: float = 2.0e-4
    gyro_bias_random_walk_rad_s2_sqrt_hz: float = 1.0e-6

    star_tracker_noise_std_deg: float = 0.02
    magnetometer_noise_std_deg: float = 0.50
    sun_sensor_noise_std_deg: float = 0.50

    star_tracker_nis_gate: float = 25.0
    magnetometer_nis_gate: float = 25.0
    sun_sensor_nis_gate: float = 25.0

    covariance_floor: float = 1.0e-12
    innovation_regularization: float = 1.0e-10


@dataclass(frozen=True)
class ObservationConfig:
    omega_scale: float = 0.20
    include_previous_action: bool = True
    include_wheel_mask: bool = True


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
    hidden_sizes: Tuple = (128, 128)
    activation: str = "tanh"
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
    seed: int = 1492
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
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    run: RunConfig = field(default_factory=RunConfig)


def default_config() -> ExperimentConfig:
    return ExperimentConfig()

def config_from_dict(data: dict) -> ExperimentConfig:
    """Reconstructs a config saved in a checkpoint.

    Older A.1/A.2 checkpoints did not contain an ``estimator`` section. They
    are loaded with the estimator disabled so evaluation preserves the original
    observation semantics. New A.3 checkpoints store the estimator config
    explicitly and therefore restore it exactly.
    """
    estimator_data = data.get("estimator")
    estimator = (
        EstimatorConfig(**estimator_data)
        if estimator_data is not None
        else EstimatorConfig(enabled=False)
    )
    return ExperimentConfig(
        physics=PhysicsConfig(**data["physics"]),
        task=TaskConfig(**data["task"]),
        control=ControlConfig(**data["control"]),
        faults=WheelFaultConfig(**data.get("faults", {})),
        orbit=OrbitConfig(**data.get("orbit", {})),
        sensors=SensorConfig(**data.get("sensors", {})),
        estimator=estimator,
        observation=ObservationConfig(**data["observation"]),
        reward=RewardConfig(**data["reward"]),
        network=NetworkConfig(**data["network"]),
        ppo=PPOConfig(**data["ppo"]),
        run=RunConfig(**data["run"]),
    )


def validate_config(config: ExperimentConfig) -> None:
    """Raises ValueError for inconsistent or non-physical settings."""
    p, t, o, r, c, f, orbit, s, ppo, e = (
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
    )
    if any(value <= 0.0 for value in p.body_inertia):
        raise ValueError("body_inertia entries must be positive")
    if p.wheel_inertia <= 0.0:
        raise ValueError("wheel_inertia must be positive")
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

    nonnegative_estimator_values = (
        e.initial_gyro_bias_sigma_rad_s,
        e.gyro_process_noise_rad_s_sqrt_hz,
        e.gyro_bias_random_walk_rad_s2_sqrt_hz,
    )
    if any(value < 0.0 for value in nonnegative_estimator_values):
        raise ValueError("estimator process noise and bias sigma must be non-negative")

    positive_estimator_values = (
        e.initial_attitude_sigma_deg,
        e.star_tracker_noise_std_deg,
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
