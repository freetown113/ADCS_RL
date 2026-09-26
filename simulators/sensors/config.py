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
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    run: RunConfig = field(default_factory=RunConfig)


def default_config() -> ExperimentConfig:
    return ExperimentConfig()


def validate_config(config: ExperimentConfig) -> None:
    """Raises ValueError for inconsistent or non-physical settings."""
    f = config.faults,

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
