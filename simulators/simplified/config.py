from dataclasses import dataclass, field
from typing import Literal, Tuple


ResetMode = Literal["fixed", "cone"]
ControlMode = Literal["residual_pd", "direct", "motor_direct"]
WheelSelection = Literal["fixed", "random"]
FaultMode = Literal[
    "none",
    "permanent",
    "fixed_interval",
    "random_interval",
    "stochastic",
]


@dataclass(frozen=True)
class PhysicsConfig:
    body_inertia: Tuple[float, float, float] = (2.0, 2.0, 2.5)
    wheel_inertia: float = 9.0e-4
    bearing_friction: float = 3.0e-6
    max_wheel_speed: float = 628.0
    max_motor_torque: float = 0.05

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
    fixed_angle_deg: float = 120.0
    max_initial_angle_deg: float = 180.0
    max_initial_rate: float = 0.1

    episode_seconds: float = 180.0
    success_angle_deg: float = 1.0
    success_rate: float = 0.01
    success_dwell_seconds: float = 1.0


@dataclass(frozen=True)
class ControlConfig:
    """Maps the learned action to physical actuation.
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
    mode: FaultMode = "stochastic"
    wheel_selection: WheelSelection = "random"
    wheel_index: int = 0

    start_seconds: float = 90.0
    duration_seconds: float = 2.0

    random_start_min_seconds: float = 0.5
    random_start_max_seconds: float = 5.0
    random_duration_min_seconds: float = 0.5
    random_duration_max_seconds: float = 3.0

    failure_rate_per_second: float = 0.10
    recovery_rate_per_second: float = 0.50


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
    total_updates: int = 50_000
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
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    run: RunConfig = field(default_factory=RunConfig)


def default_config() -> ExperimentConfig:
    return ExperimentConfig()
