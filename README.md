
# Milestone A.0 — initial implementation

This project aims to train a reinforcement learning agent to control satellite’s Attitude Determination and Control System (ADCS). The system uses three orthogonal reaction wheels to provide full three-axis control, while a fourth wheel is added as a backup, creating a redundant configuration.

The project focuses on the following key elements:

 - Reaction Wheels (RWs): Electric motors equipped with flywheels that accelerate or decelerate to generate precise torque through angular momentum exchange.
 - Three-Axis Stabilization: An active attitude-control method enabled by the reaction wheel configuration.
 - Redundant Configuration: A 3+1 setup, also known as a pyramid architecture, in which the fourth wheel operates as part of the regular control system, reducing the load on the three primary wheels. If one of the primary wheels fails, the fourth wheel can take over its function, ensuring system redundancy.

## Current observation

The existing project does not expose a raw quaternion. Its base observation is:

```text
attitude error vector (3) + body rate (3) + wheel speeds (4)
```

## Rate hierarchy

Default configuration:

```text
physics:          100 Hz (`physics_dt = 0.01`)
control/policy:    20 Hz (`control_dt = 0.05`)
```

One environment step  holds the selected action for five physics substeps.
Each physics substep now also advances the sensor histories, sample clocks and
fixed-latency delivery logic.

## Controller/plant separation

The body-torque allocator was split into two stages:

1. `allocate_body_torque_command(...)` computes motor commands from desired body
   torque and **tachometer measurements**.
2. `motor_torque_to_net_rotor_torque(...)` applies those commands to the **true**
   rotor state, including true bearing friction, torque limits and speed limits.

Thus stale or damaged tachometer telemetry can later create a genuine
controller/plant mismatch without corrupting the physical truth model.

The residual/classical PD damping term now uses `state.sensors.gyro` instead of
`state.physical.omega`. Attitude remains truth-derived until the MEKF milestone.

## New configuration

```python
SensorConfig(
    gyro_rate_hz=100.0,
    gyro_latency_seconds=0.0,
    wheel_tach_rate_hz=100.0,
    wheel_tach_latency_seconds=0.0,
)
```

Rates must divide the physics rate exactly. Latencies must be integer multiples
of `physics_dt`; this keeps schedules and history-array shapes static under JIT.

Example:

```python
from dataclasses import replace
from your_package.config import SensorConfig, default_config

config = default_config()
config = replace(
    config,
    sensors=SensorConfig(
        gyro_rate_hz=20.0,
        gyro_latency_seconds=0.02,
        wheel_tach_rate_hz=50.0,
        wheel_tach_latency_seconds=0.01,
    ),
)
```

Equivalent training CLI flags were added:

```text
--gyro-rate-hz
--gyro-latency-seconds
--wheel-tach-rate-hz
--wheel-tach-latency-seconds
```

## Sensor state and diagnostics

`EnvState.sensors` contains:

- latest gyro and wheel-speed measurements;
- validity flags;
- original sample-step timestamps;
- fixed-length internal truth histories used only to implement latency.

`StepInfo` exposes delivered measurements and their ages for diagnostics.
Evaluation trajectories also retain these fields and report mean/max sample age.

Run the dependency-light tests with:

```bash
python -m adcs_sensors_a2.sensor_diagnostics
```

Expected result:

```text
PASS test_perfect_sensors_match_truth
PASS test_low_rate_sensor_sample_and_hold
PASS test_fixed_sensor_latency
PASS test_pd_uses_gyro_measurement
PASS test_allocator_uses_tachometer_measurement
PASS test_jitted_batched_step
All sensor diagnostics passed.
```

The default perfect 100 Hz configuration was also compared with the original
project for repeated PD-controlled steps. Commands, observations, physical
states and rewards matched exactly.

## Deliberately not implemented yet

- gyro noise, bias or bias random walk;
- tachometer quantization/noise/faults;
- star tracker, magnetometer or Sun sensor;
- attitude estimator;
- sensor-validity fields in the policy observation.

The next milestone should add stochastic gyro/tachometer models while preserving
this same pure-functional state architecture. Then the absolute attitude sensors
and MEKF can replace the remaining truth-derived attitude-error vector.









This project aims to train a reinforcement learning agent capable of fully controlling a satellite’s Attitude Determination and Control System (ADCS). The system uses three orthogonal reaction wheels to provide full three-axis control, while a fourth wheel is added as a backup, creating a redundant configuration.

The project focuses on the following key elements:

 - Reaction Wheels (RWs): Electric motors equipped with flywheels that accelerate or decelerate to generate precise torque through angular momentum exchange.
 - Three-Axis Stabilization: An active attitude-control method enabled by the reaction wheel configuration.
 - Redundant Configuration: A 3+1 setup, also known as a pyramid architecture, in which the fourth wheel operates as part of the regular control system, reducing the load on the three primary wheels. If one of the primary wheels fails, the fourth wheel can take over its function, ensuring system redundancy.

https://www.hanspeterschaub.info/PapersPrivate/Hogan2015a.pdf

https://control.asu.edu/Classes/MAE462/462Lecture15.pdf





“I developed an autonomous fault-tolerant spacecraft attitude-control stack, starting with reinforcement learning in simulation and progressively introducing imitation learning, sensor-based state estimation, actuator uncertainty, domain randomization, fault detection and recovery, independent simulation verification, embedded neural-network quantization, deterministic Rust inference, distributed communication and flight-software architecture.”