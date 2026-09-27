# Milestone A.1 — gyro and wheel-tachometer integration

This version adapts the sensor layer to the project's existing JAX,
batched, multi-rate environment. All sensor state is part of `EnvState` 
and is advanced inside the same `jax.lax.scan` that advances the physics.

## Current observation

Previously the project does not expose a raw quaternion. Its base observation is:

```
attitude error vector (3) + body rate (3) + wheel speeds (4)
```

and optionally previous action and the four-wheel motor mask.

After this change:

```
attitude error vector: still truth-derived temporarily
body rate:             delivered gyro measurement
wheel speeds:          delivered tachometer measurements
```

The observation shape is unchanged, so existing checkpoints remain compatible
with the default perfect sensor settings.

## Rate hierarchy

Default configuration:

```
physics:          100 Hz (`physics_dt = 0.01`)
control/policy:    20 Hz (`control_dt = 0.05`)
gyro:             100 Hz
wheel tachometers:100 Hz
```

One environment step still holds the selected action for five physics substeps.
Each physics substep now also advances the sensor histories, sample clocks and
fixed-latency delivery logic.

Asynchronous clean sensor clocks and fixed transport latency. Rates must be 
integer divisors of the physics update rate. Latencies must be integer multiples 
of ``physics_dt`` so array shapes and schedules remain static during JAX compilation.

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


## Orbit

Simple LEO reference environment for attitude sensors.

The ADCS plant remains rotational only. It adds a decoupled circular
orbit used to generate navigation truth and inertial reference vectors for the
magnetometer, Sun sensor, and GNSS receiver. These is clean deterministic 
reference models, not yet high-fidelity orbit or space-weather models.

Earth-to-Sun unit direction in ECI, treated as constant over one episode.






# Milestone A.0 — initial implementation

This project aims to train a reinforcement learning agent to control satellite’s Attitude Determination and Control System (ADCS). The system uses three orthogonal reaction wheels to provide full three-axis control, while a fourth wheel is added as a backup, creating a redundant configuration.

The project focuses on the following key elements:

 - Reaction Wheels (RWs): Electric motors equipped with flywheels that accelerate or decelerate to generate precise torque through angular momentum exchange.
 - Three-Axis Stabilization: An active attitude-control method enabled by the reaction wheel configuration.
 - Redundant Configuration: A 3+1 setup, also known as a pyramid architecture, in which the fourth wheel operates as part of the regular control system, reducing the load on the three primary wheels. If one of the primary wheels fails, the fourth wheel can take over its function, ensuring system redundancy.

## Current observation

The existing project does not expose a raw quaternion. Its base observation is:

```
attitude error vector (3) + body rate (3) + wheel speeds (4)
```

## Rate hierarchy

Default configuration:

```
physics:          100 Hz (`physics_dt = 0.01`)
control/policy:    20 Hz (`control_dt = 0.05`)
```

One environment step  holds the selected action for five physics substeps.

## Wheel fault

At most one reaction-wheel motor is unavailable at a time.

A fault means *motor torque is unavailable*. The rotor remains part of the
spacecraft: its stored angular momentum and bearing friction remain in the
physics. This models loss of command/power much better than deleting a wheel.

```
Modes:
    none            no failures.
    permanent       selected wheel fails at start_seconds until episode end.
    fixed_interval  selected wheel fails for a fixed time window.
    random_interval start/duration are sampled independently per environment.
    stochastic      healthy<->failed transitions follow per-second hazard rates.
```

``wheel_selection='random'`` samples the failed wheel independently for every
environment (and every new failure event in stochastic mode).

Hazard rates are converted to per-control-step probabilities 
``p = 1 - exp(-rate * control_dt)``, making them dt-consistent.


## Imitation learning

Teacher target for supervised pretraining of a motor-direct policy.

By default the teacher is PD. Passing the body action from a
trained ``direct`` policy can give a better task-specific teacher.