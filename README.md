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