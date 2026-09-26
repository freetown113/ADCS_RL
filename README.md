
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

