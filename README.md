# Milestone A.3 — Six-state MEKF integration

Previous version is extended with a multiplicative attitude EKF(MEKF). 
The estimator is part of `EnvState`, advances inside every physics
substep, and is compatible with batching, `jax.jit`, `jax.lax.scan`, PPO, and
imitation-learning rollouts.

Previously the star tracker, magnetometer, Sun sensor, and GNSS were generated and
logged, but they did part of observation or the residual PD controller.
Only the gyro and wheel tachometers affected control. Therefore, with identical
gyro/tachometer settings and identical seeds/actions, changing only attitude
sensor rates, latency, or eclipse must produce identical physical trajectories,
observations, and rewards.Now all sensor settings do affect control because the MEKF consumes them.

## Filter definition

Nominal state:
- body-to-inertial quaternion `q_hat`, order `[w, x, y, z]`;
- additive gyro bias estimate `b_hat`.

Six-dimensional local error state:
```
error_state = [delta_theta_x, delta_theta_y, delta_theta_z,
               delta_bias_x,  delta_bias_y,  delta_bias_z]
```
The implementation uses a right-multiplicative attitude error:
```
q_true = q_hat ⊗ delta_q(delta_theta)
```
The estimator propagates at `physics_dt` using the latest delivered gyro sample:
```
omega_hat = gyro_measurement - b_hat
q_hat(k+1) = q_hat(k) ⊗ exp_quaternion(omega_hat * physics_dt)
```
The covariance is propagated with a first-order discrete state transition and 
multiplicative covariance reset are included for numerical stability.

The filter consumes every newly delivered packet exactly once.

### Star tracker

The star tracker supplies a body-to-inertial quaternion. The residual is the
small local rotation between the estimated and measured quaternions. A 3-D
attitude update is applied.

### Magnetometer

The magnetometer supplies the measured body-frame magnetic vector. GNSS position
is passed through the centered-dipole model to construct the inertial reference
vector. Both vectors are normalized before the update, so the update primarily
uses direction rather than field magnitude.

### Sun sensor

The Sun sensor supplies a body-frame Sun direction. It is fused against the
configured inertial Sun direction. When eclipse handling is enabled, an invalid
Sun packet is consumed but does not update the filter.

### GNSS

GNSS is not itself an attitude update. Its position is used to calculate the
magnetic inertial reference. GNSS time and velocity remain available for later
navigation and reference-frame work.

## Fixed-latency compensation

When `compensate_fixed_latency=True`, delayed measurements are extrapolated from
their sample timestamp to delivery time using the current bias-corrected gyro rate:
- star-tracker quaternion: propagated forward;
- body-frame magnetic/Sun vectors: rotated into the estimated current body frame.
This is a first-order approximation.

## Observation and controller paths

With the estimator enabled, the actor receives:
```
3  target-relative estimated attitude vector
3  estimated body rate
4  measured wheel speeds
N  previous action
4  wheel motor availability mask
```

The observation size is unchanged. Existing network architecture remains valid,
but a policy should be retrained because the meaning of the attitude/rate fields
has changed from privileged/partially privileged values to estimated values.

The residual PD controller now also uses `q_hat` and `omega_hat`. Therefore the
policy and classical controller share the same estimated state.

Truth remains available only for:

- rotational/orbit physics;
- sensor generation;
- reward and success calculation;
- estimator-error diagnostics and plots.

## Delayed star-tracker acquisition

Star traker initialization caused an erroneous behavior:
With zero star-tracker latency, the star packet was valid at environment reset and the MEKF initialized directly from it, but with any positive star-tracker latency, reset started from the identity quaternion. Tthe first delayed star quaternion was then treated as an ordinary MEKF update and subjected to the NIS gate, so with initial attitude errors higher than 53 degree the valid star solution was rejected.

It was fixed with adding `EstimatorConfig.hard_acquire_first_star_tracker=True`, that makes that when the first valid star quaternion arrives and no star solution has previously been processed, the estimator:
- applies the existing fixed-latency quaternion compensation;
- uses that quaternion as a **global attitude acquisition/reset**;
- keeps covariance conservative;
- marks the first star solution accepted;
- returns to ordinary NIS-gated local MEKF corrections for all later star packets.
This does not bypass NIS gating for subsequent star-tracker outliers.

However current solution still extrapolates a delayed star quaternion to the present using one current bias-corrected gyro rate over the whole latency interval. That is only first-order compensation. Under rapid motor-direct challenging maneuvers, 80 ms can span appreciable angular acceleration, so a residual estimate jump/oscillation can remain.


# Milestone A.2 — clean asynchronous attitude/navigation sensors

Extends projects with sensors readings as observation, updates orbit logic. 
The new star tracker, magnetometer, Sun sensor, and GNSS are not yet fed 
directly to the policy. Because `physics_dt=0.01`, every latency above is an 
integer multiple of 10 ms.

## Added reference environment

`orbit.py` adds a decoupled circular LEO truth state:

- 500 km default altitude;
- 51.6° default inclination;
- exact circular-orbit rotation at every physics substep;
- centered tilted magnetic dipole;
- fixed inertial Sun direction over one short episode;
- geometric Earth-eclipse truth.

The orbit is used only to generate sensor/reference truth. It does not apply
translational forces to the rotational plant.

## Added sensors

All sensors remain pure JAX state carried inside `EnvState` and are updated inside
the existing physics `lax.scan`.

| Sensor | Default rate | Measurement |
|---|---:|---|
| Gyroscope | 100 Hz | body angular rate, rad/s |
| Wheel tachometers | 100 Hz | four rotor speeds, rad/s |
| Star tracker | 2 Hz | body-to-inertial quaternion `[w,x,y,z]` |
| Magnetometer | 10 Hz | magnetic field in body coordinates, tesla |
| Sun sensor | 10 Hz | unit Sun direction in body coordinates |
| GNSS | 1 Hz | ECI position, ECI velocity, simulation/navigation time |

Every sensor supports independent fixed latency, sample-and-hold, validity, and
sample age. Rates must divide the physics rate exactly and latencies must be
integer multiples of `physics_dt`.



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