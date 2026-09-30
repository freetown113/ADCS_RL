# Milestone B.1 Fault tolerance closer to reality

### 1. Timestamp-correct MEKF

Previous implementation compensated fixed latency by extrapolating a delayed 
measurement with one current angular-rate value. That shortcut was replaced with 
a bounded fixed-lag replay buffer:

- Every piece of sensor data gets a precise timestamp of when it was actually taken
- If a packet arrives late, the system travels back in time to that exact timestamp
- It fixes the historical satellits position at that past moment
- It "replays" all the recorded gyroscope movements from that past moment back to the present day

Star-tracker residuals use the quaternion log/rotation vector rather than 
`2*q_vec`, and unit-vector measurements are updated in a two-dimensional tangent 
plane instead of applying three independent attitude constraints to a normalized 
vector in previous version.

The first valid global star solution can succeede attitude acquisition even in hard conditions
Insterad of rely on first available data from star tracker, when the satellite finds its 
orientation for the very first time, the estimator,make a guess of correct measurements but keep a 
margin of error wide. First guess is doubtful, if the next measurement contradicts the first one, 
the estimator accepts the new data and fix itself. If the star tracker doesn't see any stars at all, 
the estimator falls back to TRIAD method using the Earth's magnetic field and the Sun to get a 
rough baseline orientation.


### 2. Sensors now inject actual errors

The previous EKF noise values were mainly covariance assumptions, adding errors
to telemetry itself:

- gyro: white noise density, initial bias, bias random walk, scale factor,
  misalignment, quantization, clipping, packet loss
- wheel tachometers: noise, scale factor, quantization, packet loss/staleness
- star tracker: alignment error, measurement noise, outliers, packet loss,
  lost-in-space acquisition timer, acquisition/tracking angular-rate limits,
  Sun and Earth-limb keep-out, loss/reacquisition of lock
- magnetometer: bias, scale factor, misalignment, white noise, quantization,
  clipping, packet loss
- Sun sensor: angular noise, misalignment, eclipse invalidity, packet loss
- GNSS: position/velocity noise and packet loss

Sensor calibration truth and RNG state stay private inside `SensorState` and are
never exposed to the actor.

### 3. Reaction-wheel plant is no longer an ideal torque source

Default wheel/body scale is now consistent with the ~50 kg body:

```python
body_inertia = (2.0, 2.0, 2.5)       # kg m^2
wheel_inertia = 9.0e-4               # kg m^2
max_wheel_speed = 628.0              # rad/s (~6000 rpm)
max_motor_torque = 0.05              # N m hardware clamp
motor_control_limit = 0.03            # N m exposed to motor-direct policy
bearing_friction = 3.0e-6            # N m / (rad/s), viscous term
coulomb_friction_torque = 2.0e-4     # N m
stiction_torque = 3.0e-4             # N m
```

The plant also includes motor torque lag, dead zone, command quantization,
Coulomb friction, local stiction/breakaway behavior, speed limiting, and partial
wheel authority. `fault_torque_fraction=0.5`, for example, models 50% retained
motor authority instead of only total failure.

### 4. Moving-reference guidance

A new `guidance.py` separates reference generation from estimation and control.
Supported modes are:

```python
guidance.mode = "inertial_hold"   # original behavior
guidance.mode = "nadir_lvlh"      # body +Z nadir, +X horizontal velocity
```

For a moving LVLH target the desired angular rate is non-zero. The actor
observation, PD teacher, and reward therefore use target-relative body-rate error
rather than absolute body rate. In inertial-hold mode the desired rate is zero.

Reset distributions are target-relative as well: `fixed_angle_deg` / `max_initial_angle_deg` describe attitude error from the current guidance frame, and `max_initial_rate` describes tracking-rate error around the guidance frame angular velocity.

## Privileged-information boundary

Two compatibility paths intentionally remain explicit:

- `estimator.enabled=False` restores the pre-MEKF truth-attitude path and should
  be used only for legacy regression/debugging;
- `observation.wheel_mask_source="truth"` exposes real actuator availability.
  From now in project use `"unknown"` (or omit the mask) until
  the future FDIR module supplies an estimated health vector.

Body-torque allocation also currently receives the true authority mask.

## Deliberate remaining model boundaries

These are not TODOs, but reither they are separate fidelity modules still needed for a
mission-grade simulator:

- circular analytic LEO instead of a force-model orbit propagator;
- centered tilted-dipole magnetic field instead of IGRF/WMM;
- constant inertial Sun direction over an episode instead of epoch-aware solar
  ephemerides;
- no gravity-gradient, aerodynamic, solar-radiation-pressure, residual-magnetic,
  or thruster disturbance torque model yet;
- one abstract star-tracker head rather than a detailed optical/catalog/image
  processing model;
- Sun sensor returns a fused body Sun vector rather than simulating individual
  photodiodes/FOV/albedo;
- magnetometer does not yet include spacecraft-current/self-field contamination;
- GNSS is a PVT measurement model, not a constellation/link/DOP receiver model;
- sensor latency is deterministic and quantized to `physics_dt`; timestamp jitter
  and variable processing latency are not yet modeled;
- reaction-wheel imbalance/jitter, wheel-axis mounting error, thermal/supply
  dependence, and per-wheel calibration dispersion are not yet modeled;
- mission mode management (DETUMBLE -> ACQUIRE -> FINE_POINTING -> SAFE/DEGRADED)
  and detected health/reconfiguration belong to the planned supervisory GNC/FDIR
  module.



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


## Curriculum learning
Previous implementations were simple enough for the agent to learn the underlying mechanics of attitude control. Although the task may not appear particularly difficult, the agent still struggles to learn direct control of the reaction-wheel motors. This is where interchangeable control modes and imitation learning become useful.
I started by collecting a dataset generated by a PD controller and used it to train a separate model to learn the distribution of motor torques. I then used the pretrained weights to initialize the actor and began reinforcement learning under relatively easy conditions:
 - 5–10° fixed angular offset from the target, with no initial disturbance
Once the agent adapted to these conditions, I saved the approximator parameters and increased the difficulty:
 - 30–45° fixed angular offset, with no initial disturbance
After further adaptation, I increased the difficulty again:
 - 45–60° random angular offset, with no initial disturbance
The next stage introduced a small initial disturbance:
 - 60–90° random angular offset, with a small initial disturbance of 0.05 rad/s
This process continues until the agent can reliably handle the most challenging conditions:
 - Up to 180° random angular offset, with a strong initial disturbance of 0.2 rad/s

This is not the only curriculum that works, and I believe the training process could probably be accelerated. However, this particular schedule has proven to be reliable.

When moving between curriculum stages, it is not always necessary to restart training from scratch. However, some architectural changes can make direct adaptation from previously learned parameters more difficult.


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