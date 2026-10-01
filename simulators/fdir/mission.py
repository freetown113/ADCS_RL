from dataclasses import dataclass
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from simulators.fdir.config import GuidanceConfig, MissionConfig, OrbitConfig, PhysicsConfig, TaskConfig
from simulators.fdir.guidance import GuidanceTarget, earth_target_elevation_rad, earth_tracking_target, fallback_target
from simulators.fdir.math3d import canonicalize_quat, quat_conjugate, quat_multiply, quat_normalize, quat_to_rotation_vector, rotate_body_to_inertial, rotation_vector_to_quat
from simulators.fdir.orbit import OrbitState, orbit_state_at_time


STANDBY = 0
PRE_SLEW = 1
TRACK = 2
POST_SLEW = 3
NUM_MISSION_PHASES = 4
MISSION_PHASE_NAMES = ("STANDBY", "PRE_SLEW", "TRACK", "POST_SLEW")


class MissionState(NamedTuple):
    phase: jax.Array
    enter_dwell_steps: jax.Array
    exit_dwell_steps: jax.Array
    phase_elapsed_steps: jax.Array
    commanded_q: jax.Array
    commanded_omega_inertial: jax.Array
    slew_rate_rad_s: jax.Array
    target_elevation_rad: jax.Array


@dataclass(frozen=True)
class GroundPassWindow:
    aos_s: float
    tca_s: float
    los_s: float
    max_elevation_deg: float


def mission_phase_one_hot(phase: jax.Array, dtype=jnp.float32) -> jax.Array:
    return jax.nn.one_hot(phase, NUM_MISSION_PHASES, dtype=dtype)


def _steps(seconds: float, physics: PhysicsConfig) -> int:
    return max(1, int(round(seconds / physics.control_dt)))


def _is_ground_mode(guidance: GuidanceConfig) -> bool:
    return guidance.mode in ("ground_target", "ground_station")


def _pursuit_slew(
    current_q: jax.Array,
    current_rate: jax.Array,
    goal_q: jax.Array,
    dt: float,
    max_rate_rad_s: float,
    max_accel_rad_s2: float,
    catch_angle_rad: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Move a commanded quaternion toward a possibly moving goal with bounded rate.
    The rate state is a scalar slew speed.  The instantaneous slew axis follows the
    shortest current-to-goal rotation.  This is deliberately a guidance-reference
    limiter, not a spacecraft dynamic model.
    """
    rel = canonicalize_quat(quat_multiply(quat_conjugate(current_q), goal_q))
    rotvec = quat_to_rotation_vector(rel)
    angle = jnp.linalg.norm(rotvec, axis=-1)
    axis_body = rotvec / (angle[..., None] + 1.0e-12)

    vmax = jnp.asarray(max_rate_rad_s, dtype=current_q.dtype)
    accel = jnp.asarray(max_accel_rad_s2, dtype=current_q.dtype)
    dt_a = jnp.asarray(dt, dtype=current_q.dtype)
    stop_speed = jnp.sqrt(jnp.maximum(0.0, 2.0 * accel * angle))
    requested = jnp.minimum(vmax, stop_speed)
    delta = jnp.clip(requested - current_rate, -accel * dt_a, accel * dt_a)
    next_rate = jnp.clip(current_rate + delta, 0.0, vmax)
    step_angle = jnp.minimum(angle, next_rate * dt_a)
    delta_q = rotation_vector_to_quat(axis_body * step_angle[..., None])
    stepped_q = quat_normalize(quat_multiply(current_q, delta_q))

    caught = angle <= jnp.asarray(catch_angle_rad, angle.dtype)
    next_q = jnp.where(caught[..., None], goal_q, stepped_q)
    axis_i = rotate_body_to_inertial(current_q, axis_body)
    omega_i = axis_i * next_rate[..., None]
    next_rate = jnp.where(caught, 0.0, next_rate)
    omega_i = jnp.where(caught[..., None], jnp.zeros_like(omega_i), omega_i)
    return next_q, omega_i, next_rate, caught


def _future_max_elevation(
    orbit: OrbitState,
    guidance: GuidanceConfig,
    orbit_config: OrbitConfig,
    horizon_s: float,
) -> jax.Array:
    if horizon_s <= 0.0:
        return earth_target_elevation_rad(orbit, guidance, orbit_config)
    fractions = jnp.linspace(0.0, 1.0, 9, dtype=orbit.time_s.dtype)
    offsets = fractions[:, None] * jnp.asarray(horizon_s, orbit.time_s.dtype)
    times = orbit.time_s[None, :] + offsets
    flat = times.reshape((-1,))
    future = orbit_state_at_time(flat, orbit_config)
    elev = earth_target_elevation_rad(future, guidance, orbit_config).reshape(times.shape)
    return jnp.max(elev, axis=0)


def reset_mission_state(
    orbit: OrbitState,
    guidance: GuidanceConfig,
    mission: MissionConfig,
    physics: PhysicsConfig,
    orbit_config: OrbitConfig,
) -> tuple[MissionState, GuidanceTarget]:
    """Initialize mission scheduling for any guidance mode."""
    if _is_ground_mode(guidance) and mission.ground_pass_enabled:
        return reset_ground_mission_state(orbit, guidance, mission, physics, orbit_config)
    from .guidance import guidance_target
    target = guidance_target(orbit, guidance, orbit_config)
    phase = jnp.full((orbit.time_s.shape[0],), STANDBY, dtype=jnp.int32)
    z_i = jnp.zeros_like(phase)
    z_f = jnp.zeros_like(orbit.time_s)
    return MissionState(phase, z_i, z_i, z_i, target.q_body_to_inertial, target.omega_inertial_rad_s, z_f, z_f), target


def update_mission(
    state: MissionState,
    orbit: OrbitState,
    guidance: GuidanceConfig,
    mission: MissionConfig,
    physics: PhysicsConfig,
    orbit_config: OrbitConfig,
) -> tuple[MissionState, GuidanceTarget]:
    if _is_ground_mode(guidance) and mission.ground_pass_enabled:
        return update_ground_mission(state, orbit, guidance, mission, physics, orbit_config)
    from .guidance import guidance_target
    target = guidance_target(orbit, guidance, orbit_config)
    return state._replace(
        phase=jnp.full_like(state.phase, STANDBY),
        phase_elapsed_steps=state.phase_elapsed_steps + 1,
        commanded_q=target.q_body_to_inertial,
        commanded_omega_inertial=target.omega_inertial_rad_s,
        slew_rate_rad_s=jnp.zeros_like(state.slew_rate_rad_s),
    ), target


def reset_ground_mission_state(
    orbit: OrbitState,
    guidance: GuidanceConfig,
    mission: MissionConfig,
    physics: PhysicsConfig,
    orbit_config: OrbitConfig,
) -> tuple[MissionState, GuidanceTarget]:
    ground = earth_tracking_target(orbit, guidance, orbit_config)
    fallback = fallback_target(orbit, guidance)
    elevation = earth_target_elevation_rad(orbit, guidance, orbit_config)
    enter = jnp.deg2rad(jnp.asarray(mission.ground_pass_enter_elevation_deg, elevation.dtype))

    curriculum_track = mission.ground_pass_reset_mode in ("pass_centered", "random_visible")
    if curriculum_track:
        phase = jnp.full((orbit.time_s.shape[0],), TRACK, dtype=jnp.int32)
        command = ground
    else:
        # In full configured-mission mode do not silently assume tracking merely
        # because reset occurs inside a pass. Start from fallback and acquire it.
        phase = jnp.where(elevation >= enter, PRE_SLEW, STANDBY).astype(jnp.int32)
        command = fallback

    zeros_i = jnp.zeros_like(phase)
    zeros_f = jnp.zeros_like(elevation)
    state = MissionState(
        phase=phase,
        enter_dwell_steps=zeros_i,
        exit_dwell_steps=zeros_i,
        phase_elapsed_steps=zeros_i,
        commanded_q=command.q_body_to_inertial,
        commanded_omega_inertial=command.omega_inertial_rad_s,
        slew_rate_rad_s=zeros_f,
        target_elevation_rad=elevation,
    )
    return state, command


def update_ground_mission(
    state: MissionState,
    orbit: OrbitState,
    guidance: GuidanceConfig,
    mission: MissionConfig,
    physics: PhysicsConfig,
    orbit_config: OrbitConfig,
) -> tuple[MissionState, GuidanceTarget]:
    ground = earth_tracking_target(orbit, guidance, orbit_config)
    fallback = fallback_target(orbit, guidance)
    elevation = earth_target_elevation_rad(orbit, guidance, orbit_config)
    enter = jnp.deg2rad(jnp.asarray(mission.ground_pass_enter_elevation_deg, elevation.dtype))
    exit_ = jnp.deg2rad(jnp.asarray(mission.ground_pass_exit_elevation_deg, elevation.dtype))
    catch = jnp.deg2rad(jnp.asarray(mission.ground_pass_catch_angle_deg, elevation.dtype))
    enter_steps = _steps(mission.ground_pass_enter_dwell_seconds, physics)
    exit_steps = _steps(mission.ground_pass_exit_dwell_seconds, physics)

    above_enter = elevation >= enter
    below_exit = elevation <= exit_
    enter_dwell = jnp.where(above_enter, state.enter_dwell_steps + 1, 0)
    exit_dwell = jnp.where(below_exit, state.exit_dwell_steps + 1, 0)

    imminent = _future_max_elevation(
        orbit, guidance, orbit_config, mission.ground_pass_pre_slew_seconds
    ) >= enter

    phase = state.phase
    start_pre = (phase == STANDBY) & imminent
    phase1 = jnp.where(start_pre, PRE_SLEW, phase)

    pre_or_new = phase1 == PRE_SLEW
    post = phase1 == POST_SLEW

    pre_q, pre_omega, pre_rate, pre_caught = _pursuit_slew(
        state.commanded_q,
        state.slew_rate_rad_s,
        ground.q_body_to_inertial,
        physics.control_dt,
        jnp.deg2rad(mission.ground_pass_slew_max_rate_deg_s),
        jnp.deg2rad(mission.ground_pass_slew_max_accel_deg_s2),
        catch,
    )
    post_q, post_omega, post_rate, post_caught = _pursuit_slew(
        state.commanded_q,
        state.slew_rate_rad_s,
        fallback.q_body_to_inertial,
        physics.control_dt,
        jnp.deg2rad(mission.ground_pass_slew_max_rate_deg_s),
        jnp.deg2rad(mission.ground_pass_slew_max_accel_deg_s2),
        catch,
    )

    # PRE_SLEW may enter TRACK only after both operational visibility dwell and
    # the bounded reference have physically caught the moving target attitude.
    enter_track = pre_or_new & (enter_dwell >= enter_steps) & pre_caught
    phase2 = jnp.where(enter_track, TRACK, phase1)

    leave_track = (phase2 == TRACK) & (exit_dwell >= exit_steps)
    phase3 = jnp.where(leave_track, POST_SLEW, phase2)

    # POST ignores a noisy reacquisition until fallback is reached; this is what
    # prevents TRACK/fallback chatter around the horizon/elevation boundary.
    post_active = phase3 == POST_SLEW
    finish_post = post_active & post_caught
    phase4 = jnp.where(finish_post, STANDBY, phase3).astype(jnp.int32)

    # Reference selection. On the transition tick into POST start from the prior
    # TRACK command; the next ticks pursue fallback rather than jumping to it.
    q_cmd = state.commanded_q
    w_cmd = state.commanded_omega_inertial
    slew_rate = state.slew_rate_rad_s

    q_cmd = jnp.where((phase4 == STANDBY)[..., None], fallback.q_body_to_inertial, q_cmd)
    w_cmd = jnp.where((phase4 == STANDBY)[..., None], fallback.omega_inertial_rad_s, w_cmd)
    slew_rate = jnp.where(phase4 == STANDBY, 0.0, slew_rate)

    q_cmd = jnp.where((phase4 == PRE_SLEW)[..., None], pre_q, q_cmd)
    w_cmd = jnp.where((phase4 == PRE_SLEW)[..., None], pre_omega, w_cmd)
    slew_rate = jnp.where(phase4 == PRE_SLEW, pre_rate, slew_rate)

    q_cmd = jnp.where((phase4 == TRACK)[..., None], ground.q_body_to_inertial, q_cmd)
    w_cmd = jnp.where((phase4 == TRACK)[..., None], ground.omega_inertial_rad_s, w_cmd)
    slew_rate = jnp.where(phase4 == TRACK, 0.0, slew_rate)

    # Do not apply post_q on the exact leave_track tick: retaining the prior track
    # command guarantees no one-tick jump. Pursuit begins next control tick.
    use_post_pursuit = (phase4 == POST_SLEW) & (~leave_track)
    q_cmd = jnp.where(use_post_pursuit[..., None], post_q, q_cmd)
    w_cmd = jnp.where(use_post_pursuit[..., None], post_omega, w_cmd)
    slew_rate = jnp.where(use_post_pursuit, post_rate, slew_rate)

    changed = phase4 != phase
    elapsed = jnp.where(changed, 0, state.phase_elapsed_steps + 1)
    new_state = MissionState(
        phase=phase4,
        enter_dwell_steps=jnp.where(changed, 0, enter_dwell),
        exit_dwell_steps=jnp.where(changed, 0, exit_dwell),
        phase_elapsed_steps=elapsed,
        commanded_q=quat_normalize(q_cmd),
        commanded_omega_inertial=w_cmd,
        slew_rate_rad_s=slew_rate,
        target_elevation_rad=elevation,
    )
    active_ground = phase4 == TRACK
    direction = jnp.where(active_ground[..., None], ground.reference_direction_eci, fallback.reference_direction_eci)
    target = GuidanceTarget(
        q_body_to_inertial=new_state.commanded_q,
        omega_inertial_rad_s=new_state.commanded_omega_inertial,
        reference_valid=active_ground & ground.reference_valid,
        reference_direction_eci=direction,
    )
    return new_state, target


def find_ground_pass_windows(
    orbit_config: OrbitConfig,
    guidance: GuidanceConfig,
    mission: MissionConfig,
    *,
    elevation_deg: float | None = None,
    search_orbits: float | None = None,
    step_seconds: float | None = None,
) -> list[GroundPassWindow]:
    """Host-side circular-orbit pass search returning AOS/TCA/LOS windows."""
    threshold_deg = mission.ground_pass_enter_elevation_deg if elevation_deg is None else elevation_deg
    n = np.sqrt(orbit_config.earth_mu_m3_s2 / (orbit_config.earth_radius_m + orbit_config.altitude_m) ** 3)
    period = 2.0 * np.pi / n
    n_orbits = mission.ground_pass_search_orbits if search_orbits is None else search_orbits
    dt = mission.ground_pass_search_step_seconds if step_seconds is None else step_seconds
    times = np.arange(0.0, n_orbits * period + dt, dt, dtype=np.float32)
    orbit = orbit_state_at_time(jnp.asarray(times), orbit_config)
    elev_deg_arr = np.asarray(jnp.rad2deg(earth_target_elevation_rad(orbit, guidance, orbit_config)))
    inside = elev_deg_arr >= threshold_deg
    changes = np.diff(inside.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    windows: list[GroundPassWindow] = []
    for s, e in zip(starts, ends):
        if e <= s:
            continue
        segment = elev_deg_arr[s:e + 1]
        k = s + int(np.argmax(segment))
        windows.append(GroundPassWindow(float(times[s]), float(times[k]), float(times[e]), float(elev_deg_arr[k])))
    return windows


def usable_ground_pass_start_intervals(
    windows: Sequence[GroundPassWindow],
    episode_seconds: float,
    margin_seconds: float = 0.0,
) -> list[tuple[float, float, float]]:
    """Return (earliest_start, latest_start, tca) intervals fitting an episode."""
    result = []
    for w in windows:
        lo = w.aos_s + margin_seconds
        hi = w.los_s - margin_seconds - episode_seconds
        if hi >= lo:
            result.append((lo, hi, w.tca_s))
    return result


def select_ground_pass_start_times(
    key: jax.Array,
    batch_size: int,
    intervals: Sequence[tuple[float, float, float]],
    task: TaskConfig,
    mission: MissionConfig,
) -> jax.Array:
    if mission.ground_pass_reset_mode == "configured" or not intervals:
        return jnp.zeros((batch_size,), dtype=jnp.float32)
    arr = jnp.asarray(intervals, dtype=jnp.float32)
    if mission.ground_pass_reset_mode == "pass_centered":
        # Cycle usable passes deterministically across batch; center near TCA while
        # clamping so the complete episode stays inside the operational window.
        ids = jnp.arange(batch_size, dtype=jnp.int32) % arr.shape[0]
        chosen = arr[ids]
        centered = chosen[:, 2] - 0.5 * task.episode_seconds
        return jnp.clip(centered, chosen[:, 0], chosen[:, 1])
    if mission.ground_pass_reset_mode == "random_visible":
        key_i, key_u = jax.random.split(key)
        ids = jax.random.randint(key_i, (batch_size,), 0, arr.shape[0])
        chosen = arr[ids]
        u = jax.random.uniform(key_u, (batch_size,))
        return chosen[:, 0] + u * (chosen[:, 1] - chosen[:, 0])
    raise ValueError(f"Unsupported ground_pass_reset_mode: {mission.ground_pass_reset_mode}")
