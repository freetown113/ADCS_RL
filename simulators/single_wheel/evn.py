import jax
import jax.numpy as jnp

from brax.envs.base import Env, State
from src.env.single_wheel.reward import compute_reward

I_satellite = 2.0
J_wheel = 0.05
J_wheel_inv = 1.0 / J_wheel
I_satellite_inv = 1.0 / I_satellite
b_friction = 0.001
OMEGA_MAX = 628.0

@jax.jit
def step(state, action, dt=0.01):
    """
    Simplified 1D physics: one wheel is used to control rotation in a single plane.
    Satellites dyamics is inverse wheel rotation: a torque applied at a wheel makes 
    satellite move in opposite direction
    state: tuple (omega, theta, omega_w)
    action: one unique torque
    """
    omega, theta, omega_w = state
    tau_m = jnp.clip(action, -1.0, 1.0) * 0.8
    
    d_omega_w = J_wheel_inv * (tau_m - b_friction * omega_w)
    
    d_omega = I_satellite_inv * (-tau_m)
    
    next_omega = omega + d_omega * dt
    next_theta = theta + omega * dt
    next_omega_w = jnp.clip(omega_w + d_omega_w * dt, -OMEGA_MAX, OMEGA_MAX)
    
    next_theta = jnp.mod(next_theta + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    
    return (next_omega, next_theta, next_omega_w)



class Satellite1DEnv(Env):
    def __init__(self, dt: float = 0.01, episode_length: int = 1000):
        super().__init__()
        self._dt = dt
        self._episode_length = episode_length
        self._action_size = 1
        self._observation_size = 3

    @property
    def backend(self) -> str:
        return 'custom'
    
    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def observation_size(self) -> int:
        return self._observation_size

    def reset(self, rng: jnp.ndarray):
        '''
        Initial heading is set to an angle from uniform distribution (0, 2*Pi)
        Initial angular speed is set to be non 0, imitating stabilisation from
        emergency.
        '''
        rng_theta, rng_target, rng_omega = jax.random.split(rng, 3)

        theta = jax.random.uniform(rng_theta, minval=0, maxval=jnp.pi*2)
        omega = jnp.array(0.1)
        omega_w = jnp.array(0.0)
        target = jax.random.uniform(rng_target, minval=0, maxval=jnp.pi*2)

        plain_err = target - theta
        angle_err = (plain_err + jnp.pi) % (2 * jnp.pi) - jnp.pi
        
        obs = jnp.array([omega, angle_err, omega_w / OMEGA_MAX])
        
        physics_state = {"omega": omega, "theta": theta, "omega_w": omega_w, "target": target}
        return State(
            obs=obs,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
            metrics={"r_att": 0.0, "r_omega": 0.0, "total": 0.0},
            pipeline_state=physics_state
        )

    def step(self, state, action):
        phys = state.pipeline_state
        current_state = (phys["omega"], phys["theta"], phys["omega_w"])
        target = phys["target"]
        
        action_scalar = jnp.squeeze(action)
        next_omega, next_theta, next_omega_w = step(current_state, action_scalar, self._dt)
        plain_err = target - next_theta
        angle_err = (plain_err + jnp.pi) % (2 * jnp.pi) - jnp.pi
        abs_ang_err = jnp.abs(angle_err)
        
        total_reward, rews = compute_reward(action_scalar, next_omega, abs_ang_err)
        r_att, r_omega, _ = rews

        next_physics = {"omega": next_omega, "theta": next_theta, "omega_w": next_omega_w, "target": target}
        next_obs = jnp.array([next_omega, angle_err, next_omega_w / OMEGA_MAX])
        
        return state.replace(
            obs=next_obs,
            reward=total_reward,
            pipeline_state=next_physics,
            metrics={"r_att": r_att, "r_omega": r_omega, "total": total_reward}
        )
    