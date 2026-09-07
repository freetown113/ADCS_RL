import jax
import jax.numpy as jnp

@jax.jit
def compute_reward(action, omega, angle_err):
    r_att = 10.0 * (1.0 - angle_err / jnp.pi)
    
    r_omega = -2.0 * jnp.square(omega)
    
    r_energy = -0.1 * jnp.square(action)
    
    total_reward = r_att + r_omega + r_energy
    return total_reward, (r_att, r_omega, r_energy)