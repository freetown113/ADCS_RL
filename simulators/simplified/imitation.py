from typing import NamedTuple

import jax
import jax.numpy as jnp

from simulators.simplified.control import pd_command_action
from simulators.simplified.env import SatelliteEnv



class ImitationDataset(NamedTuple):
    observations: jax.Array
    teacher_actions: jax.Array


def collect_pd_motor_teacher_dataset(
    env: SatelliteEnv,
    reset_key: jax.Array,
    batch_size: int,
) -> ImitationDataset:
    """Collects one full episode of motor-level PD/allocator demonstrations.

    ``env`` must use ``control.mode='motor_direct'`` so observation/action shapes
    exactly match the motor policy that will later run at inference.
    """
    if env.config.control.mode != "motor_direct":
        raise ValueError("teacher dataset requires control.mode='motor_direct'")
    initial = env.reset(reset_key, batch_size)

    def step_fn(state, _):
        obs = env.observe(state)
        teacher_action = pd_command_action(env, state)
        next_state, _, _, _ = env.step(state, teacher_action)
        return next_state, (obs, teacher_action)

    _, (obs, action) = jax.lax.scan(
        step_fn, initial, None, length=env.episode_steps
    )
    return ImitationDataset(
        observations=obs.reshape((-1, obs.shape[-1])),
        teacher_actions=action.reshape((-1, action.shape[-1])),
    )


def imitation_mse(predicted_action: jax.Array, teacher_action: jax.Array) -> jax.Array:
    return jnp.mean(jnp.square(predicted_action - teacher_action))
