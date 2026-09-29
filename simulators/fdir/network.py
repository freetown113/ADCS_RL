from typing import Callable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from simulators.fdir.config import NetworkConfig


def make_network(
    observation_size: int,
    action_size: int,
    config: NetworkConfig,
):
    def activation(x: jax.Array) -> jax.Array:
        if config.activation == "tanh":
            return jnp.tanh(x)
        if config.activation == "relu":
            return jax.nn.relu(x)
        raise ValueError(f"Unsupported activation: {config.activation}")

    def mlp(x: jax.Array, prefix: str) -> jax.Array:
        for index, size in enumerate(config.hidden_sizes):
            x = hk.Linear(
                size,
                w_init=hk.initializers.Orthogonal(jnp.sqrt(2.0)),
                b_init=hk.initializers.Constant(0.0),
                name=f"{prefix}_hidden_{index}",
            )(x)
            x = activation(x)
        return x

    def forward(observation: jax.Array) -> Tuple[jax.Array, jax.Array]:
        actor_features = mlp(observation, "actor")
        critic_features = mlp(observation, "critic")

        raw_mean = hk.Linear(
            action_size,
            w_init=hk.initializers.Orthogonal(0.01),
            b_init=hk.initializers.Constant(0.0),
            name="policy_mean",
        )(actor_features)
        limit = config.policy_mean_limit
        mean = limit * jnp.tanh(raw_mean / limit)

        value = hk.Linear(
            1,
            w_init=hk.initializers.Orthogonal(1.0),
            b_init=hk.initializers.Constant(0.0),
            name="value",
        )(critic_features)[..., 0]
        return mean, value

    return hk.without_apply_rng(hk.transform(forward))
