import jax
import jax.numpy as jnp
import haiku as hk


class ActorCritic(hk.Module):
    def __init__(self, 
                 num_actions: int = 4,
                 hidden: int = 128,
                 name='A2C'):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.hidden = hidden

    def __call__(self, x):
        h = hk.Linear(self.hidden, name='linear1')(x)
        h = jax.nn.relu(h)
        h = hk.Linear(self.hidden, name='linear2')(h)
        h = jax.nn.relu(h)
        
        actor_mean = hk.Linear(self.num_actions,
                               with_bias=True, 
                               w_init=hk.initializers.Orthogonal(scale=0.01), 
                               b_init=hk.initializers.Constant(0),  
                               name='actor_mean')(h)
        actor_std = hk.Linear(self.num_actions, 
                              with_bias=True, 
                              w_init=hk.initializers.Orthogonal(scale=0.01), 
                              b_init=hk.initializers.Constant(0), 
                              name='actor_std')(h)

        actor_mean = jax.nn.tanh(actor_mean)
        actor_std = jax.nn.tanh(actor_std)

        critic_value = hk.Linear(1)(h)
        
        return (actor_mean, actor_std), jnp.squeeze(critic_value, axis=-1)

