from brax import envs
import jax
import haiku as hk
import optax
import functools
import numpy as np
import jax.numpy as jnp
from time import time
from src.env.single_wheel.evn import Satellite1DEnv
from src.env.single_wheel.visualisation import save_simulation_video

from src.env.single_wheel.algo import ActorCritic
from typing import NamedTuple


# jax.config.update("jax_disable_jit", True)


NUM_ENVS = 256
EVAL_NUM_ENVS = 2
N_STEPS = 32
LR = 1e-5
WEIGHT_DECAY = 1e-4
GRAD_NORM = 0.5
GAMMA = 0.99
LOG_STD_MIN = -5
LOG_STD_MAX = 2
NUM_EVAL_STEPS = 3000


@jax.tree_util.register_pytree_node_class
class TrainState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

    def tree_flatten(self):
            children = (self.params, self.opt_state)
            aux_data = None
            return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def make_model(num_actions):  
    def closure(state): 
        forward = ActorCritic(num_actions=num_actions, name='ReactionWheelPolicy')
        return forward(state)
    return hk.transform(closure)


def make_train(vmap_env, eval_vmap_env, forward_fn_apply, optimizer, eval_num_envs=100):
    @functools.partial(jax.jit, donate_argnums=(0,))
    def eval_epoch(train_state, env_states, rng):
        def rollout_step(carry, unused_input):
            current_env_states, current_rng = carry
            
            current_rng, action_rng, infer_rng = jax.random.split(current_rng, 3)
            (actor_means, raw_log_std), critic_values = forward_fn_apply(train_state.params, infer_rng, current_env_states.obs)

            actions = actor_means
            actions = jnp.clip(actions, -1.0, 1.0)
        
            next_env_states = eval_vmap_env.step(current_env_states, actions)
            
            next_carry = (next_env_states, current_rng)
            outputs = (current_env_states.obs, actions, next_env_states.reward * (1.0-next_env_states.done), critic_values, current_env_states.pipeline_state['theta'])
            return next_carry, outputs
        
        rng, rollout_rng = jax.random.split(rng)
        init_carry = (env_states, rollout_rng)
        (final_env_states, _), (obs_b, actions_b, rewards_b, values_b, theta) = jax.lax.scan(
            rollout_step, init_carry, None, length=NUM_EVAL_STEPS
        )
        
        env = 0
        q_vis = theta[:,env]
        target = final_env_states.pipeline_state['target'][env]

        return rewards_b, final_env_states, q_vis, target
    

    @functools.partial(jax.jit, donate_argnums=(0,))
    def train_epoch(train_state, env_states, rng):

        def rollout_step(carry, unused_input):
            current_env_states, current_rng = carry
            
            current_rng, action_rng, infer_rng = jax.random.split(current_rng, 3)

            (actor_means, raw_log_std), critic_values = forward_fn_apply(train_state.params, infer_rng, current_env_states.obs)
            actor_log_std = jnp.clip(raw_log_std, LOG_STD_MIN, LOG_STD_MAX)
            actor_std = jnp.exp(actor_log_std)

            noise = jax.random.normal(action_rng, shape=(NUM_ENVS, 1))
            actions = actor_means + noise * actor_std
            actions = jnp.clip(actions, -1.0, 1.0)
            
            next_env_states = vmap_env.step(current_env_states, actions)
            
            next_carry = (next_env_states, current_rng)
            outputs = (current_env_states.obs, actions, next_env_states.reward, next_env_states.done, critic_values)
            return next_carry, outputs

        rng, rollout_rng = jax.random.split(rng)
        init_carry = (env_states, rollout_rng)
        (final_env_states, _), (obs_b, actions_b, rewards_b, done_b, values_b) = jax.lax.scan(
            rollout_step, init_carry, None, length=N_STEPS
        )
        
        _, critic_values_b = forward_fn_apply(train_state.params, rng, final_env_states.obs)
        next_value = jax.lax.stop_gradient(critic_values_b)
        returns = []

        for r, d in zip(reversed(rewards_b), reversed(done_b)):
            next_value = r + GAMMA * next_value * (1.0 - d)
            returns.insert(0, next_value)

        returns = jnp.array(returns)
        returns_mean = jnp.mean(returns)
        returns_std = jnp.std(returns) + 1e-8
        returns_normalized = (returns - returns_mean) / returns_std

        advantage_detached = jax.lax.stop_gradient(returns_normalized - values_b)

        def loss_fn(model_params, rng):
            rng, loss_rng = jax.random.split(rng)
            
            obs_flat = obs_b.reshape(-1, obs_b.shape[-1])
            (actor_means_flat, raw_log_std_flat), critic_values_flat = forward_fn_apply(model_params, loss_rng, obs_flat)
            
            actor_means_b = actor_means_flat.reshape(N_STEPS, NUM_ENVS, 1)
            raw_log_std = raw_log_std_flat.reshape(N_STEPS, NUM_ENVS, 1)
            critic_values_b = critic_values_flat.reshape(N_STEPS, NUM_ENVS)
            
            log_std = jnp.clip(raw_log_std, -5.0, 2.0)
            actor_std = jnp.exp(log_std)
            
            critic_loss = 0.5 * jnp.mean(jnp.square(returns_normalized - critic_values_b))
            
            log_probs = jax.scipy.stats.norm.logpdf(actions_b, loc=actor_means_b, scale=actor_std)
            log_prob = jnp.sum(log_probs, axis=-1)
            policy_loss = -jnp.mean(log_prob * advantage_detached)
            
            entropy = 0.5 * (1.0 + jnp.log(2 * jnp.pi * jnp.square(actor_std)))
            entropy_loss = -jnp.mean(jnp.sum(entropy, axis=-1))
            
            c1 = 0.5
            c2 = 1e-5
            return policy_loss + (c1 * critic_loss) + (c2 * entropy_loss)

        loss, grads = jax.value_and_grad(loss_fn)(train_state.params, rng)
        
        updates, new_opt_state = optimizer.update(grads, train_state.opt_state, params=train_state.params)
        new_params = optax.apply_updates(train_state.params, updates)
        
        new_train_state = TrainState(params=new_params, opt_state=new_opt_state)
        
        return new_train_state, final_env_states, loss

    return train_epoch, eval_epoch


def launch():

    envs.register_environment('SatelliteEnv-v0', Satellite1DEnv)

    vmap_env = envs.create(
        env_name='SatelliteEnv-v0', 
        batch_size=NUM_ENVS,
        auto_reset=True,
        episode_length=3000,
    )

    test_env = envs.create(
        env_name='SatelliteEnv-v0', 
        batch_size=EVAL_NUM_ENVS,
        auto_reset=False,
        episode_length=3000,
    )

    actor_critic_net = make_model(vmap_env.action_size)

    rng_global = jax.random.PRNGKey(521)
    rng_global, rng_network, rng_envs, rng_eval_envs = jax.random.split(rng_global, 4)

    dummy_obs = jnp.zeros((vmap_env.observation_size,))
    initial_params = actor_critic_net.init(rng_network, dummy_obs)

    optimizer = optax.chain(
            optax.clip_by_global_norm(GRAD_NORM),
            optax.add_decayed_weights(weight_decay=WEIGHT_DECAY),
            optax.adam(LR)
        )
    initial_opt_state = optimizer.init(initial_params)
    train_state = TrainState(params=initial_params, opt_state=initial_opt_state)

    train_fn, eval_fn = make_train(vmap_env, test_env, actor_critic_net.apply, optimizer, EVAL_NUM_ENVS)

    env_states = vmap_env.reset(rng_envs)

    elapsed = time()
    print("Commence training agent")
    for epoch in range(int(10e7)):
        rng_global, epoch_rng = jax.random.split(rng_global)
        
        train_state, env_states, loss_value = train_fn(train_state, env_states, epoch_rng)

        if not epoch % 1_000:
            print(f'Epoque {epoch} | loss : {loss_value:.6f}')
   
        if not epoch % 100_000 and epoch:
            rng_eval_envs, rng_curr = jax.random.split(rng_eval_envs)
            test_env_states = test_env.reset(rng_curr)
            return_val, fin_test_env_states, observations, target = eval_fn(train_state, test_env_states, epoch_rng)
            result = np.abs(fin_test_env_states.metrics['r_att'] / (10.))
            print(f"Epoque {epoch} | loss : {loss_value:.6f} | return mean: {np.mean(result):.7f} / med: {np.median(result):.7f} | took {time()-elapsed:.2f} sec")
            elapsed = time()
            save_simulation_video(np.array(observations), target, output_filename=f"results/single_wheel_task_{epoch}.mp4", fps=20)


if __name__=='__main__':
    launch()



