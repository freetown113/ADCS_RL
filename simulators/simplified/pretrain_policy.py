import os
import argparse
import csv
import json
import pickle
import random
from dataclasses import asdict, replace, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax

from simulators.simplified.config import ExperimentConfig, default_config
from simulators.simplified.env import SatelliteEnv
from simulators.simplified.network import make_network
from simulators.simplified.ppo import (
    TrainState,
    make_ppo_update
)


def tree_l2_norm(tree: Any):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))

@dataclass
class DataBatch:
    obs: jnp.array
    target: jnp.array

class Dataset:
    def __init__(self, path, batch_size, file='dataset_direct_control.npz'):
        self.train_data = []
        self.test_data = []
        self.bs = batch_size
        self.load_data(path, file)
        self.train_chunks = self.__len__() // self.bs
        self.test_chunks = len(self.test_data) // self.bs
        
    def __len__(self):
        return len(self.train_data)

    def data_example(self):
        indices = np.random.choice(range(self.__len__()), size=self.bs, replace=False)
        ex = [self.train_data[idx] for idx in indices]
        obs, tar = [], []
        for ob, tg in ex:
            obs.append(ob)
            tar.append(tg)
        obj = DataBatch(obs=jax.device_put(jnp.stack(obs)),
                        target=jax.device_put(jnp.stack(tar))
                        )
        return obj

    def sample(self, is_train=True):
        if is_train:
            data = self.train_data
            random.shuffle(data)
            chunks = self.train_chunks
        else:
            data = self.test_data
            chunks = self.test_chunks
        for _ in range(chunks):
            indices = np.random.choice(range(self.__len__()), size=self.bs, replace=False)
            el = [data[idx] for idx in indices]
            obs, tar = [], []
            for ob, tg in el:
                obs.append(ob)
                tar.append(tg)
            obj = DataBatch(obs=jax.device_put(jnp.stack(obs)),
                            target=jax.device_put(jnp.stack(tar))
                            )
            yield obj

    def load_data(self, location: str, filename: str, split: float = 0.9):
        data = np.load(os.path.join(location, filename))
        for x,y in zip(data['observation'], data['teacher_action']):
            self.train_data.append((x,y))
        np.random.shuffle(self.train_data)
        cut = int(self.__len__()*split)
        self.test_data = self.train_data[cut:]
        self.train_data = self.train_data[:cut]
        


def train_fn(
        dataset,
        apply_fn,
        optimizer
    ):
    def loss_fn(params: hk.Params, batch):
        mean, _ = apply_fn(params, batch.obs)
        predicted_action = jnp.tanh(mean)
        imitation_loss = jnp.mean(
            jnp.square(
                predicted_action - batch.target
            )
        )
        aux = {
            "loss": imitation_loss,
        }
        return imitation_loss, aux

    def minibatch_step(train_state: TrainState, batch):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            train_state.params, batch
        )
        updates, next_opt_state = optimizer.update(
            grads, train_state.opt_state, train_state.params
        )
        next_params = optax.apply_updates(train_state.params, updates)

        return TrainState(next_params, next_opt_state), metrics

    def update(train_state: TrainState, key):
        sample_count = dataset.train_chunks
        epoch_keys = jax.random.split(key, sample_count)

        def epoch_step(carry, _):
            state, epoch_keys = carry
            batch = next(dataset.sample())
            next_state, metrics = minibatch_step(state, batch)
            return (next_state, epoch_keys), metrics

        (next_state, _), epoch_metrics = jax.lax.scan(
            epoch_step, (train_state, epoch_keys), None, length=len(epoch_keys)
        )

        metrics = jax.tree_util.tree_map(
            lambda x: jnp.mean(x, axis=0), epoch_metrics
        )
        return next_state, metrics

    return jax.jit(update)

def eval_fn(
        dataset,
        apply_fn
    ):
    def loss_fn(params: hk.Params, batch):
        mean, _ = apply_fn(params, batch.obs)
        predicted_action = jnp.tanh(mean)
        imitation_loss = jnp.mean(
            jnp.square(
                predicted_action - batch.target
            )
        )
        aux = {
            "loss": imitation_loss,
        }
        return imitation_loss, aux

    def minibatch_step(train_state: TrainState, batch):
        _, aux = loss_fn(train_state.params, batch)
        return aux

    def update(train_state: TrainState, key):
        sample_count = dataset.test_chunks
        epoch_keys = jax.random.split(key, sample_count)

        def epoch_step(carry, _):
            state, epoch_keys = carry
            batch = next(dataset.sample())
            metrics = minibatch_step(state, batch)
            return (state, epoch_keys), metrics

        (next_state, _), epoch_metrics = jax.lax.scan(
            epoch_step, (train_state, epoch_keys), None, length=len(epoch_keys)
        )

        metrics = jax.tree_util.tree_map(
            lambda x: jnp.mean(x, axis=0), epoch_metrics
        )
        return metrics

    return jax.jit(update)


def _device_float_dict(metrics: dict[str, Any]) -> dict[str, float]:
    return {name: float(value) for name, value in jax.device_get(metrics).items()}


def _merge_metrics(*groups: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for group in groups:
        merged.update(group)
    return merged

def make_optimizer(max_grad_norm, learning_rate, adam_eps) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(
            learning_rate=learning_rate,
            eps=adam_eps,
        ),
    )

def save_checkpoint(path: Path, train_state: TrainState, config: ExperimentConfig, update: int):
    payload = {
        "update": update,
        "config": asdict(config),
        "params": jax.device_get(train_state.params),
        "opt_state": jax.device_get(train_state.opt_state),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def train(config: ExperimentConfig) -> TrainState:
    output_dir = Path(config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pretrain_checkpoints").mkdir(exist_ok=True)
    (output_dir / "pretrain_config.json").write_text(json.dumps(asdict(config), indent=2))

    env = SatelliteEnv(config)
    if (env.episode_steps * config.ppo.num_envs) % config.ppo.num_minibatches != 0:
        raise ValueError("episode_steps * num_envs must divide num_minibatches")

    dataset = Dataset(path='policy_direct_control_dataset', batch_size=2048)

    network = make_network(env.observation_size, env.action_size, config.network)
    rng = jax.random.PRNGKey(config.run.seed)
    rng, init_key, dummy_reset_key = jax.random.split(rng, 3)
    dummy_state = env.reset(dummy_reset_key, 1)
    dummy_obs = env.observe(dummy_state)
    params = network.init(init_key, dummy_obs)

    optimizer = make_optimizer(max_grad_norm=0.5, learning_rate=3e-5, adam_eps=1e-5)
    train_state = TrainState(params=params, opt_state=optimizer.init(params))
    ppo_update = make_ppo_update(network.apply, optimizer, config)
    trainer = train_fn(dataset, network.apply, optimizer)
    evaluat = eval_fn(dataset, network.apply)
    
    csv_path = output_dir / "pretrain_metrics.csv"
    csv_file = csv_path.open("w", newline="")
    
    try:
        for update_index in range(1, 50):
            rng, reset_key, rollout_key, update_key = jax.random.split(rng, 4)

            train_state, metrics = trainer(train_state, rollout_key)

            eval_metrics = evaluat(train_state, rollout_key)

            print(f"Epoch {update_index} loss is {_device_float_dict(metrics)} | eval {_device_float_dict(eval_metrics)}")

            if update_index % 10 == 0:
                save_checkpoint(
                    output_dir / "pretrain_checkpoints" / f"update_{update_index:07d}.pkl",
                    train_state,
                    config,
                    update_index,
                )

    finally:
        csv_file.close()

    save_checkpoint(output_dir / "final.pkl", train_state, config, config.run.total_updates)
    return train_state


def parse_config() -> ExperimentConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--eval-envs", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    config = default_config()
    if args.updates is not None:
        config = replace(config, run=replace(config.run, total_updates=args.updates))
    if args.num_envs is not None:
        config = replace(config, ppo=replace(config.ppo, num_envs=args.num_envs))
    if args.eval_envs is not None:
        config = replace(config, run=replace(config.run, eval_envs=args.eval_envs))
    if args.output is not None:
        config = replace(config, run=replace(config.run, output_dir=args.output))

    config = replace(
            config,
            control=replace(
                config.control,
                mode="motor_direct",
            ),
        )

    return config


def main():
    train(parse_config())


if __name__ == "__main__":
    main()
