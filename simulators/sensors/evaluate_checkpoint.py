"""Evaluate a saved checkpoint and optionally render MP4."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax

from .config import config_from_dict
from .env import SatelliteEnv
from .evaluation import compare_baselines, evaluate_policy
from .network import make_network
from .video import save_trajectory_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--eval-envs", type=int, default=None)
    args = parser.parse_args()

    with args.checkpoint.open("rb") as handle:
        checkpoint = pickle.load(handle)
    config = config_from_dict(checkpoint["config"])
    env = SatelliteEnv(config)
    network = make_network(env.observation_size, env.action_size, config.network)

    eval_envs = args.eval_envs or config.run.eval_envs
    state = env.reset(jax.random.PRNGKey(config.run.seed + 10_000), eval_envs)
    _, trajectory, metrics = jax.jit(
        lambda params, initial: evaluate_policy(env, params, network.apply, initial)
    )(checkpoint["params"], state)
    print(json.dumps({k: float(v) for k, v in jax.device_get(metrics).items()}, indent=2))

    baseline_state = env.reset(jax.random.PRNGKey(config.run.seed + 10_000), eval_envs)
    baseline = compare_baselines(env, baseline_state)
    print("Baselines:")
    print(json.dumps(
        {
            name: {k: float(v) for k, v in jax.device_get(values).items()}
            for name, values in baseline.items()
        },
        indent=2,
    ))

    if args.video is not None:
        one = jax.tree_util.tree_map(lambda x: x[:, :1] if x.ndim >= 2 else x, trajectory)
        save_trajectory_video(env, one, args.video, title=f"Checkpoint {checkpoint['update']}")
        print(f"Saved {args.video}")


if __name__ == "__main__":
    main()
