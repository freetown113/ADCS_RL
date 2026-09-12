import argparse
import json
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from simulators.simplified.config import ExperimentConfig, default_config
from simulators.simplified.env import SatelliteEnv
from simulators.simplified.evaluation import evaluate_policy
from simulators.simplified.network import make_network
from simulators.simplified.ppo import (
    TrainState,
    collect_rollout,
    compute_gae,
    flatten_rollout,
    make_optimizer,
    make_ppo_update,
    rollout_metrics,
)


def _device_float_dict(metrics: dict[str, Any]) -> dict[str, float]:
    return {name: float(value) for name, value in jax.device_get(metrics).items()}


def _merge_metrics(*groups: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for group in groups:
        merged.update(group)
    return merged


def save_checkpoint(path: Path, train_state: TrainState, config: ExperimentConfig, update: int):
    payload = {
        "update": update,
        "config": asdict(config),
        "params": jax.device_get(train_state.params),
        "opt_state": jax.device_get(train_state.opt_state),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

def load_checkpoint(path: Path):
    with open(path, "rb") as handle:
        state = pickle.load(handle)

    return state


def train(config: ExperimentConfig) -> TrainState:
    output_dir = Path(config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "videos").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    env = SatelliteEnv(config)
    if (env.episode_steps * config.ppo.num_envs) % config.ppo.num_minibatches != 0:
        raise ValueError("episode_steps * num_envs must divide num_minibatches")

    network = make_network(env.observation_size, env.action_size, config.network)
    rng = jax.random.PRNGKey(config.run.seed)
    rng, init_key, dummy_reset_key = jax.random.split(rng, 3)
    dummy_state = env.reset(dummy_reset_key, 1)
    dummy_obs = env.observe(dummy_state)
    params = network.init(init_key, dummy_obs)

    if config.control.mode=="motor_direct":
        _state = load_checkpoint('output_motors_pretrain/satellite_reference/final.pkl')
        train_state = TrainState(_state['params'], _state['opt_state'])

    optimizer = make_optimizer(config)
    train_state = TrainState(params=params, opt_state=optimizer.init(params))
    ppo_update = make_ppo_update(network.apply, optimizer, config)

    rollout_fn = jax.jit(
        lambda params, initial_state, key: collect_rollout(
            env,
            params,
            network.apply,
            initial_state,
            key,
            config.network.fixed_log_std,
        )
    )

    def prepare(rollout):
        advantage, returns = compute_gae(
            rollout.reward,
            rollout.done,
            rollout.old_value,
            config.ppo.gamma,
            config.ppo.gae_lambda,
        )
        normalized_advantage = (advantage - jnp.mean(advantage)) / (
            jnp.std(advantage) + 1.0e-8
        )
        batch = flatten_rollout(rollout, normalized_advantage, returns)
        metrics = rollout_metrics(rollout, advantage, returns)
        return batch, metrics

    prepare_fn = jax.jit(prepare)

    fixed_eval_key = jax.random.PRNGKey(config.run.seed + 10_000)
    fixed_eval_state = env.reset(fixed_eval_key, config.run.eval_envs)
    eval_fn = jax.jit(
        lambda params, state: evaluate_policy(env, params, network.apply, state)
    )


    for update_index in range(1, config.run.total_updates + 1):
        rng, reset_key, rollout_key, update_key = jax.random.split(rng, 4)
        initial_state = env.reset(reset_key, config.ppo.num_envs)
        _, rollout = rollout_fn(train_state.params, initial_state, rollout_key)
        batch, rollout_summary = prepare_fn(rollout)
        train_state, update_summary = ppo_update(
            train_state, batch, update_key
        )

        metrics = _merge_metrics(rollout_summary, update_summary)

        if update_index % config.run.eval_every == 0 or update_index == 1:
            _, _, eval_summary = eval_fn(train_state.params, fixed_eval_state)
            metrics.update({f"eval/{k}": v for k, v in eval_summary.items()})

        scalar_metrics = _device_float_dict(metrics)
        scalar_metrics["update"] = float(update_index)

        if update_index % config.run.log_every == 0 or update_index == 1:
            keys = [
                "episode_return_mean",
                "final_angle_deg_mean",
                "final_rate_mean",
                "success_rate",
                "deterministic_action_abs",
                "action_saturation",
                "approx_kl",
                "clip_fraction",
                "explained_variance",
            ]
            if "eval/final_angle_deg_mean" in scalar_metrics:
                keys.extend(
                    [
                        "eval/final_angle_deg_mean",
                        "eval/final_rate_mean",
                        "eval/success_rate",
                    ]
                )
            text = " | ".join(
                f"{key}={scalar_metrics[key]:.5g}"
                for key in keys
                if key in scalar_metrics
            )
            print(f"update={update_index} | {text}")

        if update_index % config.run.eval_every == 0:
            save_checkpoint(
                output_dir / "checkpoints" / f"update_{update_index:07d}.pkl",
                train_state,
                config,
                update_index,
            )

        if config.run.video_every > 0 and update_index % config.run.video_every == 0:
            from .video import save_policy_video

            one_state = env.reset(fixed_eval_key, 1)
            save_policy_video(
                env=env,
                params=train_state.params,
                apply_fn=network.apply,
                initial_state=one_state,
                output_path=output_dir / "videos" / f"update_{update_index:07d}.mp4",
                title=f"PPO update {update_index}",
            )

    save_checkpoint(output_dir / "final.pkl", train_state, config, config.run.total_updates)
    return train_state


def parse_config() -> ExperimentConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--eval-envs", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--reset-mode", choices=["fixed", "cone"], default=None)
    parser.add_argument("--max-angle-deg", type=float, default=None)
    parser.add_argument("--fixed-angle-deg", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--control-mode", choices=["residual_pd", "direct", "motor_direct"], default=None)
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--video-every", type=int, default=None)
    parser.add_argument(
        "--fault-mode",
        choices=["none", "permanent", "fixed_interval", "random_interval", "stochastic"],
        default=None,
    )
    parser.add_argument("--fault-wheel-selection", choices=["fixed", "random"], default=None)
    parser.add_argument("--fault-wheel-index", type=int, default=None)
    parser.add_argument("--fault-start-seconds", type=float, default=None)
    parser.add_argument("--fault-duration-seconds", type=float, default=None)
    parser.add_argument("--fault-failure-rate", type=float, default=None)
    parser.add_argument("--fault-recovery-rate", type=float, default=None)
    parser.add_argument("--motor-control-limit", type=float, default=None)
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
    if args.reset_mode is not None:
        config = replace(config, task=replace(config.task, reset_mode=args.reset_mode))
    if args.max_angle_deg is not None:
        config = replace(
            config, task=replace(config.task, max_initial_angle_deg=args.max_angle_deg)
        )
    if args.fixed_angle_deg is not None:
        config = replace(
            config, task=replace(config.task, fixed_angle_deg=args.fixed_angle_deg)
        )
    if args.learning_rate is not None:
        config = replace(
            config, ppo=replace(config.ppo, learning_rate=args.learning_rate)
        )
    if args.control_mode is not None:
        config = replace(
            config, control=replace(config.control, mode=args.control_mode)
        )
    if args.residual_scale is not None:
        config = replace(
            config, control=replace(config.control, residual_scale=args.residual_scale)
        )
    if args.video_every is not None:
        config = replace(
            config, run=replace(config.run, video_every=args.video_every)
        )
    if args.fault_mode is not None:
        config = replace(config, faults=replace(config.faults, mode=args.fault_mode))
    if args.fault_wheel_selection is not None:
        config = replace(
            config, faults=replace(config.faults, wheel_selection=args.fault_wheel_selection)
        )
    if args.fault_wheel_index is not None:
        config = replace(
            config, faults=replace(config.faults, wheel_index=args.fault_wheel_index)
        )
    if args.fault_start_seconds is not None:
        config = replace(
            config, faults=replace(config.faults, start_seconds=args.fault_start_seconds)
        )
    if args.fault_duration_seconds is not None:
        config = replace(
            config, faults=replace(config.faults, duration_seconds=args.fault_duration_seconds)
        )
    if args.fault_failure_rate is not None:
        config = replace(
            config, faults=replace(
                config.faults, failure_rate_per_second=args.fault_failure_rate
            )
        )
    if args.fault_recovery_rate is not None:
        config = replace(
            config, faults=replace(
                config.faults, recovery_rate_per_second=args.fault_recovery_rate
            )
        )
    if args.motor_control_limit is not None:
        config = replace(
            config, physics=replace(
                config.physics, motor_control_limit=args.motor_control_limit
            )
        )
    return config


def main():
    train(parse_config())


if __name__ == "__main__":
    main()
