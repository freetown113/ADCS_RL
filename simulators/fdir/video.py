from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import jax
import matplotlib.pyplot as plt
import numpy as np

from simulators.fdir.env import EnvState, SatelliteEnv
from simulators.fdir.evaluation import EvaluationTrajectory, evaluate_policy


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1.0e-12)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _draw_axes(ax, rotation: np.ndarray, dashed: bool, alpha: float):
    colors = ("r", "g", "b")
    labels = ("x", "y", "z")
    for index in range(3):
        vector = rotation[:, index]
        ax.quiver(
            0,
            0,
            0,
            vector[0],
            vector[1],
            vector[2],
            length=0.85,
            normalize=True,
            color=colors[index],
            linestyle="--" if dashed else "-",
            alpha=alpha,
            linewidth=1.5 if dashed else 2.5,
        )
        ax.text(
            *(0.95 * vector),
            labels[index] + ("*" if dashed else ""),
            color=colors[index],
            alpha=alpha,
        )


def _draw_body(ax, rotation: np.ndarray):
    half = np.asarray([0.28, 0.20, 0.14])
    vertices = np.asarray(
        [
            [-half[0], -half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], half[1], half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], -half[2]],
            [half[0], half[1], half[2]],
        ]
    )
    vertices = vertices @ rotation.T
    edges = (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    )
    for a, b in edges:
        line = vertices[[a, b]]
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color="black", linewidth=1.5)


def _single_environment(trajectory: EvaluationTrajectory, index: int = 0):
    return jax.tree_util.tree_map(lambda x: np.asarray(x)[:, index], trajectory)


def save_trajectory_video(
    env: SatelliteEnv,
    trajectory: EvaluationTrajectory,
    output_path: str | Path,
    *,
    title: str = "Satellite stabilization",
    fps: int = 20,
    frame_stride: int = 4,
    environment_index: int = 0,
) -> Path:
    """Saves one environment trajectory as an MP4."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tr = _single_environment(trajectory, environment_index)

    frame_indices = np.arange(0, len(tr.reward), max(1, frame_stride))
    time = np.arange(len(tr.reward)) * env.config.physics.control_dt
    cumulative_return = np.cumsum(tr.reward)

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.25, 1.0, 1.0))
    ax_3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_att = fig.add_subplot(grid[0, 1])
    ax_action = fig.add_subplot(grid[0, 2])
    ax_wheel = fig.add_subplot(grid[1, 1])
    ax_reward = fig.add_subplot(grid[1, 2])

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    try:
        for frame_index in frame_indices:
            ax_3d.cla()
            ax_att.cla()
            ax_action.cla()
            ax_wheel.cla()
            ax_reward.cla()

            rotation = quaternion_to_rotation_matrix(tr.q[frame_index])
            target_rotation = quaternion_to_rotation_matrix(tr.target_q[frame_index])
            _draw_body(ax_3d, rotation)
            _draw_axes(ax_3d, target_rotation, dashed=True, alpha=0.35)
            _draw_axes(ax_3d, rotation, dashed=False, alpha=1.0)
            ax_3d.set_xlim(-1, 1)
            ax_3d.set_ylim(-1, 1)
            ax_3d.set_zlim(-1, 1)
            ax_3d.set_box_aspect((1, 1, 1))
            ax_3d.set_xlabel("Inertial X")
            ax_3d.set_ylabel("Inertial Y")
            ax_3d.set_zlabel("Inertial Z")
            ax_3d.set_title(
                f"{title}\n"
                f"t={time[frame_index]:.2f}s  "
                f"angle={np.rad2deg(tr.angle_rad[frame_index]):.3f}°  "
                f"|ω_err|={tr.rate_norm[frame_index]:.4f} rad/s"
            )

            stop = frame_index + 1
            ax_att.plot(time[:stop], np.rad2deg(tr.angle_rad[:stop]), label="Angle error (deg)")
            ax_att.plot(time[:stop], np.rad2deg(tr.rate_norm[:stop]), label="Tracking rate error (deg/s)")
            ax_att.axhline(env.config.task.success_angle_deg, linestyle="--", linewidth=1)
            ax_att.axhline(np.rad2deg(env.config.task.success_rate), linestyle=":", linewidth=1)
            ax_att.set_xlim(0, time[-1] if len(time) > 1 else 1)
            ax_att.set_title("Attitude and rate")
            ax_att.set_xlabel("Time (s)")
            ax_att.grid(True, alpha=0.3)
            ax_att.legend(loc="upper right")

            for axis in range(env.action_size):
                label = (
                    f"motor {axis + 1}"
                    if env.config.control.mode == "motor_direct"
                    else f"body {'XYZ'[axis]}"
                )
                ax_action.plot(time[:stop], tr.command_action[:stop, axis], label=label)
            ax_action.set_ylim(-1.05, 1.05)
            ax_action.set_xlim(0, time[-1] if len(time) > 1 else 1)
            ax_action.set_title(
                "Normalized wheel-motor command"
                if env.config.control.mode == "motor_direct"
                else "Normalized body-torque command"
            )
            ax_action.set_xlabel("Time (s)")
            ax_action.grid(True, alpha=0.3)
            ax_action.legend(loc="upper right")

            for wheel in range(4):
                ax_wheel.plot(
                    time[:stop],
                    tr.wheel_speed[:stop, wheel],
                    label=f"wheel {wheel + 1}",
                )
            ax_wheel.set_xlim(0, time[-1] if len(time) > 1 else 1)
            failed_now = np.where(tr.wheel_mask[frame_index] < 0.5)[0]
            fault_text = (
                "healthy" if failed_now.size == 0
                else "motor off: " + ",".join(str(int(i + 1)) for i in failed_now)
            )
            ax_wheel.set_title(
                f"Wheel speed (rad/s), limit ±{env.config.physics.max_wheel_speed:g} | {fault_text}"
            )
            ax_wheel.set_xlabel("Time (s)")
            ax_wheel.grid(True, alpha=0.3)
            ax_wheel.legend(loc="upper right", ncol=2)

            ax_reward.plot(time[:stop], tr.reward[:stop], label="Step reward")
            ax_reward.plot(time[:stop], cumulative_return[:stop], label="Cumulative return")
            ax_reward.set_xlim(0, time[-1] if len(time) > 1 else 1)
            ax_reward.set_title("Reward")
            ax_reward.set_xlabel("Time (s)")
            ax_reward.grid(True, alpha=0.3)
            ax_reward.legend(loc="best")

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            writer.append_data(frame)
    finally:
        writer.close()
        plt.close(fig)
    return output_path


def save_policy_video(
    *,
    env: SatelliteEnv,
    params: Any,
    apply_fn,
    initial_state: EnvState,
    output_path: str | Path,
    title: str = "Satellite stabilization",
    fps: int = 20,
    frame_stride: int = 4,
) -> Path:
    _, trajectory, _ = evaluate_policy(env, params, apply_fn, initial_state)
    trajectory = jax.device_get(trajectory)
    return save_trajectory_video(
        env,
        trajectory,
        output_path,
        title=title,
        fps=fps,
        frame_stride=frame_stride,
    )
