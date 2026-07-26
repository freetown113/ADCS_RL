import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import imageio


def save_simulation_video(historique_theta, target, output_filename="single_wheel_control.mp4", fps=20, step=10):
    if len(historique_theta.shape) > 1:
        thetas = np.array(historique_theta[:, 0])
    else:
        thetas = np.array(historique_theta)
        
    num_steps = len(thetas)
    frames = []

    fig = plt.figure(figsize=(12, 5))
    ax_polar = fig.add_subplot(121, projection='polar')
    ax_time = fig.add_subplot(122)

    for step in range(0, num_steps, step):
        ax_polar.cla()
        ax_time.cla()
        
        current_theta = thetas[step]
        
        ax_polar.set_title("single wheel control task", pad=15)
        ax_polar.plot([target, target], [0, 1], color='green', linestyle='--', alpha=0.6, label='target')
        ax_polar.plot([0, current_theta], [0, 1], color='red', linewidth=3, label='heading')
        ax_polar.scatter(current_theta, 1, color='red', s=50)
        
        ax_polar.set_yticklabels([])
        ax_polar.set_rmax(1.1)
        ax_polar.legend(loc="upper left")

        ax_time.set_title("angle evolution")
        ax_time.set_xlabel("steps")
        ax_time.set_ylabel("radians")
        ax_time.set_xlim([0, num_steps])
        ax_time.set_ylim([-np.pi - 0.2, np.pi + 0.2])
        
        if target >= np.pi:
            tgt = -(np.pi*2 - target)
        else:
            tgt = target
        ax_time.axhline(tgt, color='green', linestyle='--', alpha=0.5)

        ax_time.plot(range(step + 1), thetas[:step + 1], color='blue', linewidth=2)
        ax_time.scatter(step, current_theta, color='red', s=40, zorder=5)
        
        ax_time.grid(True, alpha=0.3)

        fig.tight_layout()
        
        fig.canvas.draw()
        rgba_buffer = fig.canvas.buffer_rgba()
        img_rgba = np.array(rgba_buffer, dtype=np.uint8, copy=True)
        frame = img_rgba[..., :3].copy() 
        frames.append(frame)
        
    plt.close(fig)

    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    imageio.mimsave(output_filename, frames, fps=fps)
    print(f"video successfully saved to {output_filename}")
