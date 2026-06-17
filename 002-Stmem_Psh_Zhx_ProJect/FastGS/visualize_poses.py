"""Visualize camera trajectory from pose files (4x4 transformation matrices).

Usage:
    python3 visualize_poses.py /path/to/poses/
    python3 visualize_poses.py /path/to/poses/ --save output.png
"""

import argparse
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection


def load_poses(pose_dir):
    files = sorted(f for f in os.listdir(pose_dir) if f.endswith('.txt'))
    poses = []
    for f in files:
        P = np.loadtxt(os.path.join(pose_dir, f))
        if P.shape == (4, 4):
            poses.append(P)
    return np.array(poses)


def plot_trajectory(poses, save_path=None):
    translations = poses[:, :3, 3]
    xs, ys, zs = translations[:, 0], translations[:, 1], translations[:, 2]
    n = len(xs)
    colors = plt.cm.coolwarm(np.linspace(0, 1, n))

    diffs = np.diff(translations, axis=0)
    total_length = np.sum(np.linalg.norm(diffs, axis=1))
    loop_error = np.linalg.norm(translations[-1] - translations[0])

    fig, (ax2d, ax3d) = plt.subplots(1, 2, figsize=(14, 6),
                                      subplot_kw={'projection': None})

    # --- 2D top-down: X (right) as horizontal, Z (forward) as vertical ---
    for i in range(n - 1):
        ax2d.plot(xs[i:i+2], zs[i:i+2], color=colors[i], linewidth=1.5)
    ax2d.scatter(xs[0], zs[0], color='green', s=100, marker='o', zorder=5, label='Start')
    ax2d.scatter(xs[-1], zs[-1], color='red', s=100, marker='x', zorder=5, label='End')
    ax2d.set_xlabel('X (m)')
    ax2d.set_ylabel('Z (m)')
    ax2d.set_title('2D Trajectory (Top-down)')
    ax2d.set_aspect('equal')
    ax2d.legend()

    # --- 3D trajectory: natural camera coordinates (X right, Y down, Z forward) ---
    ax3d.remove()
    ax3d = fig.add_subplot(122, projection='3d')

    for i in range(n - 1):
        ax3d.plot(xs[i:i+2], zs[i:i+2], ys[i:i+2], color=colors[i], linewidth=1.5)
    ax3d.scatter(xs[0], zs[0], ys[0], color='green', s=100, marker='o', label='Start', zorder=5)
    ax3d.scatter(xs[-1], zs[-1], ys[-1], color='red', s=100, marker='x', label='End', zorder=5)

    # equal aspect ratio for real 3D feel
    max_range = np.array([xs.max()-xs.min(), zs.max()-zs.min(), ys.max()-ys.min()]).max() / 2
    mid_x, mid_z, mid_y = (xs.max()+xs.min())/2, (zs.max()+zs.min())/2, (ys.max()+ys.min())/2
    ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3d.set_ylim(mid_z - max_range, mid_z + max_range)
    ax3d.set_zlim(mid_y - max_range, mid_y + max_range)

    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Z (m)')
    ax3d.set_zlabel('Y (m)')
    ax3d.set_title('3D Trajectory')
    ax3d.legend()

    fig.suptitle(f'{n} frames | Length: {total_length:.2f}m | Loop error: {loop_error:.2f}m',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize camera trajectory from pose files')
    parser.add_argument('pose_dir', help='Directory containing 4x4 pose txt files')
    parser.add_argument('--save', help='Output image path (default: <pose_dir>/trajectory.png)')
    args = parser.parse_args()

    poses = load_poses(args.pose_dir)
    print(f"Loaded {len(poses)} poses from {args.pose_dir}")

    translations = poses[:, :3, 3]
    diffs = np.diff(translations, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    print(f"  Trajectory length: {np.sum(dists):.2f}m")
    print(f"  Loop error: {np.linalg.norm(translations[-1] - translations[0]):.2f}m")
    print(f"  Max frame-to-frame jump: {dists.max():.3f}m (frame {dists.argmax()})")

    save_path = args.save or os.path.join(os.path.dirname(os.path.abspath(args.pose_dir)), 'trajectory.png')
    plot_trajectory(poses, save_path=save_path)


if __name__ == '__main__':
    main()
