"""
将生成好的高斯球数据导出为PLY格式，用于supersplat打开
python scripts/export_ply.py configs/mydata/dgsg.py
"""

import os
import argparse
from importlib.machinery import SourceFileLoader

import numpy as np
from plyfile import PlyData, PlyElement

# Spherical harmonic constant
C0 = 0.28209479177387814


def rgb_to_spherical_harmonic(rgb):
    return (rgb-0.5) / C0


def spherical_harmonic_to_rgb(sh):
    return sh*C0 + 0.5


def save_ply(path, means, scales, rotations, rgbs, opacities, normals=None):
    if normals is None:
        normals = np.zeros_like(means)

    colors = rgb_to_spherical_harmonic(rgbs)

    if scales.shape[1] == 1:
        scales = np.tile(scales, (1, 3))

    attrs = ['x', 'y', 'z',
             'nx', 'ny', 'nz',
             'f_dc_0', 'f_dc_1', 'f_dc_2',
             'opacity',
             'scale_0', 'scale_1', 'scale_2',
             'rot_0', 'rot_1', 'rot_2', 'rot_3',]

    dtype_full = [(attribute, 'f4') for attribute in attrs]
    elements = np.empty(means.shape[0], dtype=dtype_full)

    attributes = np.concatenate((means, normals, colors, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

    print(f"Saved PLY format Splat to {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument("--mask", action="store_true", help="Export with instance mask colors.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load SplaTAM config
    experiment = SourceFileLoader(os.path.basename(args.config), args.config).load_module()
    config = experiment.config
    work_path = config['workdir']
    group_name = os.path.basename(work_path)
    run_name = config['run_name']
    params_path = os.path.join(work_path, run_name, "params_with_idx.npz")

    params = dict(np.load(params_path, allow_pickle=True))
    means = params['means3D']
    scales = params['log_scales']
    rotations = params['unnorm_rotations']
    opacities = params['logit_opacities']

    if args.mask:
        obj_indices = params['object_idx'].squeeze()
        unique_ids = np.unique(obj_indices)
        original_rgbs = params['rgb_colors']
        
        # 固定随机种子，确保每次导出的颜色一致
        np.random.seed(42)
        # 为每个唯一的物体 ID 生成一个随机的 RGB 颜色 (0~1)
        color_map = {uid: np.random.rand(3) for uid in unique_ids}
        
        # 更高效的 numpy 向量化操作
        # 先把所有点都赋上随机的 mask 颜色
        rgbs = np.array([color_map[idx] for idx in obj_indices])
        
        # 找出 idx 为 0 的点的掩码
        bg_mask = (obj_indices == 0)
        
        # 将这些点替换为原来的颜色
        rgbs[bg_mask] = original_rgbs[bg_mask]

        ply_path = os.path.join(work_path, run_name, f"dgsg_{group_name}_{run_name}_mask.ply")
    else:
        rgbs = params['rgb_colors']
        ply_path = os.path.join(work_path, run_name, f"dgsg_{group_name}_{run_name}.ply")

    save_ply(ply_path, means, scales, rotations, rgbs, opacities)