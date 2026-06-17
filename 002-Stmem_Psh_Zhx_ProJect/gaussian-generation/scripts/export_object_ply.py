"""
将生成好的高斯球数据按物体导出为多个PLY格式，用于supersplat打开
python scripts/export_object_ply.py configs/mydata/dgsg.py
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load SplaTAM config
    experiment = SourceFileLoader(os.path.basename(args.config), args.config).load_module()
    config = experiment.config
    work_path = config['workdir']
    run_name = config['run_name']
    
    # 兼容配置里的相对路径情况，最好转为绝对路径或依赖当前执行路径
    # experiments对应的目录为 work_path/run_name
    experiment_dir = os.path.join(work_path, run_name)
    params_path = os.path.join(experiment_dir, "params_with_idx.npz")

    print(f"Loading {params_path}...")
    params = dict(np.load(params_path, allow_pickle=True))
    
    # 尝试加载 scene_graph.json 获取中心点
    import json
    json_path = os.path.join(experiment_dir, "scene_graph.json")
    center_map = {}
    if os.path.exists(json_path):
        print(f"Loading scene graph from {json_path}...")
        with open(json_path, 'r') as f:
            scene_graph = json.load(f)
        nodes = scene_graph.get('nodes', [])
        for node in nodes:
            if 'idx' in node and 'center' in node:
                center_map[node['idx']] = node['center']
    else:
        print(f"Warning: {json_path} not found. Object centers will not be visualized.")

    means_all = params['means3D']
    scales_all = params['log_scales']
    rotations_all = params['unnorm_rotations']
    rgbs_all = params['rgb_colors']
    opacities_all = params['logit_opacities']
    
    # 确保有 object_idx
    if 'object_idx' not in params:
        raise ValueError(f"'object_idx' not found in {params_path}")
        
    object_idx_all = params['object_idx']

    # Create output directory: ply
    ply_dir = os.path.join(experiment_dir, "ply")
    os.makedirs(ply_dir, exist_ok=True)
    
    unique_ids = np.unique(object_idx_all)
    print(f"Found {len(unique_ids)} objects. Exporting to {ply_dir}...")

    for obj_id in unique_ids:
        # Skip background / invalid object index 0
        if obj_id == 0:
            continue

        # Create mask for current object
        # object_idx_all shape is typically (N, 1) or (N,), so squeeze ensures 1D mask
        mask = (object_idx_all == obj_id).squeeze()
        
        # Extract properties
        means = means_all[mask]
        scales = scales_all[mask]
        rotations = rotations_all[mask]
        rgbs = rgbs_all[mask]
        opacities = opacities_all[mask]
        
        # Handle shape cases
        if means.ndim == 1:
            means = means[np.newaxis, :]
            scales = scales[np.newaxis, :]
            rotations = rotations[np.newaxis, :]
            rgbs = rgbs[np.newaxis, :]
            opacities = opacities[np.newaxis, :]

        if len(means) == 0:
            continue
            
        # Add center points if available
        if int(obj_id) in center_map:
            center = center_map[int(obj_id)]
            cx, cy, cz = center
            
            marker_radius = 0.01
            points_per_center = 200
            
            new_means = []
            new_scales = []
            new_rotations = []
            new_rgbs = []
            new_opacities = []
            
            for _ in range(points_per_center):
                u = np.random.rand()
                r = marker_radius * (u ** (1.0/3.0))
                dir = np.random.normal(size=3)
                dir /= np.linalg.norm(dir)
                offset = dir * r
                
                new_means.append([cx + offset[0], cy + offset[1], cz + offset[2]])
                
                # Colors: f_dc_0 = -2.0, f_dc_1 = 2.0, f_dc_2 = -2.0 -> approx [0.0, 1.0, 0.0] bright green
                # Reverse rgb_to_spherical_harmonic: rgb = f_dc * C0 + 0.5
                r_c = -2.0 * C0 + 0.5
                g_c = 2.0 * C0 + 0.5
                b_c = -2.0 * C0 + 0.5
                new_rgbs.append([r_c, g_c, b_c])
                
                new_opacities.append([100.0])
                if scales.shape[1] == 1:
                    new_scales.append([-4.0])
                else:
                    new_scales.append([-4.0, -4.0, -4.0])
                new_rotations.append([1.0, 0.0, 0.0, 0.0])
                
            means = np.concatenate([means, np.array(new_means, dtype=means.dtype)], axis=0)
            scales = np.concatenate([scales, np.array(new_scales, dtype=scales.dtype)], axis=0)
            rotations = np.concatenate([rotations, np.array(new_rotations, dtype=rotations.dtype)], axis=0)
            rgbs = np.concatenate([rgbs, np.array(new_rgbs, dtype=rgbs.dtype)], axis=0)
            opacities = np.concatenate([opacities, np.array(new_opacities, dtype=opacities.dtype)], axis=0)
            
        ply_path = os.path.join(ply_dir, f"{int(obj_id)}.ply")
        save_ply(ply_path, means, scales, rotations, rgbs, opacities)
        
    print("All objects exported successfully!")
