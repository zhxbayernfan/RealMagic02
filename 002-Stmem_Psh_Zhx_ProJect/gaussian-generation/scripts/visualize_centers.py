"""
可视化场景图中物体的中心位置，将中心位置渲染为小的点云，用于可视化场景图的效果
python scripts/visualize_centers.py configs/mydata/dgsg.py
"""
import json
import numpy as np
import os
import sys
from plyfile import PlyData, PlyElement
import argparse
from importlib.machinery import SourceFileLoader

def create_debug_ply(ply_path, json_path, output_path):
    print(f"Loading original PLY from {ply_path}...")
    plydata = PlyData.read(ply_path)
    
    # Get original properties
    orig_vertex = plydata['vertex']
    count = orig_vertex.count
    
    # Extract existing data columns
    # We need to keep all properties to make it a valid splat file
    prop_names = [p.name for p in orig_vertex.properties]
    data_dict = {name: orig_vertex[name] for name in prop_names}
    
    print(f"Loading JSON from {json_path}...")
    with open(json_path, 'r') as f:
        scene_graph = json.load(f)
    
    nodes = scene_graph['nodes']
    
    # Create new points for centers
    # We will create a small cluster of points for each center to make it visible as a "ball"
    new_points = []
    
    print("Generating marker points for object centers...")
    for node in nodes:
        center = node['center']
        category = node['category']

        # if node['idx'] != 1: continue
        
        # Only visualize Chair
        # if category != 'Chair': continue
        
        cx, cy, cz = center
        
        marker_radius = 0.01   # meters (例：0.05 = 5 cm)
        points_per_center = 200

        # Create a small cloud of 200 points around the center (More points for better visibility)
        for _ in range(points_per_center):
            u = np.random.rand()
            r = marker_radius * (u ** (1.0/3.0))     # cubic root -> uniform in volume
            dir = np.random.normal(size=3)
            dir /= np.linalg.norm(dir)
            offset = dir * r
            px, py, pz = cx + offset[0], cy + offset[1], cz + offset[2]
            
            point_data = {}
            for name in prop_names:
                point_data[name] = 0.0 # Default zero
            
            # Set coordinates
            point_data['x'] = px
            point_data['y'] = py
            point_data['z'] = pz
            
            # Set color to GREEN
            point_data['f_dc_0'] = -2.0 
            point_data['f_dc_1'] = 2.0 # Very bright Green
            point_data['f_dc_2'] = -2.0 
            
            # Make it opaque and large scale (Larger splats)
            point_data['opacity'] = 100.0 # High logit for opacity
            point_data['scale_0'] = -4.0 # Much larger scale (exp(-2) approx 0.13)
            point_data['scale_1'] = -4.0
            point_data['scale_2'] = -4.0
            point_data['rot_0'] = 1.0
            
            new_points.append(point_data)

    print(f"Added {len(new_points)} marker points.")
    
    # Merge data
    final_count = count + len(new_points)
    
    # Prepare arrays for PlyElement
    # We need to construct a structured array
    # Fix: PlyProperty.dtype is a method in some versions or property in others, but we need the numpy string
    dtype_list = []
    for p in orig_vertex.properties:
        # Map plyfile types to numpy types
        t = p.val_dtype
        dtype_list.append((p.name, t))
        
    final_array = np.zeros(final_count, dtype=dtype_list)
    
    # Copy original data
    for name in prop_names:
        final_array[name][:count] = data_dict[name]
        
    # Fill new data
    for i, p_data in enumerate(new_points):
        idx = count + i
        for name in prop_names:
            if name in p_data:
                final_array[name][idx] = p_data[name]
                
    # Save
    print(f"Saving to {output_path}...")
    el = PlyElement.describe(final_array, 'vertex')
    PlyData([el]).write(output_path)
    print("Done!")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    experiment = SourceFileLoader(os.path.basename(args.config), args.config).load_module()
    config = experiment.config
    workdir = config['workdir']
    group_name = os.path.basename(workdir)
    run_name = config['run_name']
    experiment_path = os.path.join(workdir, run_name)
    ply_path = os.path.join(experiment_path, f"dgsg_{group_name}_{run_name}.ply")
    json_path = os.path.join(experiment_path, "scene_graph.json")
    output_path = os.path.join(experiment_path, f"dgsg_{group_name}_{run_name}_vis_center.ply")
    
    # Arguments are parsed via argparse, manual sys.argv override removed to fix config loading
    pass

    create_debug_ply(ply_path, json_path, output_path)
