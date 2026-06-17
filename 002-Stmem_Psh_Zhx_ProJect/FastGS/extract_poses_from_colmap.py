"""
Extract rgb/, poses/, depth/, point/, intrinsics.yaml from COLMAP SfM output.

Input:
  - A COLMAP dataset directory (processed by convert.py: has sparse/0/, images/)
  - An optional source dataset with depth/ files for point generation

Output: rgb/, poses/, depth/, point/, intrinsics.yaml — same format as lingbot-style datasets.

Usage:
    # Basic: extract rgb + poses + intrinsics
    python extract_poses_from_colmap.py -s datasets/lingbot300

    # Full: also copy depth and generate point files
    python extract_poses_from_colmap.py -s datasets/lingbot300 \
        --source_dataset /path/to/original/lingbot300 \
        --generate_points

The output can then be fed into convert_custom_to_colmap.py:
    python convert_custom_to_colmap.py -s datasets/lingbot300_extracted -o output_colmap
"""

import os
import sys
import shutil
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add FastGS root to path for colmap_loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat


def read_sensor_intrinsics(yaml_path):
    """Read sensor intrinsics from YAML file."""
    import yaml
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cam = cfg['camera_params']
    return {
        'fx': cam['fx'],
        'fy': cam['fy'],
        'cx': cam['cx'],
        'cy': cam['cy'],
        'width': cam['image_width'],
        'height': cam['image_height'],
        'png_depth_scale': cam.get('png_depth_scale', 1000.0),
    }


def generate_points(depth_dir, poses_dir, output_dir, intrinsics):
    """Generate point/ files from depth + poses + sensor intrinsics."""
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    scale = intrinsics['png_depth_scale']
    height = intrinsics['height']
    width = intrinsics['width']

    # Pre-compute pixel grid
    u_coords = np.arange(width, dtype=np.float32)  # (W,)
    v_coords = np.arange(height, dtype=np.float32)  # (H,)
    uu, vv = np.meshgrid(u_coords, v_coords)  # (H, W)

    point_out = os.path.join(output_dir, "point")
    os.makedirs(point_out, exist_ok=True)

    # Find matching depth and pose files
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(('.png', '.npy'))])
    print(f"Generating points for {len(depth_files)} frames...")

    for depth_file in tqdm(depth_files):
        stem = os.path.splitext(depth_file)[0]
        pose_path = os.path.join(poses_dir, f"{stem}.txt")

        if not os.path.exists(pose_path):
            print(f"  Warning: no pose for {stem}, skipping")
            continue

        # Load depth
        depth_path = os.path.join(depth_dir, depth_file)
        if depth_file.endswith('.png'):
            depth = np.array(Image.open(depth_path)).astype(np.float32) / scale  # meters
        else:
            depth = np.load(depth_path).astype(np.float32)

        # Load c2w pose
        c2w = np.loadtxt(pose_path)  # 4x4
        R = c2w[:3, :3]
        t = c2w[:3, 3]

        # Deproject: pixel (u,v) + depth → camera coords → world coords
        # X_cam = [(u - cx) * d / fx, (v - cy) * d / fy, d]
        X_cam = np.stack([
            (uu - cx) * depth / fx,
            (vv - cy) * depth / fy,
            depth,
        ], axis=-1)  # (H, W, 3)

        # Transform to world: X_world = R @ X_cam + t
        X_world = np.einsum('ij,hwj->hwi', R, X_cam) + t  # (H, W, 3)

        # Save
        out_path = os.path.join(point_out, f"{stem}.npy")
        np.save(out_path, X_world.astype(np.float32))


def extract_poses(colmap_dir, output_dir, source_dataset=None, generate_points_flag=False):
    sparse_dir = os.path.join(colmap_dir, "sparse", "0")
    images_bin = os.path.join(sparse_dir, "images.bin")
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    colmap_images_dir = os.path.join(colmap_dir, "images")

    # Verify required files exist
    for f in [images_bin, cameras_bin]:
        if not os.path.exists(f):
            print(f"Error: {f} not found. Run convert.py first.")
            sys.exit(1)

    # Read COLMAP data
    colmap_images = read_extrinsics_binary(images_bin)
    colmap_cameras = read_intrinsics_binary(cameras_bin)

    # Create output directories
    rgb_out = os.path.join(output_dir, "rgb")
    poses_out = os.path.join(output_dir, "poses")
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(poses_out, exist_ok=True)

    # Sort images by name for consistent ordering
    sorted_images = sorted(colmap_images.values(), key=lambda x: x.name)

    print(f"Extracting {len(sorted_images)} images...")

    for img in sorted_images:
        # Build w2c matrix from quaternion + translation
        R = qvec2rotmat(img.qvec)
        t = img.tvec

        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        # c2w = inv(w2c)
        c2w = np.linalg.inv(w2c)

        # Write pose
        stem = os.path.splitext(img.name)[0]
        pose_path = os.path.join(poses_out, f"{stem}.txt")
        np.savetxt(pose_path, c2w, fmt="%.18e")

        # Copy RGB image
        src_img = os.path.join(colmap_images_dir, img.name)
        dst_img = os.path.join(rgb_out, img.name)
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        else:
            print(f"  Warning: image not found: {src_img}")

    # Write intrinsics.yaml — use sensor intrinsics if source_dataset provided
    if source_dataset:
        sensor_yaml = os.path.join(source_dataset, "intrinsics.yaml")
        if os.path.exists(sensor_yaml):
            shutil.copy2(sensor_yaml, os.path.join(output_dir, "intrinsics.yaml"))
            intrinsics = read_sensor_intrinsics(sensor_yaml)
            fx, fy = intrinsics['fx'], intrinsics['fy']
            cam_w, cam_h = intrinsics['width'], intrinsics['height']
            print(f"  Using sensor intrinsics: fx={fx:.2f}, fy={fy:.2f}, {cam_w}x{cam_h}")
        else:
            print(f"  Warning: no intrinsics.yaml in {source_dataset}, using COLMAP intrinsics")
            fx, fy, cam_w, cam_h = _write_colmap_intrinsics(colmap_cameras, output_dir)
    else:
        fx, fy, cam_w, cam_h = _write_colmap_intrinsics(colmap_cameras, output_dir)

    # Copy depth from source dataset
    if source_dataset:
        src_depth = os.path.join(source_dataset, "depth")
        dst_depth = os.path.join(output_dir, "depth")
        if os.path.isdir(src_depth):
            shutil.copytree(src_depth, dst_depth, dirs_exist_ok=True)
            n_depth = len(os.listdir(dst_depth))
            print(f"  depth/:      {n_depth} files copied")
        else:
            print(f"  Warning: no depth/ in {source_dataset}")

    # Generate point files
    if generate_points_flag and source_dataset:
        src_depth = os.path.join(source_dataset, "depth")
        if os.path.isdir(src_depth) and os.path.exists(os.path.join(source_dataset, "intrinsics.yaml")):
            intrinsics = read_sensor_intrinsics(os.path.join(source_dataset, "intrinsics.yaml"))
            generate_points(src_depth, poses_out, output_dir, intrinsics)
            n_points = len(os.listdir(os.path.join(output_dir, "point")))
            print(f"  point/:      {n_points} files generated")
        else:
            print("  Warning: cannot generate points — need depth/ and intrinsics.yaml in source")

    # Summary
    print(f"\nDone! Extracted to: {output_dir}")
    print(f"  rgb/:       {len(sorted_images)} images")
    print(f"  poses/:     {len(sorted_images)} pose files")
    print(f"  intrinsics: fx={fx:.2f}, fy={fy:.2f}")


def _write_colmap_intrinsics(colmap_cameras, output_dir):
    """Write intrinsics.yaml from COLMAP cameras.bin. Returns (fx, fy, w, h)."""
    cam = list(colmap_cameras.values())[0]
    if cam.model == "PINHOLE":
        fx, fy, cx, cy = cam.params
    elif cam.model == "SIMPLE_PINHOLE":
        fx = cam.params[0]; fy = fx; cx, cy = cam.params[1], cam.params[2]
    else:
        fx, fy, cx, cy = cam.params[:4]
        print(f"  Note: camera model is {cam.model}, using first 4 params as fx,fy,cx,cy")

    yaml_content = f"""camera_params:
  crop_edge: 0
  cx: {cx}
  cy: {cy}
  fx: {fx}
  fy: {fy}
  image_height: {cam.height}
  image_width: {cam.width}
  png_depth_scale: 1000.0
dataset_name: mydata
"""
    with open(os.path.join(output_dir, "intrinsics.yaml"), "w") as f:
        f.write(yaml_content)
    return fx, fy, cam.width, cam.height


def main():
    parser = argparse.ArgumentParser(description="Extract rgb + poses + depth + point + intrinsics from COLMAP output")
    parser.add_argument("--source", "-s", type=str, required=True,
                        help="COLMAP dataset directory (processed by convert.py)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: <source>_extracted)")
    parser.add_argument("--source_dataset", type=str, default=None,
                        help="Original dataset with depth/ and intrinsics.yaml (for point generation)")
    parser.add_argument("--generate_points", action="store_true",
                        help="Generate point/ files from depth + poses + sensor intrinsics")
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    output = args.output if args.output else source + "_extracted"
    output = os.path.abspath(output)

    extract_poses(source, output, args.source_dataset, args.generate_points)


if __name__ == "__main__":
    main()
