#
# Convert custom dataset (RGB + pose + depth + point) to COLMAP format for FastGS training.
#
# Input structure:
#   <source>/
#   ├── rgb/frame_XXXXXX.jpg
#   ├── poses/frame_XXXXXX.txt      (4x4 c2w matrix)
#   ├── depth/frame_XXXXXX.png      (uint16)
#   ├── point/frame_XXXXXX.npy      (H, W, 3) world xyz
#   └── intrinsics.yaml             (fx, fy, cx, cy, width, height)
#
# Output structure (COLMAP format):
#   <output>/
#   ├── images/frame_XXXXXX.jpg
#   └── sparse/0/
#       ├── cameras.bin
#       ├── images.bin
#       └── points3D.bin
#

import os
import sys
import struct
import shutil
import argparse
import yaml
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def write_cameras_binary(cameras, path):
    """Write COLMAP cameras.bin format."""
    with open(path, 'wb') as fid:
        fid.write(struct.pack('<Q', len(cameras)))
        for cam_id, cam in cameras.items():
            model_id = cam['model_id']
            fid.write(struct.pack('<iiQQ', cam_id, model_id, cam['width'], cam['height']))
            for p in cam['params']:
                fid.write(struct.pack('<d', p))


def write_images_binary(images, path):
    """Write COLMAP images.bin format."""
    with open(path, 'wb') as fid:
        fid.write(struct.pack('<Q', len(images)))
        for img_id, img in images.items():
            fid.write(struct.pack('<idddddddi',
                                  img['id'],
                                  img['qvec'][0], img['qvec'][1], img['qvec'][2], img['qvec'][3],
                                  img['tvec'][0], img['tvec'][1], img['tvec'][2],
                                  img['camera_id']))
            # Image name (null-terminated)
            fid.write(img['name'].encode('utf-8') + b'\x00')
            # Number of 2D points (empty)
            fid.write(struct.pack('<Q', 0))


def write_points3d_binary(points3d, path):
    """Write COLMAP points3D.bin format."""
    with open(path, 'wb') as fid:
        fid.write(struct.pack('<Q', len(points3d)))
        for pt in points3d:
            fid.write(struct.pack('<QdddBBBd',
                                  pt['id'],
                                  pt['xyz'][0], pt['xyz'][1], pt['xyz'][2],
                                  pt['rgb'][0], pt['rgb'][1], pt['rgb'][2],
                                  pt['error']))
            # Empty track
            fid.write(struct.pack('<Q', 0))


def rotmat2qvec(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def voxel_downsample(points, colors, voxel_size):
    """Voxel grid downsampling. Keeps one point per voxel (closest to voxel center)."""
    # Quantize to voxel grid
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)

    # Create unique voxel keys
    # Use a simple hash: multiply each dimension by a large prime-like number
    keys = (voxel_indices[:, 0] * 73856093 ^
            voxel_indices[:, 1] * 19349663 ^
            voxel_indices[:, 2] * 83492791)

    # Find unique voxels and pick first occurrence
    _, unique_idx = np.unique(keys, return_index=True)

    return points[unique_idx], colors[unique_idx]


def main():
    parser = argparse.ArgumentParser(description="Convert custom dataset to COLMAP format")
    parser.add_argument('--source', '-s', type=str, required=True,
                        help="Source dataset directory")
    parser.add_argument('--output', '-o', type=str, default=None,
                        help="Output COLMAP directory (default: <source>_colmap)")
    parser.add_argument('--voxel_size', type=float, default=0.02,
                        help="Voxel size for point cloud downsampling in meters (default: 0.02)")
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    output = args.output if args.output else source + "_colmap"
    output = os.path.abspath(output)

    # --- Read intrinsics ---
    with open(os.path.join(source, 'intrinsics.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    cam = cfg['camera_params']
    fx, fy = cam['fx'], cam['fy']
    cx, cy = cam['cx'], cam['cy']
    width, height = cam['image_width'], cam['image_height']
    # Scale intrinsics to match training image resolution
    _test_rgb = os.path.join(source, 'rgb')
    if os.path.isdir(_test_rgb):
        _train_imgs = sorted([f for f in os.listdir(_test_rgb) if f.endswith('.jpg')])
        if _train_imgs:
            from PIL import Image as _PILImg2
            _ti = _PILImg2.open(os.path.join(_test_rgb, _train_imgs[0]))
            _tw, _th = _ti.size
            if _tw != width or _th != height:
                sx, sy = _tw / width, _th / height
                fx *= sx
                fy *= sy
                cx *= sx
                cy *= sy
                width, height = _tw, _th
                print(f"Scaled intrinsics to match training: {width}x{height} ({sx:.3f}x{sy:.3f})")

    print(f"Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.1f}, cy={cy:.1f}, "
          f"size={width}x{height}")

    # --- Discover frames ---
    rgb_dir = os.path.join(source, 'rgb')
    frame_names = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
    num_frames = len(frame_names)
    print(f"Found {num_frames} frames")

    # --- Create output directories ---
    images_out = os.path.join(output, 'images')
    sparse_out = os.path.join(output, 'sparse', '0')
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(sparse_out, exist_ok=True)

    # --- Write cameras.bin ---
    # PINHOLE model: model_id=1, params=[fx, fy, cx, cy]
    cameras = {
        1: {
            'model_id': 1,  # PINHOLE
            'width': width,
            'height': height,
            'params': [fx, fy, cx, cy],
        }
    }
    cameras_path = os.path.join(sparse_out, 'cameras.bin')
    write_cameras_binary(cameras, cameras_path)
    print(f"Wrote cameras.bin (1 camera, PINHOLE)")

    # --- Process each frame: copy image, convert pose ---
    images = {}
    all_points = []
    all_colors = []

    print("Processing frames...")
    for idx, frame_name in enumerate(tqdm(frame_names)):
        stem = Path(frame_name).stem  # e.g. frame_000000

        # Copy image
        src_img = os.path.join(source, 'rgb', frame_name)
        dst_img = os.path.join(images_out, frame_name)
        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)

        # Load pose (c2w)
        pose_path = os.path.join(source, 'poses', f'{stem}.txt')
        if not os.path.exists(pose_path):
            continue  # skip frames without pose (e.g., deleted by motion filter)
        c2w = np.loadtxt(pose_path)  # 4x4 camera-to-world

        # c2w -> w2c (world-to-camera)
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        t = w2c[:3, 3]

        # Rotation matrix -> quaternion
        qvec = rotmat2qvec(R)

        images[idx + 1] = {
            'id': idx + 1,
            'qvec': qvec,
            'tvec': t,
            'camera_id': 1,
            'name': frame_name,
        }

        # Load point cloud for this frame
        point_path = os.path.join(source, 'point', f'{stem}.npy')
        if os.path.exists(point_path):
            pts = np.load(point_path)  # (H, W, 3)
            # Subsample: take every 4th pixel to keep memory manageable
            pts = pts[::4, ::4, :].reshape(-1, 3)

            # Filter out invalid points (zeros or NaNs)
            valid = np.all(pts != 0, axis=1) & ~np.any(np.isnan(pts), axis=1)
            pts = pts[valid]

            if len(pts) > 0:
                # Load corresponding RGB for colors
                color_src = os.path.join(source, 'frames', frame_name)
                if not os.path.exists(color_src):
                    color_src = src_img
                rgb_img = np.array(Image.open(color_src))  # (H, W, 3)
                rgb_sub = rgb_img[::4, ::4, :].reshape(-1, 3)
                rgb_sub = rgb_sub[valid]

                all_points.append(pts)
                all_colors.append(rgb_sub)

    # --- Write images.bin ---
    images_path = os.path.join(sparse_out, 'images.bin')
    write_images_binary(images, images_path)
    print(f"Wrote images.bin ({len(images)} images)")

    # --- Merge and downsample point cloud ---
    print("Merging point clouds...")
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors).astype(np.uint8)
    print(f"  Total points before downsampling: {len(all_points):,}")

    # Voxel downsample
    all_points, all_colors = voxel_downsample(all_points, all_colors, args.voxel_size)
    print(f"  Points after voxel downsampling (voxel={args.voxel_size}m): {len(all_points):,}")

    # Build points3D list
    points3d = []
    for i in range(len(all_points)):
        points3d.append({
            'id': i + 1,
            'xyz': all_points[i],
            'rgb': all_colors[i],
            'error': 1.0,
        })

    # --- Write points3D.bin ---
    points_path = os.path.join(sparse_out, 'points3D.bin')
    print(f"Writing points3D.bin ({len(points3d):,} points)...")
    write_points3d_binary(points3d, points_path)

    print(f"\nDone! COLMAP dataset created at: {output}")
    print(f"\nTo train with FastGS:")
    print(f"  python train.py -s {output} -i images --eval --test_iterations 30000 "
          f"--densification_interval 500 --optimizer_type default --grad_abs_thresh 0.0012")


if __name__ == '__main__':
    main()
