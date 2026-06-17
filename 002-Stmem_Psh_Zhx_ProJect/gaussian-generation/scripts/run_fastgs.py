"""
Run FastGS 3D Gaussian Splatting training.

Two modes:
  --mode custom  (default)   Use existing poses+point cloud (fast, for point cloud streaming)
  --mode colmap              Run COLMAP SfM to recompute poses (accurate, for 3DGS rendering)

Usage:
    python run_fastgs.py --images /path/to/data --output /path/to/output
    python run_fastgs.py --images /path/to/data --output /path/to/output --mode colmap --ply /path/to/output.ply
"""

import argparse
import os
import sys
import shutil
import subprocess

# ─── Resolve FastGS path (local to this project) ────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FASTGS_DIR = os.path.join(PROJECT_ROOT, "FastGS")

if not os.path.isdir(FASTGS_DIR):
    for candidate in [os.path.join(os.path.dirname(PROJECT_ROOT), "FastGS"), "/home/liangjiahua/FastGS"]:
        if os.path.isdir(candidate):
            FASTGS_DIR = candidate
            break

if not os.path.isdir(FASTGS_DIR):
    print(f"Error: FastGS directory not found. Searched: {FASTGS_DIR}")
    sys.exit(1)

DEFAULTS = {
    "iterations": 30000, "voxel_size": 0.02, "densification_interval": 500,
    "grad_abs_thresh": 0.0012, "optimizer_type": "default",
    "mult": 0.5, "dense": 0.001, "loss_thresh": 0.1,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run FastGS training")
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--ply", type=str, default=None)
    parser.add_argument("--mode", type=str, default="custom", choices=["custom", "colmap"])
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument("--densification_interval", type=int, default=None)
    parser.add_argument("--grad_abs_thresh", type=float, default=None)
    parser.add_argument("--mult", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    return parser.parse_args()

def get_params(args):
    params = dict(DEFAULTS)
    for k in DEFAULTS:
        v = getattr(args, k, None)
        if v is not None: params[k] = v
    return params

def prep_input_dir(images_dir, output_dir):
    """Create input/ symlinks from frames for COLMAP convert.py."""
    input_dir = os.path.join(output_dir, "input")
    os.makedirs(input_dir, exist_ok=True)

    # Find frames
    for sub in ["frames", "rgb", "images"]:
        src = os.path.join(images_dir, sub)
        if os.path.isdir(src):
            for f in os.listdir(src):
                src_path = os.path.join(src, f)
                dst = os.path.join(input_dir, f)
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(src_path), dst)
            break
    return input_dir

def find_intrinsics(images_dir):
    """Find intrinsics file, ensure YAML format (convert from JSON if needed)."""
    yaml_path = os.path.join(images_dir, "intrinsics.yaml")
    json_path = os.path.join(images_dir, "intrinsics.json")

    if os.path.isfile(yaml_path):
        return yaml_path
    if os.path.isfile(json_path):
        import json
        with open(json_path) as f:
            d = json.load(f)
        with open(yaml_path, "w") as f:
            f.write("camera_params:\n")
            f.write(f"  fx: {d.get('fx', 1000)}\n")
            f.write(f"  fy: {d.get('fy', 1000)}\n")
            f.write(f"  cx: {d.get('cx', 640)}\n")
            f.write(f"  cy: {d.get('cy', 360)}\n")
            f.write(f"  image_width: {d.get('w', d.get('width', 1280))}\n")
            f.write(f"  image_height: {d.get('h', d.get('height', 720))}\n")
        return yaml_path
    return None

def run_colmap_sfm(images_dir, output_dir, intrinsics_path):
    """Run COLMAP SfM via FastGS convert.py to get accurate poses."""
    input_dir = prep_input_dir(images_dir, output_dir)
    convert_script = os.path.join(FASTGS_DIR, "convert.py")

    cmd = [
        sys.executable, convert_script,
        "-s", output_dir,
        "--sequential",
        "--camera", "PINHOLE",
        "--no_gpu",
    ]
    if intrinsics_path:
        cmd += ["--intrinsics", intrinsics_path]

    print(f"\n{'='*60}")
    print(f"Step 1: Running COLMAP SfM for accurate poses")
    print(f"  Data dir:  {output_dir}")
    print(f"  Intrinsics: {intrinsics_path or 'auto'}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)

    # Verify output exists (don't trust truncated exit codes from os.system in convert.py)
    sparse0 = os.path.join(output_dir, "sparse", "0")
    ok = (os.path.isdir(sparse0)
          and os.path.isfile(os.path.join(sparse0, "cameras.bin"))
          and os.path.isfile(os.path.join(sparse0, "images.bin")))
    if result.returncode != 0 or not ok:
        print(f"\nError: COLMAP SfM failed (exit code {result.returncode})")
        sys.exit(1)

    # convert.py runs image_undistorter → restores sparse/0/
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    images_dir_out = os.path.join(output_dir, "images")

    # If no images/ dir, create from input
    if not os.path.isdir(images_dir_out):
        os.makedirs(images_dir_out, exist_ok=True)
        for f in os.listdir(input_dir):
            src = os.path.join(input_dir, f)
            dst = os.path.join(images_dir_out, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    return output_dir


def run_custom_conversion(images_dir, output_dir, voxel_size):
    """Use existing lingbot poses + point cloud."""
    source_dir = os.path.join(output_dir, "fastgs_dataset")
    os.makedirs(source_dir, exist_ok=True)

    # Link frames → rgb
    rgb_dir = os.path.join(source_dir, "rgb")
    for sub in ["rgb", "frames", "images"]:
        src = os.path.join(images_dir, sub)
        if os.path.isdir(src):
            if os.path.lexists(rgb_dir): os.unlink(rgb_dir)
            os.symlink(os.path.abspath(src), rgb_dir)
            break
    else:
        if os.path.lexists(rgb_dir): os.unlink(rgb_dir)
        os.symlink(os.path.abspath(images_dir), rgb_dir)

    # Link poses
    poses_dir = os.path.join(source_dir, "poses")
    poses_src = os.path.join(images_dir, "poses")
    if os.path.isdir(poses_src):
        if os.path.lexists(poses_dir): os.unlink(poses_dir)
        os.symlink(os.path.abspath(poses_src), poses_dir)

    # Link point
    point_dir = os.path.join(source_dir, "point")
    point_src = os.path.join(images_dir, "point")
    if os.path.isdir(point_src):
        if os.path.lexists(point_dir): os.unlink(point_dir)
        os.symlink(os.path.abspath(point_src), point_dir)

    # Intrinsics yaml
    yaml_path = os.path.join(source_dir, "intrinsics.yaml")
    js_path = os.path.join(images_dir, "intrinsics.json")
    if os.path.exists(js_path):
        import json
        with open(js_path) as f:
            d = json.load(f)
        with open(yaml_path, "w") as f:
            f.write("camera_params:\n")
            for k in ["fx","fy","cx","cy"]: f.write(f"  {k}: {d.get(k,0)}\n")
            f.write(f"  image_width: {d.get('w',d.get('width',1280))}\n")
            f.write(f"  image_height: {d.get('h',d.get('height',720))}\n")

    source_dir = os.path.join(output_dir, "fastgs_dataset")
    colmap_dir = os.path.join(output_dir, "3DGS_colmap_tmp")

    convert_script = os.path.join(FASTGS_DIR, "convert_custom_to_colmap.py")

    cmd = [sys.executable, convert_script, "-s", source_dir, "-o", colmap_dir, "--voxel_size", str(voxel_size)]
    print(f"\n{'='*60}")
    print(f"Step 1: Custom conversion (lingbot poses)")
    print(f"  Source:  {source_dir}")
    print(f"  Voxel:   {voxel_size}m")
    print(f"{'='*60}")

    r = subprocess.run(cmd, capture_output=False, text=True)
    if r.returncode != 0:
        print(f"\nError: Custom conversion failed (exit code {r.returncode})")
        sys.exit(1)

    return colmap_dir


def train(colmap_dir, output_dir, params):
    """Run FastGS training."""
    sys.path.insert(0, FASTGS_DIR)

    print(f"\n{'='*60}")
    print(f"Step 2: FastGS training")
    print(f"  Iterations: {params['iterations']}  Voxel: {params['voxel_size']}m")
    print(f"  Densify: every {params['densification_interval']} iters")
    print(f"{'='*60}\n")

    from arguments import ModelParams, OptimizationParams, PipelineParams
    from train import training
    from utils.general_utils import safe_state

    args_list = [
        "-s", colmap_dir, "-m", output_dir, "-i", "images", "--eval",
        "--test_iterations", str(params["iterations"]),
        "--save_iterations", str(params["iterations"]),
        "--checkpoint_iterations", str(params["iterations"]),
        "--densification_interval", str(params["densification_interval"]),
        "--optimizer_type", params["optimizer_type"],
        "--grad_abs_thresh", str(params["grad_abs_thresh"]),
        "--mult", str(params["mult"]),
        "--dense", str(params["dense"]),
        "--loss_thresh", str(params["loss_thresh"]),
    ]

    from argparse import ArgumentParser
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--websockets", action='store_true', default=False)

    args = parser.parse_args(args_list)
    args.save_iterations.append(args.iterations)
    safe_state(args.quiet)

    import torch
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from, args.websockets)


def main():
    args = parse_args()
    params = get_params(args)

    # Check for existing output
    final_ply = os.path.join(args.output, "point_cloud", f"iteration_{params['iterations']}", "point_cloud.ply")
    if os.path.exists(final_ply) and not args.force:
        print(f"Output already exists: {final_ply}\nUse --force to overwrite.")
        return

    if args.mode == "colmap":
        intrinsics_path = find_intrinsics(args.images)
        colmap_dir = run_colmap_sfm(args.images, args.output, intrinsics_path)
    else:
        colmap_dir = run_custom_conversion(args.images, args.output, params["voxel_size"])

    train(colmap_dir, args.output, params)

    if args.cleanup and args.mode == "custom":
        shutil.rmtree(colmap_dir, ignore_errors=True)

    if args.ply:
        os.makedirs(os.path.dirname(os.path.abspath(args.ply)), exist_ok=True)
        shutil.copy2(final_ply, args.ply)
        print(f"Copied PLY to: {args.ply}")

    print(f"\n{'='*60}")
    print(f"FastGS training complete!")
    print(f"  Mode: {args.mode}  Iterations: {params['iterations']}")
    print(f"  Output: {args.output or final_ply}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
