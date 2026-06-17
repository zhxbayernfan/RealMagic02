"""Export lingbot-map outputs for dgsg pipeline.

Generates per-frame world-space point clouds and camera poses, auto-calibrated
to real-world metric scale using sensor depth maps as absolute reference.

Usage:
    python export_for_dgsg.py \
        --data_dir data/test1_out_lingbot \
        --model_path checkpoints/lingbot-map-long.pt \
        --use_sdpa

    python export_for_dgsg.py \
        --data_dir data/test1_out_lingbot \
        --model_path checkpoints/lingbot-map-long.pt \
        --use_sdpa --scale_factor 4.2 --no_auto_calibrate

Inputs (under <data_dir>/):
    rgb/             — RGB images (.jpg/.png)
    depth/           — depth maps (.png), needed for auto-scale calibration
    intrinsics.yaml  — camera intrinsics

Outputs (under <data_dir>/):
    poses/           — per-frame 4x4 c2w pose txt files (real-world meters)
    point/           — per-frame [H, W, 3] float32 .npy (real-world meters)
"""

import argparse
import glob
import os
import sys
import time

if "--compile" not in sys.argv:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from lingbot_map.utils.geometry import (
    closed_form_inverse_se3_general,
    depth_to_world_coords_points,
)
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri


# =============================================================================
# Data loading
# =============================================================================

def load_images(image_folder, image_size=518, patch_size=14):
    exts = (".jpg", ".png", ".JPG", ".jpeg", ".JPEG", ".bmp", ".BMP")
    paths = sorted(
        p for p in glob.glob(os.path.join(image_folder, "*")) if p.endswith(exts)
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {image_folder}")
    images = load_and_preprocess_images(
        paths, mode="crop", image_size=image_size, patch_size=patch_size
    )
    filenames = [os.path.basename(p) for p in paths]
    return images, filenames


def load_intrinsics(yaml_path):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cam = cfg["camera_params"]
    return {
        "height": cam["image_height"],
        "width": cam["image_width"],
        "fx": cam["fx"],
        "fy": cam["fy"],
        "cx": cam["cx"],
        "cy": cam["cy"],
        "depth_scale": cam.get("png_depth_scale", 1000.0),
    }


# =============================================================================
# Model loading
# =============================================================================

def load_model(args, device):
    if getattr(args, "mode", "streaming") == "windowed":
        from lingbot_map.models.gct_stream_window import GCTStream
    else:
        from lingbot_map.models.gct_stream import GCTStream

    print("Building model...")
    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=args.enable_3d_rope,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
    )

    print(f"Loading checkpoint: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    print("  Checkpoint loaded.")
    return model.to(device).eval()


# =============================================================================
# Post-processing
# =============================================================================

def postprocess(predictions, images):
    """Convert pose_enc to c2w extrinsics, matching demo.py exactly.

    After this function:
        predictions["c2w"]       = camera-to-world 3x4 (inverted from w2c)
        predictions["intrinsic"] = intrinsic 3x3
    """
    extrinsic_w2c, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )

    # w2c → c2w (same as demo.py:278-287)
    w2c_4x4 = torch.zeros(
        (*extrinsic_w2c.shape[:-2], 4, 4),
        device=extrinsic_w2c.device, dtype=extrinsic_w2c.dtype,
    )
    w2c_4x4[..., :3, :4] = extrinsic_w2c
    w2c_4x4[..., 3, 3] = 1.0
    c2w_4x4 = closed_form_inverse_se3_general(w2c_4x4)
    c2w = c2w_4x4[..., :3, :4]

    predictions["c2w"] = c2w
    predictions["intrinsic"] = intrinsic

    print("Moving results to CPU...")
    for k in list(predictions.keys()):
        v = predictions[k]
        if isinstance(v, torch.Tensor):
            if v.ndim >= 2 and v.shape[0] == 1:
                v = v[0]
            predictions[k] = v.detach().cpu()
    return predictions


# =============================================================================
# Scale calibration — depth-ratio method
# =============================================================================

def calibrate_scale_depth_ratio(depth_lingbot, depth_files, intrinsics,
                                n_sample_frames=None):
    """Auto-calibrate scale_factor by comparing lingbot-map depth to sensor depth.

    Principle:
      sensor_depth(meters) = scale_factor × lingbot_depth(internal_units)

    For each frame, resize the sensor depth map to match lingbot-map's
    resolution, then compute the per-pixel ratio at valid pixels.

    This method uses ALL valid pixels across ALL frames — orders of magnitude
    more data points than pair-based methods — giving extremely robust results.

    Args:
        depth_lingbot: (S, H, W, 1) tensor — lingbot-map predicted depth (raw scale)
        depth_files: list of sensor depth .png paths
        intrinsics: dict with 'depth_scale' key
        n_sample_frames: number of frames to sample (None = all)

    Returns:
        estimated_scale_factor (float)
    """
    depth_scale = intrinsics["depth_scale"]
    n = min(depth_lingbot.shape[0], len(depth_files))

    if n_sample_frames is not None and n_sample_frames < n:
        indices = np.linspace(0, n - 1, n_sample_frames, dtype=int)
    else:
        indices = list(range(n))

    ratios_all = []

    print(f"\nAuto-calibrating scale_factor from depth ratio "
          f"({len(indices)} frames)...")

    for idx in tqdm(indices, desc="Calibrating scale"):
        # Load sensor depth (meters)
        d_sensor = cv2.imread(depth_files[idx], cv2.IMREAD_UNCHANGED)
        if d_sensor is None:
            continue
        d_sensor = d_sensor.astype(np.float64) / depth_scale

        # Lingbot-map depth (internal units)
        d_lb = depth_lingbot[idx].squeeze(-1).numpy()  # (H_lb, W_lb)

        # Resize sensor depth to lingbot-map resolution
        h_lb, w_lb = d_lb.shape
        d_sensor_resized = cv2.resize(
            d_sensor, (w_lb, h_lb), interpolation=cv2.INTER_NEAREST
        )

        # Valid pixels in both
        valid = (
            (d_lb > 0.01) & (d_lb < 50.0) &
            (d_sensor_resized > 0.01) & (d_sensor_resized < 50.0)
        )
        if valid.sum() < 1000:
            continue

        ratios = d_sensor_resized[valid] / d_lb[valid]

        # Per-frame median (robust to outliers)
        frame_median = np.median(ratios)
        # Also collect for cross-frame consensus
        ratios_all.append(frame_median)

    if not ratios_all:
        print("WARNING: scale calibration failed, using default 4.2")
        return 4.2

    ratios_arr = np.array(ratios_all)

    # Remove outliers across frames (MAD filter)
    med = np.median(ratios_arr)
    mad = np.median(np.abs(ratios_arr - med))
    inlier = np.abs(ratios_arr - med) < 3.0 * max(mad, 0.1)
    filtered = ratios_arr[inlier]

    if len(filtered) < 3:
        print(f"WARNING: too few inliers after MAD filter, using median={med:.3f}")
        return round(float(med), 3)

    scale = float(np.median(filtered))

    print(f"\nScale calibration results:")
    print(f"  Samples:        {len(ratios_arr)} frames")
    print(f"  Inliers:        {len(filtered)} frames ({100*len(filtered)/len(ratios_arr):.0f}%)")
    print(f"  Median (all):   {np.median(ratios_arr):.4f}")
    print(f"  Median (clean): {scale:.4f}")
    print(f"  Mean (clean):   {np.mean(filtered):.4f}")
    print(f"  Std (clean):    {np.std(filtered):.4f}")
    print(f"  Range (clean):  {filtered.min():.3f} ~ {filtered.max():.3f}")

    return round(scale, 3)


# =============================================================================
# Confidence filtering (global percentile — same logic as demo.py)
# =============================================================================

def filter_by_confidence(world_points, conf, percentile=50.0):
    """Global confidence filtering — mirrors demo.py's viser_wrapper.

    All pixels across all frames are ranked together and the global
    `percentile`-th value becomes the threshold.  Pixels below it are
    zeroed out.  High-quality frames naturally keep more points.

    Args:
        world_points: (S, H, W, 3) tensor (already scaled to meters)
        conf: (S, H, W) tensor
        percentile: keep points above this global percentile (0 = no filtering)

    Returns:
        filtered world_points as numpy (S, H, W, 3) with low-conf points → 0
    """
    if percentile <= 0:
        return world_points.numpy().astype(np.float32)

    wp_np = world_points.numpy()
    cf_np = conf.numpy()

    # Flatten all frames together — global ranking
    cf_flat = cf_np.reshape(-1)
    valid = cf_flat > 0.1
    if valid.sum() == 0:
        print("WARNING: no valid confidence values, skipping filtering")
        return wp_np.astype(np.float32)

    threshold = np.percentile(cf_flat[valid], percentile)
    mask = cf_np >= threshold

    kept = mask.sum()
    total = mask.size

    wp_np = wp_np.copy()
    wp_np[~mask] = 0.0

    print(f"Confidence filter (global {percentile}th percentile):")
    print(f"  Threshold:  {threshold:.4f}")
    print(f"  Kept:       {kept:,} / {total:,} ({100*kept/total:.1f}%)")
    print(f"  Discarded:  {total - kept:,} ({100*(total-kept)/total:.1f}%)")

    return wp_np.astype(np.float32)


# =============================================================================
# Saving
# =============================================================================

def save_poses(c2w_poses, filenames, output_dir):
    """Save per-frame 4x4 camera-to-world (c2w) pose matrices as text files."""
    os.makedirs(output_dir, exist_ok=True)
    for i, fname in enumerate(filenames):
        stem = os.path.splitext(fname)[0]
        c2w_3x4 = c2w_poses[i]
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :] = c2w_3x4.numpy().astype(np.float64)
        path = os.path.join(output_dir, f"{stem}.txt")
        np.savetxt(path, mat, fmt="%.18e")
    print(f"Saved {len(filenames)} poses to {output_dir}/")


def save_points(world_points, filenames, output_dir):
    """Save per-frame world_points as float32 .npy files.

    Args:
        world_points: (S, H, W, 3) numpy array (already filtered)
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, fname in enumerate(filenames):
        stem = os.path.splitext(fname)[0]
        np.save(os.path.join(output_dir, f"{stem}.npy"), world_points[i])
    print(f"Saved {len(filenames)} point clouds to {output_dir}/")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export lingbot-map outputs for dgsg pipeline"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--mode", type=str, default="streaming",
                        choices=["streaming", "windowed"])
    parser.add_argument("--keyframe_interval", type=int, default=None,
                        help="Keyframe interval. None/0 = auto")
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument("--use_sdpa", action="store_true")
    parser.add_argument("--enable_3d_rope", action="store_true", default=True)
    parser.add_argument("--max_frame_num", type=int, default=10000)
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64,
                        help="KV cache sliding window size (default: 64)")
    parser.add_argument("--camera_num_iterations", type=int, default=4)
    parser.add_argument("--offload_to_cpu", action="store_true", default=False)
    parser.add_argument("--first_k", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--overlap_size", type=int, default=16)
    parser.add_argument("--overlap_keyframes", type=int, default=None)

    parser.add_argument("--scale_factor", type=float, default=None,
                        help="Manual scale factor")
    parser.add_argument("--no_auto_calibrate", action="store_true",
                        help="Skip auto-calibration")
    parser.add_argument("--conf_percentile", type=float, default=50.0,
                        help="Global confidence percentile for point filtering "
                             "(same as demo.py). 0 = keep all points. Default: 50")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_total = time.time()

    # ── Load dataset ──────────────────────────────────────────────────────
    rgb_dir = os.path.join(args.data_dir, "rgb")
    intrinsics_path = os.path.join(args.data_dir, "intrinsics.yaml")
    depth_dir = os.path.join(args.data_dir, "depth")

    intrinsics = load_intrinsics(intrinsics_path)
    images, filenames = load_images(rgb_dir, args.image_size, args.patch_size)
    num_frames = images.shape[0]
    print(f"Loaded {num_frames} images, shape {images.shape}")

    has_depth = os.path.isdir(depth_dir)
    depth_files = []
    if has_depth:
        depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
        print(f"Found {len(depth_files)} depth maps")

    # ── Load model ────────────────────────────────────────────────────────
    model = load_model(args, device)

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    print(f"Casting aggregator to {dtype} (heads kept in fp32)")
    model.aggregator = model.aggregator.to(dtype)

    images_dev = images.to(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Auto keyframe_interval
    if args.keyframe_interval is None or args.keyframe_interval <= 0:
        if args.mode == "streaming" and num_frames > 320:
            args.keyframe_interval = (num_frames + 319) // 320
            print(f"Auto keyframe_interval={args.keyframe_interval}")
        else:
            args.keyframe_interval = 1

    print(f"Input: {num_frames} frames, mode={args.mode}, "
          f"keyframe_interval={args.keyframe_interval}")

    # ── Run inference ─────────────────────────────────────────────────────
    t0 = time.time()
    output_device = torch.device("cpu") if args.offload_to_cpu else None

    print(f"Running {args.mode} inference (dtype={dtype})...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if args.mode == "windowed":
            predictions = model.inference_windowed(
                images_dev,
                keyframe_interval=args.keyframe_interval,
                window_size=args.window_size,
                overlap_size=args.overlap_size,
                overlap_keyframes=args.overlap_keyframes,
                num_scale_frames=args.num_scale_frames,
                output_device=output_device,
            )
        else:
            predictions = model.inference_streaming(
                images_dev,
                keyframe_interval=args.keyframe_interval,
                num_scale_frames=args.num_scale_frames,
                output_device=output_device,
            )
    print(f"Inference done in {time.time() - t0:.1f}s")

    # ── Post-process ──────────────────────────────────────────────────────
    if args.offload_to_cpu:
        del images_dev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        images_for_post = predictions["images"]
    else:
        images_for_post = images_dev

    predictions = postprocess(predictions, images_for_post)

    # ── Scale calibration ─────────────────────────────────────────────────
    if args.scale_factor is not None:
        scale_factor = args.scale_factor
        print(f"\nUsing manual scale_factor = {scale_factor}")
    elif args.no_auto_calibrate:
        scale_factor = 4.2
        print(f"\nAuto-calibration skipped, using default scale_factor = {scale_factor}")
    elif not has_depth:
        scale_factor = 4.2
        print(f"\nNo depth/ dir found, using default scale_factor = {scale_factor}")
        print("(Place sensor depth .png files in <data_dir>/depth/ for auto-calibration)")
    else:
        scale_factor = calibrate_scale_depth_ratio(
            predictions["depth"], depth_files, intrinsics
        )

    # ── Apply scale ───────────────────────────────────────────────────────
    predictions["c2w"][:, :3, 3] *= scale_factor
    print(f"Applied scale_factor={scale_factor} to poses")

    # ── Generate world points via depth+pose backprojection ───────────────
    # Same logic as demo.py: pass c2w directly to depth_to_world_coords_points
    # (which is named as expecting w2c but demo.py passes c2w — the function
    #  internally does inv(c2w)=w2c and uses w2c as the c2w transform).
    depth = predictions["depth"].numpy()       # (S, H, W, 1)
    c2w = predictions["c2w"].numpy()           # (S, 3, 4) — camera-to-world
    intrinsic = predictions["intrinsic"].numpy()  # (S, 3, 3)
    depth_conf = predictions.get("depth_conf")    # (S, H, W)

    depth *= scale_factor
    print(f"Applied scale_factor={scale_factor} to depth maps")

    S = depth.shape[0]
    print(f"\nBackprojecting depth+c2w to world points ({S} frames)...")
    world_points_list = []
    for i in tqdm(range(S), desc="Backprojecting"):
        pts_world, _, _ = depth_to_world_coords_points(
            depth[i].squeeze(-1), c2w[i], intrinsic[i]
        )
        world_points_list.append(pts_world)
    world_points = np.stack(world_points_list, axis=0).astype(np.float32)
    # Free memory
    del world_points_list

    print(f"world_points (backprojected): [{S}, {world_points.shape[1]}, "
          f"{world_points.shape[2]}, 3]")

    # ── Confidence filtering (uses depth_conf) ────────────────────────────
    if depth_conf is not None and args.conf_percentile > 0:
        # Convert to tensor-like interface for filter_by_confidence
        world_points = filter_by_confidence(
            torch.from_numpy(world_points),
            depth_conf,
            percentile=args.conf_percentile,
        )
    else:
        print("Confidence filtering: skipped")

    # ── Save ──────────────────────────────────────────────────────────────
    poses_dir = os.path.join(args.data_dir, "poses")
    point_dir = os.path.join(args.data_dir, "point")

    # Save c2w poses (dgsg MydataDataset.load_poses expects c2w 4x4 txt)
    save_poses(predictions["c2w"], filenames, poses_dir)
    save_points(world_points, filenames, point_dir)

    # ── Summary ───────────────────────────────────────────────────────────
    translations = predictions["c2w"][:, :3, 3].numpy()
    diffs = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    total_len = np.sum(diffs)
    total_pts = world_points.shape[0] * world_points.shape[1] * world_points.shape[2]
    nonzero = (world_points.sum(axis=-1) != 0).sum()
    print(f"\n{'='*60}")
    print(f"Export summary:")
    print(f"  Frames:            {len(filenames)}")
    print(f"  Scale factor:      {scale_factor}")
    print(f"  Conf percentile:   {args.conf_percentile:.0f}% "
          f"({'global' if args.conf_percentile > 0 else 'off'})")
    print(f"  Active points:     ~{nonzero:,} / {total_pts:,} "
          f"({100*nonzero/total_pts:.1f}%)")
    print(f"  Trajectory length: {total_len:.2f}m")
    print(f"  Point resolution:  {world_points.shape[1]}x{world_points.shape[2]}")
    print(f"  Point format:      float32 (HxWx3, world coords, meters; 0=filtered)")
    print(f"  Output dirs:")
    print(f"    {poses_dir}/  ({len(filenames)} txt)")
    print(f"    {point_dir}/    ({len(filenames)} npy)")
    print(f"  Total time:         {time.time() - t_total:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
