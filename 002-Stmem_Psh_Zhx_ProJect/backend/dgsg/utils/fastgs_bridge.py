"""
FastGS bridge: converts lingbot output to COLMAP format and runs FastGS training.

Usage (from batch_service.py or CLI):
    from dgsg.utils.fastgs_bridge import run_fastgs
    result = run_fastgs(data_dir, output_dir, fastgs_dir, **params)

Returns dict with {"ply_path": ..., "success": True/False, "error": ...}
"""
import os
import sys
import json
import yaml
import shutil
import subprocess
import tempfile
from pathlib import Path


def _convert_intrinsics(data_dir):
    """Convert intrinsics.json to intrinsics.yaml"""
    json_path = os.path.join(data_dir, "intrinsics.json")
    yaml_path = os.path.join(data_dir, "intrinsics.yaml")

    if os.path.exists(yaml_path):
        return  # already converted

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"intrinsics.json not found at {json_path}")

    with open(json_path) as f:
        d = json.load(f)

    intrinsics = {
        "camera_params": {
            "fx": d["fx"],
            "fy": d["fy"],
            "cx": d["cx"],
            "cy": d["cy"],
            "image_width": d["w"],
            "image_height": d["h"],
        }
    }
    with open(yaml_path, "w") as f:
        yaml.dump(intrinsics, f, default_flow_style=False)


def _bridge_data(data_dir):
    """Bridge lingbot output format to FastGS expected format."""
    # 1. Convert intrinsics
    _convert_intrinsics(data_dir)

    # 2. Symlink rgb/ -> raw/ (original uploaded frames, not lingbot-processed)
    # fastgs_raw/ contains raw camera frames saved during upload
    rgb_dir = os.path.join(data_dir, "rgb")
    raw_dir = os.path.join(data_dir, "fastgs_raw")
    # Prefer fastgs_raw (dedicated for FastGS), fallback to frames
    if not os.path.exists(rgb_dir):
        if os.path.exists(raw_dir):
            os.symlink(raw_dir, rgb_dir)
        elif os.path.exists(frames_dir):
            os.symlink(frames_dir, rgb_dir)


def _run_colmap_conversion(data_dir, colmap_dir, voxel_size, fastgs_dir):
    """Run convert_custom_to_colmap.py directly in-process"""
    if fastgs_dir not in sys.path:
        sys.path.insert(0, fastgs_dir)
    import convert_custom_to_colmap
    print(f"[FastGS] COLMAP conversion: {data_dir} -> {colmap_dir}")
    sys.argv = ["convert", "-s", data_dir, "-o", colmap_dir, "--voxel_size", str(voxel_size)]
    convert_custom_to_colmap.main()
    return colmap_dir
def _run_fastgs_training(colmap_dir, output_dir, params, fastgs_dir):
    """Run FastGS training by calling train.py directly."""
    import fastgs
    sys.path.insert(0, fastgs_dir)

    # Add FastGS submodules (CUDA extensions) to path
    _subs = os.path.join(fastgs_dir, "submodules")
    for _sub in ["diff-gaussian-rasterization_fastgs", "simple-knn", "fused-ssim"]:
        _p = os.path.join(_subs, _sub)
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)
        # Also add build/ directory for compiled .so files
        _build = os.path.join(_p, "build")
        if os.path.isdir(_build):
            for _b in os.listdir(_build):
                _bp = os.path.join(_build, _b)
                if os.path.isdir(_bp) and _bp not in sys.path:
                    sys.path.insert(0, _bp)

    # [DEBUG-f1] Print sys.path to find why imports fail
    import traceback as _tb
    print('[DEBUG-f1] _run_fastgs_training sys.path:', file=sys.stderr)
    for _i, _p in enumerate(sys.path):
        _check = os.path.join(_p, 'utils', 'loss_utils.py')
        _found = ' ✓' if os.path.exists(_check) else ''
        print(f'  [{_i}] {_p}{_found}', file=sys.stderr)
    print(f'[DEBUG-f1] fastgs_dir={fastgs_dir}', file=sys.stderr)
    try:
        from arguments import ModelParams, OptimizationParams, PipelineParams
        print('[DEBUG-f1] arguments import OK', file=sys.stderr)
    except Exception as _e:
        print(f'[DEBUG-f1] arguments import FAILED: {_e}', file=sys.stderr)
        _tb.print_exc()
    try:
        from train import training
        print('[DEBUG-f1] train.training import OK', file=sys.stderr)
    except Exception as _e:
        print(f'[DEBUG-f1] train.training import FAILED: {_e}', file=sys.stderr)
        _tb.print_exc()
    from utils.general_utils import safe_state

    args_list = [
        "-s", colmap_dir,
        "--iterations", str(params["iterations"]),
        "-m", output_dir,
        "-i", "images",
        "--eval",
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

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.websockets,
    )


def run_fastgs(data_dir, output_dir, fastgs_dir=None, **kwargs):
    """
    Main entry point. Bridges lingbot data to FastGS format and runs training.

    Args:
        data_dir: lingbot output directory (with frames/, poses/, depth/, point/, intrinsics.json)
        output_dir: where to save FastGS output (PLY file)
        fastgs_dir: path to FastGS source code (default: ../backend/fastgs)
        **kwargs: passed to FastGS training (iterations, voxel_size, etc.)

    Returns:
        {"ply_path": str, "success": bool, "error": str or None}
    """
    if fastgs_dir is None:
        fastgs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "fastgs"
        )

    params = {
        "iterations": 30000,
        "voxel_size": 0.05,
        "densification_interval": 500,
        "grad_abs_thresh": 0.0012,
        "optimizer_type": "default",
        "mult": 0.5,
        "dense": 0.001,
        "loss_thresh": 0.1,
    }
    params.update(kwargs)

    # Clear cached 'utils' module (may be imported from dgsg/utils/ first)
    for _mod in list(sys.modules.keys()):
        if _mod.startswith('utils'):
            del sys.modules[_mod]

    # Ensure FastGS is on sys.path before any imports
    if fastgs_dir not in sys.path:
        sys.path.insert(0, fastgs_dir)
    # Add submodules for CUDA extensions
    _subs = os.path.join(fastgs_dir, "submodules")
    for _sub in ["diff-gaussian-rasterization_fastgs", "simple-knn", "fused-ssim"]:
        _p = os.path.join(_subs, _sub)
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)

    try:
        # Step 0: Bridge data format
        _bridge_data(data_dir)

        # Step 1: COLMAP conversion
        colmap_dir = os.path.join(output_dir, "colmap_tmp")
        os.makedirs(colmap_dir, exist_ok=True)
        _run_colmap_conversion(data_dir, colmap_dir, params["voxel_size"], fastgs_dir)

        # Step 2: FastGS training
        _run_fastgs_training(colmap_dir, output_dir, params, fastgs_dir)

        # Find output PLY
        ply_path = os.path.join(
            output_dir, "point_cloud",
            f"iteration_{params['iterations']}", "point_cloud.ply"
        )
        if not os.path.exists(ply_path):
            # Try to find it in alternatives
            for root, dirs, files in os.walk(output_dir):
                for f in files:
                    if f.endswith(".ply"):
                        ply_path = os.path.join(root, f)
                        break

        # Cleanup
        shutil.rmtree(colmap_dir, ignore_errors=True)

        print(f"[FastGS] Done! PLY: {ply_path}")
        _egs_path = ply_path.replace(".ply", "_egs.ply")
        try:
            with open(ply_path, "rb") as _f:
                _hdr = []
                _vc = 0
                while True:
                    _l = _f.readline()
                    _hdr.append(_l)
                    if _l.startswith(b"element vertex"):
                        _vc = int(_l.split()[-1])
                    if _l == b"end_header\n":
                        break
                _body_start = _f.tell()
            _new_hdr = b"".join([l for l in _hdr if not l.startswith(b"property float n")]).decode()
            with open(ply_path, "rb") as _f:
                _f.seek(_body_start)
                _body = _f.read()
            _stride = 4 * (3 + 3 + 3 + 45 + 1 + 3 + 4)
            _out = bytearray()
            for _i in range(_vc):
                _s = _i * _stride
                _out.extend(_body[_s:_s+12])
                _out.extend(_body[_s+24:_s+_stride])
            with open(_egs_path, "wb") as _f:
                _f.write(_new_hdr.encode())
                _f.write(bytes(_out))
            print(f"[FastGS] EGS-compatible PLY: {_egs_path}")
        except Exception as _e:
            print(f"[FastGS] PLY conversion failed: {_e}")
        return {"ply_path": ply_path, "success": True, "error": None}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ply_path": None, "success": False, "error": str(e)}
