"""LingBot-MAP 流式批次管理服务（单用户架构）

参考 live_camera.py 和 stream.py 的设计：
- 模型常驻内存，不重复加载
- 维护KV缓存，逐帧推理
- 后台线程监控帧文件夹，自动处理新帧
- 点云数据缓存在内存中，API直接返回

文件结构：
data/
└── {batch_id}/
    ├── frames/
    │   ├── frame_000.jpg
    │   ├── frame_001.jpg
    │   └── ...
    ├── status.json
    └── logs.json
"""

import os
import sys
import json
import time
import glob
import asyncio
import gzip
import traceback
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from collections import deque

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms as TF
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_detector import MotionDetector

# 禁止 ultralytics 自动下载模型（GitHub 不可达）
os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "0")

# ── DGSG 流式管线 ──
from importlib.machinery import SourceFileLoader
_DGSG_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dgsg", "scripts")
if _DGSG_SCRIPTS not in sys.path:
    sys.path.insert(0, _DGSG_SCRIPTS)
_DGSG_BASE = os.path.dirname(_DGSG_SCRIPTS)
if _DGSG_BASE not in sys.path:
    sys.path.insert(0, _DGSG_BASE)
from dgsg_stream import StreamingDGSG

# 必须在导入torch之前设置
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general, unproject_depth_map_to_point_map
from lingbot_map.utils.load_fn import load_and_preprocess_images
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import base64 as _base64_module
def _b64(data: bytes) -> str:
    return _base64_module.b64encode(data).decode('ascii')

app = FastAPI(title="LingBot-MAP Streaming Service", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Inference-Time"],
)

# 配置
# 项目根目录 = stmem/ (batch_service.py 在 backend/ 下)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = Path(os.environ.get("STMEM_DATA_DIR", os.path.join(_PROJECT_ROOT, "data")))
MAX_FRAMES_PER_BATCH = 50000
SUPPORTED_FORMATS = {"image/jpeg", "image/png", "image/webp"}
# MODEL_PATH 指向 lingbot-map 模型权重，根据实际部署路径设置
MODEL_PATH = os.environ.get("STMEM_MODEL_PATH", os.path.join(_PROJECT_ROOT, "checkpoints", "lingbot-map-long.pt"))

# 推理参数
IMAGE_SIZE = 518
PATCH_SIZE = 14
NUM_SCALE_FRAMES = 8
# Keyframe interval: auto-selected based on frame count (same as live_camera.py).
# <= 320 frames: interval=1 (every frame is a keyframe, best accuracy).
# > 320 frames: interval=ceil(N/320) to keep KV cache at ~320 keyframes.
# The frontend can override this via the keyframe_interval parameter.
KEYFRAME_INTERVAL_DEFAULT = 1
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

# 全局状态
model_state = {
    "model": None,
    "device": None,
    "initialized": False,
    "current_batch_id": None,
    "frame_idx": 0,
    "scale_frames": NUM_SCALE_FRAMES,
    "keyframe_interval": KEYFRAME_INTERVAL_DEFAULT,
    "max_images": None,
    "known_paths": set(),
    "all_predictions": {
        "pose_enc": [],
        "depth": [],
        "depth_conf": [],
        "world_points": [],
        "world_points_conf": [],
        "images": [],
    },
    "is_streaming": False,
    "stop_event": threading.Event(),
    "finish_requested": False,
    "motion_detector": None,
}

# ── DGSG 流式管线状态 ──
dgsg_pipeline = None            # StreamingDGSG 实例
frame_data_queue = queue.Queue()  # 无界帧数据队列，保证所有帧都被 DGSG 消费
dgsg_cache = {}                 # frame_id → DGSG snapshot dict
dgsg_cache_lock = threading.Lock()
dgsg_running = threading.Event()
dgsg_finished = threading.Event()
dgsg_finish_called = False      # prevent double-calling pipeline.finish()
dgsg_thread = None

# 点云过滤参数（与 live_camera.py 一致）
CONF_THRESHOLD = 0.7        # 置信度阈值
DOWNSAMPLE_FACTOR = 10     # 下采样倍数
MAX_CACHE_FRAMES = 300     # 内存缓存最大帧数

# 诊断计数器（每次 start_inference 重置）
_diag_stats = {
    "uploaded": 0,
    "motion_skipped": 0,
    "motion_kept": 0,
    "inferred": 0,
    "preload_error": 0,
    "preload_idx_none": 0,
}

# 点云帧缓存
frame_cache = {}  # {frame_index: {"points": [...], "colors": [...], "confs": [...], "camera": {...}}}
cache_frame_order = []  # 维护缓存顺序，用于 FIFO 淘汰

# 帧到达时间记录（用于计算推理耗时）
frame_arrival_times = {}
# 帧推理耗时拆分：forward 开始/结束时间
frame_infer_start = {}
frame_forward_end = {}

# 异步解码：GPU 跑下一帧 forward 的同时，后台线程处理当前帧回调
_callback_queue = queue.Queue()
_cache_lock = threading.Lock()

# 预加载：GPU forward 期间后台读下一帧图片，forward 结束直接可用
_preloader = ThreadPoolExecutor(max_workers=1)

def _preload_frame(path):
    """后台预加载一张帧：motion filter + 读图 resize + tensor 上 GPU"""
    skip = False
    try:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            detector = model_state.get("motion_detector")
            if detector is not None:
                should_save, layer = detector.should_save(gray)
                if not should_save:
                    os.remove(path)
                    skip = True
                    _diag_stats["motion_skipped"] += 1
                    _diag_stats[f"l{layer}_skipped"] = _diag_stats.get(f"l{layer}_skipped", 0) + 1
                    write_log(f"[MOTION] 跳过帧 {os.path.basename(path)} (L{layer})", "info")
                else:
                    _diag_stats["motion_kept"] += 1
                    _diag_stats[f"l{layer}_kept"] = _diag_stats.get(f"l{layer}_kept", 0) + 1
                    write_log(f"[MOTION] 保留帧 {os.path.basename(path)} (L{layer})", "info")
            else:
                write_log(f"[MOTION] detector 未初始化，保留帧 {os.path.basename(path)}", "info")
    except Exception:
        pass
    if skip:
        return (path, True, None, None, None)
    try:
        bidx = int(os.path.splitext(os.path.basename(path))[0].split('_')[1])
        img = load_single_image(path)
        frame_image = img.unsqueeze(0).unsqueeze(0).to(model_state["device"], non_blocking=True)
        return (path, False, bidx, img, frame_image)
    except Exception:
        _diag_stats["preload_error"] += 1
        return (path, False, None, None, None)

def _callback_worker():
    """后台线程：负责 image resize + tensor→CPU + on_frame_callback，与 GPU forward 并行"""
    while True:
        item = _callback_queue.get()
        if item is None:
            break
        frame_idx, img, frame_output, infer_start, forward_end = item
        try:
            # 获取模型输出尺寸
            depth = frame_output.get("depth")
            if depth is not None:
                d = depth
                while d.ndim > 2:
                    d = d[0] if d.shape[0] == 1 else d.squeeze()
                model_h, model_w = d.shape
            else:
                model_h, model_w = IMAGE_SIZE, IMAGE_SIZE

            # 缩放颜色图到模型输出尺寸
            img_resized = img.permute(1, 2, 0).numpy()
            img_pil = Image.fromarray((img_resized * 255).clip(0, 255).astype(np.uint8))
            img_pil = img_pil.resize((model_w, model_h), Image.Resampling.BICUBIC)
            img_np = np.array(img_pil)
            del img

            # 全部 tensor 移交 CPU + 保存预测结果（保留最近 100 帧，避免 OOM）
            cb_output = {}
            for key in model_state["all_predictions"]:
                if key in frame_output:
                    v = frame_output[key].to("cpu")
                    model_state["all_predictions"][key].append(v)
                    cb_output[key] = v
            for k, v in frame_output.items():
                if k not in cb_output:
                    cb_output[k] = v.to("cpu") if isinstance(v, torch.Tensor) else v
            del frame_output

            # 每50帧截断 all_predictions，只保留最近100帧
            if frame_idx > 0 and frame_idx % 50 == 0:
                for key in model_state["all_predictions"]:
                    lst = model_state["all_predictions"][key]
                    if len(lst) > 100:
                        model_state["all_predictions"][key] = lst[-100:]

            # [DEBUG-a7c3] 每10帧打印一次 CPU 内存累计
            if frame_idx % 10 == 0:
                total_mb = 0
                for key, lst in model_state["all_predictions"].items():
                    for t in lst:
                        if hasattr(t, 'numel'):
                            total_mb += t.numel() * t.element_size() / 1024 / 1024
                import resource
                rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # macOS KB, Linux KB
                write_log(
                    f"[DEBUG-a7c3] 内存 frame={frame_idx}: all_predictions={total_mb:.0f}MB "
                    f"RSS={rss_mb:.0f}MB GPU={torch.cuda.memory_allocated()/1024**2:.0f}MB"
                    if torch.cuda.is_available() else f"[DEBUG-a7c3] 内存 frame={frame_idx}: all_predictions={total_mb:.0f}MB",
                    "info"
                )

            on_frame_callback(frame_idx, img_np, cb_output, infer_start, forward_end)
        except Exception:
            traceback.print_exc()

def add_to_frame_cache(frame_idx, points, colors, confs, camera):
    """添加点云到缓存，超过 MAX_CACHE_FRAMES 时丢弃最旧帧"""
    global frame_cache, cache_frame_order
    
    frame_cache[frame_idx] = {
        "points": np.asarray(points, dtype=np.float32),
        "colors": np.asarray(colors, dtype=np.float32),
        "confs": np.asarray(confs, dtype=np.float32),
        "camera": camera,
        "inference_time": time.time(),
    }
    cache_frame_order.append(frame_idx)
    
    # FIFO 淘汰，保持不超过 MAX_CACHE_FRAMES
    while len(cache_frame_order) > MAX_CACHE_FRAMES:
        oldest = cache_frame_order.pop(0)
        del frame_cache[oldest]

def _dgsg_consumer():
    """DGSG 消费线程：从 frame_data_queue 取帧 → process_frame → 写入 dgsg_cache"""
    write_log("[DGSG] 消费线程启动", "info")
    while dgsg_running.is_set():
        try:
            item = frame_data_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if item is None:  # Sentinel: 所有帧已入队
            break

        frame_idx, rgb_np, depth_metric, world_points_np, c2w_np = item

        try:
            t0 = time.time()
            snapshot = dgsg_pipeline.process_frame(
                rgb_image=rgb_np,
                depth_image=depth_metric,
                world_points=world_points_np,
                pose=c2w_np,
                frame_id=frame_idx,
            )
            dt = time.time() - t0

            # 缓存前验证颜色数据完整性
            pc = snapshot.get('point_cloud', {})
            colors = pc.get('rgb_colors')
            if colors is not None and len(colors) > 0:
                c_min, c_max = float(colors.min()), float(colors.max())
                if not np.isfinite(colors).all() or c_min < -0.1 or c_max > 1.1:
                    write_log(
                        f"[DGSG] 帧#{frame_idx} 颜色异常: range=[{c_min:.4f}, {c_max:.4f}], "
                        f"finite={bool(np.isfinite(colors).all())}, dtype={colors.dtype}, "
                        f"flags.owndata={colors.flags['OWNDATA']}, contiguous={colors.flags['C_CONTIGUOUS']}",
                        "err"
                    )

            with dgsg_cache_lock:
                # 只保留最新 1 帧的 snapshot（每个 snapshot 是累积全局点云的完整副本，
                # frame=280 时单个 snapshot ≈ 3GB，保留多个会导致 OOM）
                if dgsg_cache and frame_idx < max(dgsg_cache.keys()):
                    dgsg_cache.clear()
                dgsg_cache.clear()  # 只保留最新一帧
                dgsg_cache[frame_idx] = snapshot

            sg = snapshot.get('scene_graph', {})
            stats = snapshot.get('stats', {})
            ch = snapshot.get('changes', {})
            write_log(
                f"[DGSG] 帧#{frame_idx} objs={stats.get('total_objects', 0)} pts={stats.get('total_points', 0)} "
                f"nodes={sg.get('stats', {}).get('total_nodes', 0)} "
                f"+{len(ch.get('added_obj_ids', []))}/-{len(ch.get('removed_obj_ids', []))} "
                f"({dt:.2f}s)",
                "info"
            )
        except Exception as e:
            write_log(f"[DGSG] 帧#{frame_idx} 处理异常: {traceback.format_exc()}", "err")

        # 每10帧释放一次GPU碎片
        if frame_idx > 0 and frame_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_log("[DGSG] 消费线程结束", "info")
    dgsg_finished.set()

# 批次状态
batch_status = {
    "batch_id": None,
    "status": "idle",
    "total_frames": 0,
    "uploaded_frames": 0,
    "processed_frames": 0,
    "total_points": 0,
    "error_message": "",
    "dgsg_status": "idle",  # idle → building → done → error
    "scale_status": "idle",  # idle → calibrating → done → error
    "scale_factor": None,
    "scale_confidence": None,
}

# 日志
batch_logs = []

def write_log(message, log_type="info"):
    """写入日志"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "message": message,
        "type": log_type
    }
    batch_logs.append(log_entry)
    
    # 保持日志数量在合理范围
    if len(batch_logs) > 1000:
        batch_logs[:] = batch_logs[-500:]
    
    print(f"[{timestamp}] [{log_type}] {message}")

def update_status(**kwargs):
    """更新批次状态"""
    batch_status.update(kwargs)
    
    # 保存到文件
    if batch_status.get("batch_id"):
        status_file = DATA_DIR / batch_status["batch_id"] / "status.json"
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, 'w') as f:
            json.dump(batch_status, f, indent=2)

def load_single_image(image_path, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
    """加载并预处理单张图片"""
    img = Image.open(image_path)
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = img.convert("RGB")

    width, height = img.size
    new_width = image_size
    new_height = round(height * (new_width / width) / patch_size) * patch_size

    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    img = TF.ToTensor()(img)

    if new_height > image_size:
        start_y = (new_height - image_size) // 2
        img = img[:, start_y: start_y + image_size, :]

    return img

def load_model():
    """加载模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model to {device}...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU mem before load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    model = GCTStream(
        img_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        enable_3d_rope=True,
        enable_point=False,  # Depth-unprojected points (consistent with stream.py/live_camera.py)
        max_frame_num=1024,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=NUM_SCALE_FRAMES,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=os.environ.get("STMEM_USE_SDPA", "0") == "1",  # Jetson/ARM64: set STMEM_USE_SDPA=1
        camera_num_iterations=4,
    )

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    if isinstance(state_dict, list):
        state_dict = state_dict[0] if state_dict else {}
    elif hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU mem after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    write_log("模型加载成功", "ok")
    return model, device

def on_frame_callback(frame_idx, image_np, frame_output, infer_start=None, forward_end=None):
    """逐帧回调函数：处理推理结果并缓存点云原始数据。

    由后台线程调用，与 GPU forward 并行。后处理仅做数据维度压缩和格式化。
    """
    try:
        H, W = image_np.shape[:2]

        pose_enc = frame_output["pose_enc"]
        depth = frame_output.get("depth")
        depth_conf = frame_output.get("depth_conf")
        world_points = frame_output.get("world_points")

        if pose_enc is None or depth is None or depth_conf is None:
            return

        if pose_enc.dim() == 3:
            _pose_enc = pose_enc[0, 0]
        elif pose_enc.dim() == 2:
            _pose_enc = pose_enc[0]
        else:
            _pose_enc = pose_enc

        if depth.dim() == 5:
            _depth = depth[0, 0]
            _depth_conf = depth_conf[0, 0]
        elif depth.dim() == 4:
            _depth = depth[0]
            _depth_conf = depth_conf[0]
        else:
            _depth = depth
            _depth_conf = depth_conf

        _pose_enc_batch = _pose_enc.unsqueeze(0).unsqueeze(0)
        extrinsic_c2w, intrinsic = pose_encoding_to_extri_intri(
            _pose_enc_batch, image_size_hw=(H, W)
        )
        extrinsic_c2w = extrinsic_c2w[0, 0]
        intrinsic = intrinsic[0, 0]

        intrinsic_np = intrinsic.cpu().numpy()
        _depth_np = _depth.cpu().numpy()
        _depth_conf_np = _depth_conf.cpu().numpy()

        if world_points is not None:
            if world_points.dim() == 5:
                _wp = world_points[0, 0].cpu().numpy()
            elif world_points.dim() == 4:
                _wp = world_points[0].cpu().numpy()
            else:
                _wp = world_points.cpu().numpy()
        else:
            c2w_4x4 = torch.eye(4, device=extrinsic_c2w.device, dtype=extrinsic_c2w.dtype)
            c2w_4x4[:3, :4] = extrinsic_c2w
            w2c = closed_form_inverse_se3_general(c2w_4x4.unsqueeze(0)).squeeze(0)
            w2c_np = w2c[:3, :4].cpu().numpy()
            wp_t = unproject_depth_map_to_point_map(
                _depth_np[None],
                w2c_np[np.newaxis],
                intrinsic_np[np.newaxis],
            )
            _wp = wp_t[0]

        extrinsic_c2w_np = extrinsic_c2w.cpu().numpy()

        pred_pts = _wp.reshape(-1, 3)
        color_flat = image_np.reshape(-1, 3) / 255.0
        conf_flat = _depth_conf_np.reshape(-1)

        extr_np = extrinsic_c2w_np
        intr_np = intrinsic_np

        camera = {
            "focal": [float(intr_np[0, 0]), float(intr_np[1, 1])],
            "pp": [float(intr_np[0, 2]), float(intr_np[1, 2])],
            "R_c2w": extr_np[:3, :3].tolist(),
            "t_c2w": extr_np[:3, 3].tolist(),
            "image_w": W,
            "image_h": H,
        }

        # ── 保存 depth、pose、intrinsics 到硬盘（供 dgsg 管线使用）──
        batch_id = model_state.get("current_batch_id")
        if batch_id:
            batch_dir = DATA_DIR / batch_id

            depth_dir = batch_dir / "depth"
            depth_dir.mkdir(exist_ok=True)
            poses_dir = batch_dir / "poses"
            poses_dir.mkdir(exist_ok=True)

            depth_shape = _depth_np.shape
            depth_h, depth_w = depth_shape[0], depth_shape[1]
            if _depth_np.ndim > 2:
                _depth_np = _depth_np.reshape(depth_h, depth_w)

            # 从第一帧原始图片获取原始分辨率（depth 需要和 rgb 同尺寸）
            orig_h, orig_w = H, W  # 默认用模型输出尺寸
            frames_dir_check = batch_dir / "frames"
            if frames_dir_check.exists():
                any_frame = next(frames_dir_check.glob("*.jpg"), None)
                if any_frame is None:
                    any_frame = next(frames_dir_check.glob("*.png"), None)
                if any_frame is not None:
                    from PIL import Image as _Img
                    with _Img.open(any_frame) as _img:
                        orig_w, orig_h = _img.size  # PIL: (width, height)

            # 如果深度图和原始 RGB 尺寸不同，需要 resize + 缩放 intrinsics
            save_depth = _depth_np
            save_intr = intr_np
            if depth_h != orig_h or depth_w != orig_w:
                scale_x = orig_w / depth_w
                scale_y = orig_h / depth_h
                save_intr = intr_np.copy()
                save_intr[0, 0] *= scale_x
                save_intr[1, 1] *= scale_y
                save_intr[0, 2] *= scale_x
                save_intr[1, 2] *= scale_y
                d_img = Image.fromarray((_depth_np * 1000).clip(0, 65535).astype(np.uint16))
                d_img = d_img.resize((orig_w, orig_h), Image.NEAREST)
                save_depth = np.array(d_img).astype(np.float32) / 1000.0

            depth_mm = (save_depth * 1000).clip(0, 65535).astype(np.uint16)
            Image.fromarray(depth_mm).save(depth_dir / f"frame_{frame_idx:06d}.png")

            c2w_4x4 = np.eye(4, dtype=np.float64)
            c2w_4x4[:3, :4] = extrinsic_c2w_np
            np.savetxt(poses_dir / f"frame_{frame_idx:06d}.txt", c2w_4x4, "%.15e")

            intr_path = batch_dir / "intrinsics.json"
            if not intr_path.exists():
                with open(intr_path, "w") as f:
                    json.dump({
                        "fx": float(save_intr[0, 0]),
                        "fy": float(save_intr[1, 1]),
                        "cx": float(save_intr[0, 2]),
                        "cy": float(save_intr[1, 2]),
                        "w": orig_w,
                        "h": orig_h,
                    }, f)

            # ── 保存 depth_conf (用于尺度校准选最优帧) ──
            conf_dir = batch_dir / "conf"
            conf_dir.mkdir(exist_ok=True)
            np.save(str(conf_dir / f"frame_{frame_idx:06d}.npy"), _depth_conf_np.astype(np.float32))

            # ── 保存 point (置信度过滤后导出 npy) ──
            point_dir = batch_dir / "point"
            point_dir.mkdir(exist_ok=True)
            wp_filtered = _wp.copy()
            wp_filtered[_depth_conf_np < CONF_THRESHOLD] = 0.0
            np.save(str(point_dir / f"frame_{frame_idx:06d}.npy"), wp_filtered.astype(np.float32))

        t_now = time.time()
        t_start = infer_start if infer_start is not None else t_now
        t_fwd_end = forward_end if forward_end is not None else t_start
        total_ms = (t_now - t_start) * 1000
        fwd_ms = (t_fwd_end - t_start) * 1000
        decode_ms = (t_now - t_fwd_end) * 1000

        start_ts = datetime.fromtimestamp(t_start).strftime("%H:%M:%S.%f")[:-3]
        arrival = frame_arrival_times.get(frame_idx)
        wait_ms = (t_start - arrival) * 1000 if arrival else 0
        # NOTE: "[TS] 推理开始" 已由 frame_monitor_thread 在 forward 前打过一次，此处不再重复

        with _cache_lock:
            add_to_frame_cache(frame_idx, pred_pts, color_flat, conf_flat, camera)
            processed = max(frame_cache.keys()) + 1 if frame_cache else 0
        update_status(processed_frames=processed)

        # ── 推送帧数据到 DGSG 流式队列（优先于磁盘 I/O，降低单帧延迟）──
        if dgsg_running.is_set() and dgsg_pipeline is not None:
            try:
                c2w_4x4_dgsg = np.eye(4, dtype=np.float64)
                c2w_4x4_dgsg[:3, :4] = extrinsic_c2w_np
                frame_data_queue.put_nowait(
                    (frame_idx, image_np.copy(), _depth_np.copy(), _wp.copy(), c2w_4x4_dgsg)
                )
            except queue.Full:
                write_log(f"[DGSG] 队列满，丢弃帧#{frame_idx}", "err")

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        write_log(f"[TS] 缓存就绪 #{frame_idx} {ts} (解码{decode_ms:.0f}ms)", "info")

    except Exception as e:
        write_log(f"帧 {frame_idx} 处理失败: {e}", "err")
        traceback.print_exc()

def process_scale_frames(frames_dir, num_scale_frames=NUM_SCALE_FRAMES):
    """处理前N帧作为scale frames（Phase 1）"""
    write_log(f"初始化: 处理前 {num_scale_frames} 帧...", "info")
    
    # 获取所有帧路径
    exts = (".jpg", ".jpeg", ".png")
    all_paths = []
    for ext in exts:
        all_paths.extend(glob.glob(str(frames_dir / f"*{ext}")))
    all_paths = sorted(set(all_paths))
    
    if len(all_paths) < num_scale_frames:
        write_log(f"帧数不足，需要{num_scale_frames}帧，当前{len(all_paths)}帧", "err")
        return False
    
    # 取前N帧
    scale_paths = all_paths[:num_scale_frames]
    
    # 加载并预处理
    scale_images = []
    for path in scale_paths:
        img = load_single_image(path)
        scale_images.append(img)
    
    scale_tensor = torch.stack(scale_images, dim=0).unsqueeze(0).to(model_state["device"])
    
    # 清理KV缓存
    model_state["model"].clean_kv_cache()
    
    # Phase 1: Scale frames推理
    with torch.no_grad(), torch.amp.autocast(device_type=DEVICE_TYPE, dtype=DTYPE):
        scale_output = model_state["model"].forward(
            scale_tensor,
            num_frame_for_scale=num_scale_frames,
            num_frame_per_block=num_scale_frames,
            causal_inference=True,
        )
    
    # 处理每一帧的输出
    for i in range(num_scale_frames):
        # 获取模型输出尺寸（与 frame_monitor_thread 保持一致）
        depth = scale_output.get("depth")
        if depth is not None and depth.dim() >= 3 and depth.shape[1] > i:
            d = depth[:, i:i + 1]
            while d.ndim > 2:
                d = d[0] if d.shape[0] == 1 else d.squeeze()
            model_h, model_w = d.shape
        else:
            model_h, model_w = IMAGE_SIZE, IMAGE_SIZE
        
        # 将预处理后的图片缩放到模型输出尺寸
        img_tensor = scale_images[i]
        img_resized = img_tensor.permute(1, 2, 0).numpy()
        img_pil = Image.fromarray((img_resized * 255).clip(0, 255).astype(np.uint8))
        img_pil = img_pil.resize((model_w, model_h), Image.Resampling.BICUBIC)
        img_np = np.array(img_pil)
        
        frame_output = {}
        for k, v in scale_output.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] > i:
                frame_output[k] = v[:, i:i + 1]
            else:
                frame_output[k] = v
        
        t_now = time.time()
        frame_infer_start[i] = t_now
        frame_forward_end[i] = t_now  # scale 帧批量推理，无单独 forward 耗时
        on_frame_callback(i, img_np, frame_output)
        model_state["known_paths"].add(scale_paths[i])
    
    model_state["frame_idx"] = num_scale_frames
    # Seed motion detector with last scale frame
    detector = model_state.get("motion_detector")
    if detector is not None and len(scale_paths) > 0:
        gray = cv2.imread(scale_paths[-1], cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            detector._update_ref(gray)
    write_log(f"初始化完成, {num_scale_frames} 帧", "ok")
    
    del scale_output, scale_tensor
    return True

def frame_monitor_thread():
    """后台线程：监控帧文件夹，处理新帧"""
    global dgsg_pipeline, dgsg_thread, dgsg_finish_called
    write_log("帧监控线程启动", "info")
    
    scale_processed = False
    
    while not model_state["stop_event"].is_set():
        try:
            if not model_state["is_streaming"]:
                scale_processed = False
                time.sleep(0.1)
                continue
            if not model_state.get("_dbg_printed"):
                print(f'[DEBUG-b0] Monitor streaming active, batch={model_state["current_batch_id"]}', flush=True)
                model_state["_dbg_printed"] = True
            
            batch_id = model_state["current_batch_id"]
            if not batch_id:
                time.sleep(0.1)
                continue

            batch_dir = DATA_DIR / batch_id

            frames_dir = batch_dir / "frames"
            if not frames_dir.exists():
                write_log(f"帧文件夹不存在: {frames_dir}", "info")
                time.sleep(0.1)
                continue
            
            # Phase 1: 处理scale frames（只执行一次）
            if not scale_processed:
                exts = (".jpg", ".jpeg", ".png")
                all_paths = []
                for ext in exts:
                    all_paths.extend(glob.glob(str(frames_dir / f"*{ext}")))
                all_paths = sorted(set(all_paths))
                
                write_log(f"扫描帧: {len(all_paths)} 帧 (需 {NUM_SCALE_FRAMES})", "info")

                if len(all_paths) >= NUM_SCALE_FRAMES:
                    # ── 在 scale frames 推理前清理旧 DGSG 显存，防止 OOM ──
                    if dgsg_pipeline is not None:
                        write_log("[DGSG] 清理旧管线显存（scale frames 推理前）...", "info")
                        try:
                            if hasattr(dgsg_pipeline, 'clip_model'):
                                del dgsg_pipeline.clip_model
                            if hasattr(dgsg_pipeline, 'yolo_model'):
                                del dgsg_pipeline.yolo_model
                            if hasattr(dgsg_pipeline, 'sam_model'):
                                del dgsg_pipeline.sam_model
                            if hasattr(dgsg_pipeline, 'global_cloud'):
                                dgsg_pipeline.global_cloud = np.zeros((1, 7), dtype=np.float32)
                            if hasattr(dgsg_pipeline, 'objects'):
                                dgsg_pipeline.objects = []
                            if hasattr(dgsg_pipeline, 'scene_graph'):
                                del dgsg_pipeline.scene_graph
                            del dgsg_pipeline
                            dgsg_pipeline = None
                        except Exception:
                            write_log(f"[DGSG] 清理旧管线失败: {traceback.format_exc()}", "err")
                        torch.cuda.empty_cache()
                        mem_gb = torch.cuda.memory_allocated() / 1024**3
                        write_log(f"[DGSG] 清理后显存: {mem_gb:.1f}GB", "info")

                    success = process_scale_frames(frames_dir, NUM_SCALE_FRAMES)
                    if success:
                        scale_processed = True
                        # Update motion detector with real intrinsics from first frame
                        detector = model_state.get("motion_detector")
                        if detector is not None:
                            intr_path = batch_dir / "intrinsics.json"
                            if intr_path.exists():
                                with open(intr_path) as f_intr:
                                    _intr = json.load(f_intr)
                                detector.fx = _intr["fx"]
                                detector.fy = _intr["fy"]
                                detector.cx = _intr["cx"]
                                detector.cy = _intr["cy"]
                                write_log(f"Motion detector intrinsics updated: fx={_intr['fx']:.1f} fy={_intr['fy']:.1f}", "info")
                        # ── 初始化 DGSG 流式管线 ──
                        if dgsg_pipeline is None or not dgsg_running.is_set():
                            write_log("[DGSG] 初始化流式管线...", "info")

                            # 重置 finish 标志，确保下次 finish() 能被执行
                            dgsg_finish_called = False

                            try:
                                config_path = os.path.join(
                                    os.path.dirname(os.path.abspath(__file__)),
                                    "dgsg", "configs", "mydata", "lingbot_stream.py"
                                )
                                dgsg_config_mod = SourceFileLoader(
                                    "lingbot_stream", config_path
                                ).load_module()
                                dgsg_config = dict(dgsg_config_mod.config)
                                dgsg_config["run_name"] = "stream"

                                dgsg_pipeline = StreamingDGSG(dgsg_config, save_every_n_frames=0)

                                intr_path_dgsg = DATA_DIR / batch_id / "intrinsics.json"
                                if intr_path_dgsg.exists():
                                    with open(intr_path_dgsg) as f:
                                        _intr_dgsg = json.load(f)
                                    intr_mat = np.array([
                                        [_intr_dgsg['fx'], 0, _intr_dgsg['cx']],
                                        [0, _intr_dgsg['fy'], _intr_dgsg['cy']],
                                        [0, 0, 1],
                                    ], dtype=np.float64)
                                else:
                                    intr_mat = np.eye(3, dtype=np.float64)
                                dgsg_pipeline.set_intrinsics(intr_mat)

                                dgsg_cache.clear()
                                dgsg_running.set()
                                dgsg_finished.clear()

                                dgsg_thread = threading.Thread(target=_dgsg_consumer, daemon=True)
                                dgsg_thread.start()

                                write_log("[DGSG] 流式管线初始化完成，消费线程已启动", "ok")
                            except Exception:
                                write_log(f"[DGSG] 初始化失败: {traceback.format_exc()}", "err")
                        write_log("初始化完成, 开始逐帧推理", "ok")
                    else:
                        write_log("初始化失败, 重试...", "err")
                        time.sleep(0.1)
                        continue
                else:
                    write_log(f"帧数不足: {len(all_paths)}/{NUM_SCALE_FRAMES}", "info")
                    time.sleep(0.1)
                    continue
            
            # Phase 2/3: 处理后续帧
            exts = (".jpg", ".jpeg", ".png")
            current_paths = []
            for ext in exts:
                current_paths.extend(glob.glob(str(frames_dir / f"*{ext}")))
            current_paths = sorted(set(current_paths))
            
            new_paths = [p for p in current_paths if p not in model_state["known_paths"]]

            if new_paths:
                write_log(f"新帧: {len(new_paths)} 个", "info")
                _diag_stats["monitor_saw"] = _diag_stats.get("monitor_saw", 0) + len(new_paths)

                preload_future = None

                for i, path in enumerate(new_paths):
                    _diag_stats["loop_entry"] = _diag_stats.get("loop_entry", 0) + 1

                    if model_state["stop_event"].is_set():
                        _diag_stats["loop_stop_event"] = _diag_stats.get("loop_stop_event", 0) + (len(new_paths) - i)
                        break

                    if model_state["max_images"] is not None and model_state["frame_idx"] >= model_state["max_images"]:
                        write_log(f"已达最大帧数 {model_state['max_images']}, 停止", "info")
                        _diag_stats["loop_max_images"] = _diag_stats.get("loop_max_images", 0) + (len(new_paths) - i)
                        model_state["is_streaming"] = False
                        update_status(status="completed", processed_frames=model_state["frame_idx"])
                        break

                    frame_idx = model_state["frame_idx"]

                    # 获取预加载的帧（首帧同步加载，后续帧 forward 期间已预加载完成）
                    if preload_future is not None:
                        path, skip, browser_idx, img, frame_image = preload_future.result()
                    else:
                        path, skip, browser_idx, img, frame_image = _preload_frame(path)

                    if skip or browser_idx is None:
                        if skip:
                            _diag_stats["loop_motion_skip"] = _diag_stats.get("loop_motion_skip", 0) + 1
                        if not skip and browser_idx is None:
                            _diag_stats["preload_idx_none"] += 1
                            _diag_stats["loop_idx_none"] = _diag_stats.get("loop_idx_none", 0) + 1
                        model_state["known_paths"].add(path)
                        continue

                    _diag_stats["entered_forward"] += 1

                    # 在 forward 阻塞 CPU 前，触发下一帧的预加载
                    if i + 1 < len(new_paths):
                        preload_future = _preloader.submit(_preload_frame, new_paths[i + 1])

                    # 判断是否是关键帧
                    ki = model_state["keyframe_interval"]
                    is_keyframe = (ki <= 1) or \
                                  ((frame_idx - model_state["scale_frames"]) % ki == 0)

                    if not is_keyframe:
                        model_state["model"]._set_skip_append(True)

                    # 推理
                    t_infer0 = time.time()
                    frame_infer_start[browser_idx] = t_infer0
                    arrival = frame_arrival_times.get(browser_idx)
                    wait_ms = int((t_infer0 - arrival) * 1000) if arrival else 0
                    ts_start = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    write_log(f"[TS] 推理开始 #{browser_idx} {ts_start} (排队{wait_ms}ms)", "info")
                    with torch.no_grad(), torch.amp.autocast(device_type=DEVICE_TYPE, dtype=DTYPE):
                        frame_output = model_state["model"].forward(
                            frame_image,
                            num_frame_for_scale=model_state["scale_frames"],
                            num_frame_per_block=1,
                            causal_inference=True,
                        )
                    frame_forward_end[browser_idx] = time.time()
                    fwd_ms = int((frame_forward_end[browser_idx] - frame_infer_start.get(browser_idx, frame_forward_end[browser_idx])) * 1000)
                    ts_fwd = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    write_log(f"[TS] 推理成功 #{browser_idx} {ts_fwd} (推理{fwd_ms}ms)", "ok")
                    _diag_stats["inferred"] += 1

                    if not is_keyframe:
                        model_state["model"]._set_skip_append(False)

                    model_state["known_paths"].add(path)
                    model_state["frame_idx"] += 1

                    # 最小化 post-forward：直接入队，image tensor + GPU outputs 全交后台
                    _callback_queue.put((
                        browser_idx, img, frame_output,
                        frame_infer_start.pop(browser_idx, None),
                        frame_forward_end.pop(browser_idx, None),
                    ))
                    del frame_image
                    # img 和 frame_output 的所有权移交后台线程，这里不再 del
                
                write_log(f"推理完成, 共 {model_state['frame_idx']} 帧", "ok")
            else:
                if model_state.get("finish_requested"):
                    write_log(f"推理结束, 共 {model_state['frame_idx']} 帧", "ok")
                    model_state["is_streaming"] = False
            
            time.sleep(0.1)
            
        except Exception as e:
            write_log(f"监控线程异常: {e}", "err")
            traceback.print_exc()
            time.sleep(1)

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    write_log("服务启动中...", "info")
    
    if not model_state["initialized"]:
        model_state["model"], model_state["device"] = load_model()
        model_state["initialized"] = True
        
        # 启动监控线程
        monitor = threading.Thread(target=frame_monitor_thread, daemon=True)
        monitor.start()

        # 启动异步解码线程
        worker = threading.Thread(target=_callback_worker, daemon=True)
        worker.start()

        write_log("服务启动完成", "ok")

@app.get("/api/health")
async def health():
    return {"status": "running", "service": "LingBot-MAP Streaming Service"}

@app.post("/batch/{batch_id}/start_inference")
async def start_inference(batch_id: str, body: dict):
    """开始流式推理"""
    batch_dir = DATA_DIR / batch_id
    frames_dir = batch_dir / "frames"

    # 固定 batch_id 场景：清理旧数据避免残留帧干扰
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    
    # 获取已上传的帧数（保留之前上传的帧计数）
    exts = (".jpg", ".jpeg", ".png")
    all_paths = []
    for ext in exts:
        all_paths.extend(glob.glob(str(frames_dir / f"*{ext}")))
    existing_frames = len(sorted(set(all_paths)))
    
    ki = body.get("keyframe_interval", KEYFRAME_INTERVAL_DEFAULT)
    max_img = body.get("max_images", None)
    
    # 检查是否有推理任务正在运行（单用户架构，不支持并发）
    if model_state["is_streaming"]:
        raise HTTPException(
            status_code=409,
            detail="当前有推理任务正在运行，请先结束当前任务"
        )

    # 按需加载模型（上次推理完成后已卸载释放显存）
    if model_state["model"] is None:
        write_log("加载 lingbot-map 模型...", "info")
        try:
            model_state["model"], model_state["device"] = load_model()
        except Exception as e:
            write_log(f"模型加载失败: {traceback.format_exc()}", "err")
            raise HTTPException(status_code=500, detail=f"模型加载失败: {e}")

    model_state["current_batch_id"] = batch_id
    model_state["frame_idx"] = 0
    model_state["keyframe_interval"] = ki
    model_state["max_images"] = max_img
    model_state["known_paths"] = set()
    model_state["is_streaming"] = True
    model_state["stop_event"].clear()
    global dgsg_finish_called
    batch_status["dgsg_status"] = "idle"  # Reset from previous run
    dgsg_finish_called = False
    model_state["finish_requested"] = False
    model_state["all_predictions"] = {
        "pose_enc": [],
        "depth": [],
        "depth_conf": [],
        "world_points": [],
        "world_points_conf": [],
        "images": [],
    }

    # 诊断计数器重置
    global _diag_stats
    _diag_stats = {
        "uploaded": 0,
        "motion_skipped": 0,
        "motion_kept": 0,
        "inferred": 0,
        "preload_error": 0,
        "preload_idx_none": 0,
        "monitor_saw": 0,
        "entered_forward": 0,
        "scale_frames": NUM_SCALE_FRAMES,
    }

    frame_cache.clear()
    cache_frame_order.clear()
    frame_arrival_times.clear()
    frame_infer_start.clear()
    frame_forward_end.clear()
    batch_logs.clear()

    # 清空 DGSG 队列残留（上次运行的 sentinel 等）
    while not frame_data_queue.empty():
        try:
            frame_data_queue.get_nowait()
        except queue.Empty:
            break

    # 清空回调队列残留（上次运行的遗留帧）
    while not _callback_queue.empty():
        try:
            _callback_queue.get_nowait()
        except queue.Empty:
            break

    # Initialize motion detector with intrinsics
    intr_path = batch_dir / "intrinsics.json"
    if intr_path.exists():
        with open(intr_path) as f:
            _intr = json.load(f)
        fx, fy, cx, cy = _intr["fx"], _intr["fy"], _intr["cx"], _intr["cy"]
    else:
        fx = fy = float(IMAGE_SIZE)
        cx, cy = float(IMAGE_SIZE) / 2, float(IMAGE_SIZE) / 2
    model_state["motion_detector"] = MotionDetector(fx, fy, cx, cy, trans_thresh=4.0, rot_thresh=0.0, flow_fallback_thresh=1.5)

    # 清理KV缓存，准备新批次
    if model_state["model"]:
        model_state["model"].clean_kv_cache()
    
    update_status(
        batch_id=batch_id,
        status="streaming",
        total_frames=existing_frames,      # ✅ 保留已上传的帧数
        uploaded_frames=existing_frames,   # ✅ 保留已上传的帧数
        processed_frames=0,
        total_points=0,
    )
    
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    write_log(f"[TS] 推理管线启动 {ts}", "ok")
    write_log(f"开始流式推理: {batch_id}, 已上传 {existing_frames} 帧", "info")

    return {"success": True, "batch_id": batch_id, "message": "流式推理已启动", "existing_frames": existing_frames}

@app.post("/batch/{batch_id}/frames")
async def upload_frames(batch_id: str, files: list[UploadFile] = File(...)):
    """上传帧到批次"""
    batch_dir = DATA_DIR / batch_id
    frames_dir = batch_dir / "frames"

    frames_dir.mkdir(parents=True, exist_ok=True)

    # 用最大帧号 +1 作为起始索引，而非文件计数
    # (motion detector 会删除帧文件，导致 len(glob) < max_idx+1)
    max_idx = -1
    for p in frames_dir.glob("frame_*.jpg"):
        try:
            idx = int(p.stem.split('_')[1])
            if idx > max_idx:
                max_idx = idx
        except (ValueError, IndexError):
            pass
    next_idx = max_idx + 1
    existing_count = len(list(frames_dir.glob("frame_*.jpg")))

    if next_idx + len(files) > MAX_FRAMES_PER_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"超过最大帧数量 {MAX_FRAMES_PER_BATCH}"
        )

    saved_count = 0
    for i, file in enumerate(files):
        if file.content_type not in SUPPORTED_FORMATS:
            raise HTTPException(status_code=400, detail=f"不支持的格式: {file.content_type}")

        frame_data = await file.read()
        frame_index = next_idx + i
        frame_path = frames_dir / f"frame_{frame_index:06d}.jpg"

        overwrite = frame_path.exists()
        if overwrite:
            write_log(f"[DEBUG-a7c3] 覆盖写入: {frame_path.name}", "info")

        with open(frame_path, 'wb') as f:
            f.write(frame_data)

        frame_arrival_times[frame_index] = time.time()
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        write_log(f"[TS] 帧到达 #{frame_index} {ts}", "info")
        saved_count += 1
        _diag_stats["uploaded"] += 1

    total_frames = next_idx + saved_count
    update_status(
        total_frames=total_frames,
        uploaded_frames=total_frames
    )

    return {
        "success": True,
        "batch_id": batch_id,
        "uploaded_count": saved_count,
        "total_frames": total_frames
    }

@app.post("/batch/{batch_id}/finish_inference")
async def finish_inference(batch_id: str):
    """完成推理：停止 lingbot → 等待 DGSG 消费完 → 卸载模型 → VLM + 保存"""
    model_state["finish_requested"] = True
    write_log("收到结束请求，等待监控线程处理完剩余帧...", "info")

    # Phase 1: 等待 lingbot 推理线程处理完剩余帧
    wait_count = 0
    while model_state["is_streaming"] and wait_count < 240:
        await asyncio.sleep(0.5)
        wait_count += 1
    model_state["is_streaming"] = False
    model_state["finish_requested"] = False

    # 现在打印 DIAG 统计（监控线程已处理完所有帧）
    l1k = _diag_stats.get("l1_kept", 0)
    l1s = _diag_stats.get("l1_skipped", 0)
    l2k = _diag_stats.get("l2_kept", 0)
    l2s = _diag_stats.get("l2_skipped", 0)
    write_log(
        f"[DIAG] 帧数统计: 上传={_diag_stats['uploaded']} | "
        f"scale={_diag_stats['scale_frames']} | "
        f"监控扫描={_diag_stats.get('monitor_saw', 0)} | "
        f"循环入口={_diag_stats.get('loop_entry', 0)} | "
        f"motion保留={_diag_stats['motion_kept']} | "
        f"motion跳过={_diag_stats['motion_skipped']} | "
        f"L1保留={l1k}/L1跳过={l1s} | L2保留={l2k}/L2跳过={l2s} | "
        f"max_images截断={_diag_stats.get('loop_max_images', 0)} | "
        f"stop_event截断={_diag_stats.get('loop_stop_event', 0)} | "
        f"预加载失败={_diag_stats.get('preload_error', 0)} | "
        f"通过motion但未推理={_diag_stats.get('preload_idx_none', 0)} | "
        f"进入推理循环={_diag_stats.get('entered_forward', 0)} | "
        f"实际推理={_diag_stats['inferred']}",
        "info"
    )

    # Phase 2: 通知 DGSG 消费线程结束（sentinel），非阻塞等待
    write_log("[DGSG] 发送结束信号，等待消费线程处理完队列中的剩余帧...", "info")
    try:
        frame_data_queue.put(None, timeout=30)
    except queue.Full:
        write_log("[DGSG] 队列满，强制结束", "err")

    # 非阻塞等待消费线程退出（避免阻塞 event loop 导致 502）
    if dgsg_thread is not None:
        for _ in range(600):  # max 300s with 0.5s interval
            if dgsg_finished.is_set():
                break
            await asyncio.sleep(0.5)
    dgsg_running.clear()
    dgsg_finished.set()

    # DGSG 处理帧数用 frame_idx（inference 实际处理的帧数），不是 dgsg_cache 大小
    total_processed = max(frame_cache.keys()) + 1 if frame_cache else 0
    write_log(f"推理完成，共 {total_processed} 帧 (DGSG缓存 {len(dgsg_cache)} 帧)", "ok")

    # Phase 3: 卸载推理模型（lingbot-map）
    if model_state["model"] is not None:
        write_log("卸载 lingbot-map 模型...", "info")
        del model_state["model"]
        model_state["model"] = None
        torch.cuda.empty_cache()
        write_log("模型已卸载", "ok")

    # Phase 4: DGSG finish — VLM 描述 + 全量重算关系 + 保存
    global dgsg_finish_called
    if dgsg_pipeline is not None and not dgsg_finish_called:
        dgsg_finish_called = True
        write_log("[DGSG] 开始 finish: VLM 描述 + 关系重算 + 保存...", "info")
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, dgsg_pipeline.finish)
            write_log(
                f"[DGSG] finish 完成: {result['total_objects']} objects, "
                f"{result['total_points']} points, "
                f"{result['scene_graph_nodes']} scene graph nodes",
                "ok"
            )
            update_status(dgsg_status="done")
            # Export GS point cloud as PLY
            try:
                gs_dir = DATA_DIR / "gaussian-splats"
                gs_dir.mkdir(parents=True, exist_ok=True)
                ply_path = gs_dir / f"{batch_id}.ply"
                if dgsg_pipeline is not None and hasattr(dgsg_pipeline, 'export_ply'):
                    loop2 = asyncio.get_event_loop()
                    await loop2.run_in_executor(None, dgsg_pipeline.export_ply, str(ply_path))
                    write_log(f"[DGSG] PLY exported: {ply_path} ({ply_path.stat().st_size / 1024 / 1024:.1f}MB)", "ok")
                else:
                    write_log(f"[DGSG] PLY export skipped (no export_ply method)", "warn")
            except Exception as _ply_err:
                write_log(f"[DGSG] PLY export failed: {_ply_err}", "err")
        except Exception as e:
            write_log(f"[DGSG] finish 失败: {traceback.format_exc()}", "err")
            update_status(dgsg_status="error")
    else:
        update_status(dgsg_status="idle")

    # Update latest symlink
    latest_link = DATA_DIR / "latest"
    batch_dir = DATA_DIR / batch_id
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    os.symlink(batch_dir, latest_link)

    update_status(
        status="completed",
        processed_frames=total_processed,
    )

    write_log("finish_inference 完成", "ok")
    return {
        "success": True,
        "batch_id": batch_id,
        "total_frames": total_processed,
    }

@app.post("/batch/{batch_id}/force_stop")
async def force_stop(batch_id: str):
    """强制停止：立即终止处理，清空所有缓存和状态，删除数据目录"""
    global dgsg_pipeline
    model_state["is_streaming"] = False
    model_state["finish_requested"] = False
    model_state["current_batch_id"] = None
    model_state["frame_idx"] = 0
    model_state["known_paths"] = set()
    model_state["all_predictions"] = {
        "pose_enc": [],
        "depth": [],
        "depth_conf": [],
        "world_points": [],
        "world_points_conf": [],
        "images": [],
    }
    
    frame_cache.clear()
    cache_frame_order.clear()
    frame_arrival_times.clear()
    frame_infer_start.clear()
    frame_forward_end.clear()
    batch_logs.clear()
    # ── 清理 DGSG 状态 ──
    dgsg_running.clear()
    dgsg_finished.clear()
    while not frame_data_queue.empty():
        try:
            frame_data_queue.get_nowait()
        except queue.Empty:
            break
    with dgsg_cache_lock:
        dgsg_cache.clear()
    if dgsg_pipeline is not None:
        del dgsg_pipeline
        dgsg_pipeline = None
    batch_status.clear()
    
    if model_state["model"]:
        model_state["model"].clean_kv_cache()
    
    batch_dir = DATA_DIR / batch_id
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    
    write_log(f"强制停止完成: {batch_id}, 所有数据已清空", "info")
    
    return {"success": True, "batch_id": batch_id, "message": "所有处理已终止，数据已清空"}

@app.get("/batch/{batch_id}/status")
async def get_status(batch_id: str):
    """获取批次状态"""
    return batch_status

@app.get("/batch/{batch_id}/logs")
async def get_logs(batch_id: str):
    """获取日志"""
    return {"logs": batch_logs}

@app.get("/batch/{batch_id}/metadata")
async def get_metadata(batch_id: str):
    """获取批次元数据（与 see/ 一致）"""
    batch_dir = DATA_DIR / batch_id
    frames_dir = batch_dir / "frames"
    
    # 获取帧数量
    exts = (".jpg", ".jpeg", ".png")
    all_paths = []
    for ext in exts:
        all_paths.extend(glob.glob(str(frames_dir / f"*{ext}")))
    num_frames = len(sorted(set(all_paths)))
    
    # 获取图片尺寸（从第一张图获取）
    image_width = 518
    image_height = 518
    if all_paths:
        try:
            first_frame = sorted(set(all_paths))[0]
            img = Image.open(first_frame)
            image_width, image_height = img.size
        except:
            pass
    
    # 计算场景中心和尺度（从缓存的点云计算）
    all_points = []
    for frame_idx in frame_cache:
        points = np.asarray(frame_cache[frame_idx]["points"])
        all_points.append(points)
    
    if all_points:
        all_points_np = np.vstack(all_points)
        scene_center = np.mean(all_points_np, axis=0).tolist()
        scene_min = np.min(all_points_np, axis=0)
        scene_max = np.max(all_points_np, axis=0)
        scene_scale = float(np.max(scene_max - scene_min)) or 1.0
    else:
        scene_center = [0.0, 0.0, 0.0]
        scene_scale = 1.0
    
    metadata = {
        "scene_center": scene_center,
        "scene_scale": scene_scale,
        "num_frames": num_frames,
        "image_width": image_width,
        "image_height": image_height,
        "processed_frames": max(frame_cache.keys()) + 1 if frame_cache else 0,
    }
    
    return {"success": True, "metadata": metadata}

@app.get("/batch/{batch_id}/frame/{frame_index}/point_cloud")
async def get_frame_point_cloud(batch_id: str, frame_index: int):
    """获取单帧点云（从内存缓存）"""
    if frame_index not in frame_cache:
        raise HTTPException(status_code=404, detail=f"帧 {frame_index} 尚未处理完成")
    
    cached = frame_cache[frame_index]
    
    # 返回原始数据（与 live_camera.py 一致，不过滤）
    points_arr = np.asarray(cached["points"], dtype=np.float32)
    colors_arr = np.asarray(cached["colors"], dtype=np.float32)
    confs_arr = np.asarray(cached["confs"], dtype=np.float32)
    
    if len(points_arr) == 0:
        raise HTTPException(status_code=404, detail=f"帧 {frame_index} 无有效点云数据")
    
    # 二进制编码传输，gzip 压缩（减少网络传输时间）
    n = np.uint32(len(points_arr))
    buf = n.tobytes() + points_arr.tobytes() + colors_arr.tobytes() + confs_arr.tobytes()
    compressed = gzip.compress(buf, compresslevel=1)
    return Response(
        content=compressed,
        media_type="application/octet-stream",
        headers={
            "X-Inference-Time": str(cached["inference_time"]),
            "Content-Encoding": "gzip",
        },
    )

@app.get("/batch/{batch_id}/frame/{frame_index}/camera")
async def get_frame_camera(batch_id: str, frame_index: int):
    """获取单帧相机参数"""
    if frame_index not in frame_cache:
        raise HTTPException(status_code=404, detail=f"帧 {frame_index} 尚未处理完成")
    
    cached = frame_cache[frame_index]
    
    return {
        "success": True,
        "batch_id": batch_id,
        "frame_index": frame_index,
        "camera": cached["camera"]
    }

@app.get("/batch/{batch_id}/frame/{frame_index}/image")
async def get_frame_image(batch_id: str, frame_index: int):
    """获取单帧原始图片"""
    batch_dir = DATA_DIR / batch_id
    frames_dir = batch_dir / "frames"
    
    image_path = frames_dir / f"frame_{frame_index:06d}.jpg"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"帧 {frame_index} 图片不存在")
    
    return FileResponse(image_path, media_type="image/jpeg")


@app.get("/batch/{batch_id}/frame/{frame_index}/dgsg")
async def get_frame_dgsg(batch_id: str, frame_index: int):
    """获取单帧 DGSG 语义快照"""
    with dgsg_cache_lock:
        if frame_index not in dgsg_cache:
            raise HTTPException(status_code=404, detail=f"DGSG 帧 {frame_index} 尚未处理完成")
        snapshot = dgsg_cache[frame_index]
        pc = snapshot['point_cloud']

        # 编码前最终验证
        colors = pc['rgb_colors']
        c_min, c_max = float(colors.min()), float(colors.max())
        if not np.isfinite(colors).all() or c_min < -0.1 or c_max > 1.1:
            write_log(
                f"[API] get_frame_dgsg 帧#{frame_index} 编码前颜色异常: "
                f"range=[{c_min:.4f}, {c_max:.4f}], "
                f"finite={bool(np.isfinite(colors).all())}, dtype={colors.dtype}, "
                f"owndata={colors.flags['OWNDATA']}, contiguous={colors.flags['C_CONTIGUOUS']}",
                "err"
            )

        payload = {
            "frame_id": snapshot['frame_id'],
            "point_cloud": {
                "means3D": _b64(pc['means3D'].tobytes()),
                "rgb_colors": _b64(pc['rgb_colors'].tobytes()),
                "object_idx": _b64(pc['object_idx'].tobytes()),
                "log_scales": _b64(pc['log_scales'].tobytes()),
                "unnorm_rotations": _b64(pc['unnorm_rotations'].tobytes()),
                "logit_opacities": _b64(pc['logit_opacities'].tobytes()),
            },
            "objects": [
                {
                    "idx": obj['idx'],
                    "class_name": obj.get('class_name', 'unknown'),
                    "center_3d": obj['center_3d'] if isinstance(obj['center_3d'], list)
                        else obj['center_3d'].tolist(),
                }
                for obj in snapshot.get('objects', [])
            ],
            "changes": {
                "added_obj_ids": snapshot['changes'].get('added_obj_ids', []),
                "removed_obj_ids": snapshot['changes'].get('removed_obj_ids', []),
                "updated_obj_ids": snapshot['changes'].get('updated_obj_ids', []),
            },
            "stats": {
                "total_objects": snapshot['stats'].get('total_objects', 0),
                "total_points": int(snapshot['stats'].get('total_points', 0)),
            },
        }
        return payload


def _dgsg_status_str():
    """获取 DGSG 状态字符串。
    "done" 只在 finish_inference 显式设置后返回（npz 已生成），
    不依赖 dgsg_finished 事件（消费线程退出 ≠ VLM+保存完成）。
    """
    explicit = batch_status.get("dgsg_status")
    if explicit == "done":
        return "done"
    if explicit == "error":
        return "error"
    if dgsg_pipeline is None:
        return "idle"
    elif dgsg_running.is_set():
        return "streaming"
    elif dgsg_finished.is_set():
        return "finishing"
    else:
        return "idle"


@app.get("/batch/{batch_id}/dgsg_status")
async def get_dgsg_status(batch_id: str):
    """获取 DGSG 流式管线状态"""
    with dgsg_cache_lock:
        latest_frame = max(dgsg_cache.keys()) if dgsg_cache else -1
        total_processed = max(dgsg_cache.keys()) + 1 if dgsg_cache else 0

    return {
        "status": _dgsg_status_str(),
        "latest_frame": latest_frame,
        "total_processed": total_processed,
    }


@app.get("/batch/{batch_id}/dgsg_objects")
async def get_dgsg_objects(batch_id: str):
    """获取 DGSG 当前对象列表+变更事件（不含点云数据，响应体几 KB）"""
    with dgsg_cache_lock:
        if not dgsg_cache:
            return {
                "status": _dgsg_status_str(),
                "latest_frame": -1,
                "total_processed": 0,
                "objects": [],
                "changes": {"added_obj_ids": [], "removed_obj_ids": [], "updated_obj_ids": []},
            }
        latest_frame = max(dgsg_cache.keys())
        snapshot = dgsg_cache[latest_frame]

    objs = snapshot.get("objects", [])
    changes = snapshot.get("changes", {})
    write_log(
        f"[DGSG-API] dgsg_objects frame={latest_frame} objs={len(objs)} "
        f"+{len(changes.get('added_obj_ids', []))}/"
        f"-{len(changes.get('removed_obj_ids', []))}/"
        f"~{len(changes.get('updated_obj_ids', []))} "
        f"status={_dgsg_status_str()}",
        "info"
    )

    return {
        "status": _dgsg_status_str(),
        "latest_frame": latest_frame,
        "total_processed": len(dgsg_cache),
        "objects": [
            {
                "idx": obj["idx"],
                "class_name": obj.get("class_name", "unknown"),
                "center_3d": obj["center_3d"] if isinstance(obj["center_3d"], list)
                    else obj["center_3d"].tolist() if hasattr(obj["center_3d"], "tolist")
                    else list(obj["center_3d"]),
            }
            for obj in objs
        ],
        "changes": {
            "added_obj_ids": changes.get("added_obj_ids", []),
            "removed_obj_ids": changes.get("removed_obj_ids", []),
            "updated_obj_ids": changes.get("updated_obj_ids", []),
        },
    }


@app.get("/batch/{batch_id}/scene_graph")
async def get_scene_graph(batch_id: str):
    """获取最终 scene_graph（finish 后，含 VLM 描述）"""
    if dgsg_pipeline is None:
        raise HTTPException(status_code=404, detail="DGSG 管线未初始化")

    sg = dgsg_pipeline.scene_graph.get_snapshot()
    for node in sg.get('nodes', []):
        sg_node = dgsg_pipeline.scene_graph._nodes.get(node['idx'])
        if sg_node:
            node['description'] = getattr(sg_node, 'description', '')
            node['category'] = getattr(sg_node, 'category', node.get('category', ''))

    return {"success": True, "scene_graph": sg}


@app.get("/batch/{batch_id}/dgsg_pointcloud")
async def get_dgsg_pointcloud(batch_id: str):
    """返回 DGSG finish 输出的 params_with_idx.npz，转为前端二进制格式"""
    # 优先用 pipeline 的路径，fallback 用约定路径（重启后 pipeline 为 None）
    if dgsg_pipeline is not None:
        npz_path = os.path.join(dgsg_pipeline.output_dir, "params_with_idx.npz")
    else:
        _dgsg_exp_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "dgsg", "experiments", "mydata", "stream"
        )
        npz_path = os.path.join(_dgsg_exp_dir, "params_with_idx.npz")

    if not os.path.exists(npz_path):
        raise HTTPException(status_code=404, detail=f"params_with_idx.npz 不存在: {npz_path}")

    data = np.load(npz_path)
    positions = data["means3D"].astype(np.float32)
    colors = data["rgb_colors"].astype(np.float32)
    obj_idx = data["object_idx"].astype(np.uint16)

    N = np.uint32(len(positions))
    write_log(f"[DGSG-API] dgsg_pointcloud: N={N}, npz={npz_path}", "info")

    # 格式: [4B uint32 N] [N*12B positions] [N*12B colors] [N*2B object_idx]
    buf = N.tobytes() + positions.tobytes() + colors.tobytes() + obj_idx.tobytes()
    compressed = gzip.compress(buf, compresslevel=1)
    return Response(
        content=compressed,
        media_type="application/octet-stream",
        headers={"Content-Encoding": "gzip"},
    )


# ============================================================
# /vlm/* — Sentrix Monitor 实时第一视角语义标注
# 设计稿：docs/superpowers/specs/2026-06-16-sentrix-live-camera-design.md
# ============================================================

import io as _io
import asyncio as _asyncio

# 模型 + 锁（懒加载，避免拖慢 startup）
_vlm_models = {
    "yolo": None,         # ultralytics YOLO 实例
    "yolo_ready": False,  # warm-up 完成
    "moondream": None,    # MoondreamVLM 实例
    "moondream_ready": False,
}
_yolo_lock = _asyncio.Lock()
_vlm_lock = _asyncio.Lock()  # 与 batch finish 阶段的 VLM 共用，防止显存峰值冲突


def _resolve_lang_config():
    """加载 dgsg 的 lang 配置，复用 yolo / moondream 路径。"""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dgsg", "configs", "mydata", "lingbot_stream.py"
    )
    mod = SourceFileLoader("lingbot_stream_for_vlm", config_path).load_module()
    return dict(mod.config)["lang"]


def _ensure_yolo():
    if _vlm_models["yolo_ready"]:
        return _vlm_models["yolo"]

    if _vlm_models["yolo"] is None:
        from ultralytics import YOLO
        lang = _resolve_lang_config()
        write_log(f"[VLM] 加载独立 YOLO: {lang['yolo_model_path']}", "info")
        _vlm_models["yolo"] = YOLO(lang["yolo_model_path"])
        # YOLO-World 需要 set_classes
        if hasattr(_vlm_models["yolo"], "set_classes"):
            with open(lang["classes_file"]) as f:
                classes = [l.strip() for l in f if l.strip()]
            _vlm_models["yolo"].set_classes(classes)
        # warm-up
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        _ = _vlm_models["yolo"](dummy, verbose=False)
        _vlm_models["yolo_ready"] = True
        write_log("[VLM] 独立 YOLO warm-up 完成", "ok")

    return _vlm_models["yolo"]


def _ensure_moondream():
    if _vlm_models["moondream_ready"]:
        return _vlm_models["moondream"]

    if _vlm_models["moondream"] is None:
        # MoondreamVLM 在 dgsg/vlm_utils/，需要把目录加 sys.path
        _VLM_UTILS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dgsg", "vlm_utils")
        if _VLM_UTILS not in sys.path:
            sys.path.insert(0, _VLM_UTILS)
        from moondream_local import MoondreamVLM
        lang = _resolve_lang_config()
        write_log(f"[VLM] 加载独立 Moondream: {lang['moondream_model_path']}", "info")
        _vlm_models["moondream"] = MoondreamVLM(
            model_path=lang["moondream_model_path"],
            load_in_4bit=False,
        )
        _vlm_models["moondream_ready"] = True
        write_log("[VLM] Moondream 加载完成", "ok")

    return _vlm_models["moondream"]


def _decode_upload(file_bytes: bytes):
    """JPEG/PNG bytes → (np.ndarray HWC RGB, PIL.Image, w, h)"""
    pil = Image.open(_io.BytesIO(file_bytes)).convert("RGB")
    arr = np.array(pil)  # HWC RGB
    return arr, pil, pil.width, pil.height


@app.post("/vlm/detect")
async def vlm_detect(file: UploadFile = File(...)):
    """实时单帧 YOLO 检测（开词类）。

    Returns: { boxes:[{xyxy,cls,conf}], img_w, img_h, infer_ms, ts }
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail=f"不支持的格式: {file.content_type}")

    raw = await file.read()
    if len(raw) > 4 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="单帧 > 4 MB")

    try:
        arr_rgb, _pil, w, h = _decode_upload(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像解码失败: {e}")

    async with _yolo_lock:
        try:
            yolo = _ensure_yolo()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"YOLO 未就绪: {e}")

        t0 = time.time()
        try:
            # ultralytics 接受 np.ndarray (HWC, RGB or BGR)；YOLO-World 内部处理
            results = yolo(arr_rgb, verbose=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"YOLO 推理异常: {e}")
        infer_ms = (time.time() - t0) * 1000.0

    boxes_out = []
    if results and len(results) > 0:
        r = results[0]
        names = r.names if hasattr(r, "names") else {}
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else np.asarray(r.boxes.xyxy)
            confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, "cpu") else np.asarray(r.boxes.conf)
            clsids = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes.cls, "cpu") else np.asarray(r.boxes.cls).astype(int)
            for i in range(len(xyxy)):
                if float(confs[i]) < 0.25:
                    continue
                cid = int(clsids[i])
                cls_name = names.get(cid, str(cid)) if isinstance(names, dict) else (
                    names[cid] if cid < len(names) else str(cid)
                )
                boxes_out.append({
                    "xyxy": [float(xyxy[i][0]), float(xyxy[i][1]),
                             float(xyxy[i][2]), float(xyxy[i][3])],
                    "cls": cls_name,
                    "conf": float(confs[i]),
                })

    return {
        "boxes": boxes_out,
        "img_w": w,
        "img_h": h,
        "infer_ms": round(infer_ms, 2),
        "ts": time.time(),
    }


_DEFAULT_CAPTION_PROMPT = "Describe what you see in this first-person view in one short sentence."

@app.post("/vlm/caption")
async def vlm_caption(file: UploadFile = File(...), prompt: Optional[str] = None):
    """实时单帧 VLM 中文描述。

    与 batch finish 阶段 VLM 共用 _vlm_lock。第二个并发 caption 请求返 429。
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail=f"不支持的格式: {file.content_type}")

    if _vlm_lock.locked():
        # 单路语义：第二个并发直接 429（注意：此检查不是原子的，但够用）
        raise HTTPException(status_code=429, detail="caption busy")

    raw = await file.read()
    if len(raw) > 4 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="单帧 > 4 MB")

    try:
        _arr, pil, _w, _h = _decode_upload(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像解码失败: {e}")

    use_prompt = prompt or _DEFAULT_CAPTION_PROMPT

    async with _vlm_lock:
        try:
            md = _ensure_moondream()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Moondream 未就绪: {e}")

        t0 = time.time()
        try:
            # 用 run_in_executor 防止阻塞事件循环
            loop = _asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None,
                lambda: md.generate_content(pil, use_prompt, max_new_tokens=120)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Moondream 推理异常: {e}")
        infer_ms = (time.time() - t0) * 1000.0

    return {
        "caption": (text or "").strip(),
        "infer_ms": round(infer_ms, 2),
        "ts": time.time(),
    }

