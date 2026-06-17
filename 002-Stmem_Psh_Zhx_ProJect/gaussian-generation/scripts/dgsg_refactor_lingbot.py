import argparse
import os
import sys
import time
import gzip
import pickle
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import open_clip
from scipy.optimize import linear_sum_assignment
from ultralytics import SAM, YOLO
import matplotlib.pyplot as plt
import supervision as sv
import torchvision
import open3d as o3d

from PIL import Image
from scipy.ndimage import binary_erosion
from utils.object_helpers import ObjectClasses
from datasets.gradslam_datasets import (
    load_dataset_config, ICLDataset, ReplicaDataset, ReplicaV2Dataset,
    AzureKinectDataset, OrbbecDataset, HiSLAMDataset,
    ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset, TUMDataset,
    ScannetPPDataset, NeRFCaptureDataset, MydataDataset
)
from utils.common_utils import seed_everything




# ============================================================================
# 配置参数
# ============================================================================

VOXEL_SIZE = 0.01  # 物体点云体素下采样粒度 1cm
STRIDE_DOWNSAMPLE = 8  # 全局点云 stride 降采样步长（与 demo.py 可视化一致）
MAX_POINTS = 20_000_000  # 全局点云最大点数 buffer (逐帧 stride-8 后约 2M，20M 留足余量)
MIN_MASK_AREA = 2000
MAX_POINTS_PER_OBJECT = 200000 # 体素下采样阈值点
MIN_VISIBLE_POINTS = 50
DILATION_KERNEL = (5, 5)
DILATION_ITERATIONS = 2

MATCH_THRESHOLD = 0.3
WEIGHT_IOU = 0.6
WEIGHT_CENTER = 0.2
WEIGHT_CLIP = 0.2

DEPTH_ERROR_THRESHOLD = 0.1
COLOR_ERROR_THRESHOLD = 50.0



# ============================================================================
# 数据结构
# ============================================================================

class ObjectMemory:
    """物体记忆"""
    def __init__(self, obj_id, class_name, clip_feature, center_3d, mask, view_score, points_3d=None, image_crop=None, masked_crop=None):
        self.id = obj_id
        self.class_name = class_name
        self.clip_feature = clip_feature
        self.center_3d = center_3d
        self.mask = mask
        self.nums = 1
        self.points_3d = points_3d if points_3d is not None else np.zeros((0, 7), dtype=np.float32)  # xyz+rgb+obj_idx
        self.image_crop = image_crop
        self.masked_crop = masked_crop  # mask遮罩后的crop（VLM输入）
        self.view_score = view_score


# ============================================================================
# GPU加速投影函数 (from point2mask.py)
# ============================================================================

def to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, dtype=dtype, device=device)


def frustum_culling_gpu(points, R, T, K, img_w, img_h, device):
    """GPU视锥剔除，返回相机坐标系的点和2D投影点，以及有效点掩码"""
    pts = to_tensor(points, device)
    R_t = to_tensor(R, device)
    T_t = to_tensor(T.reshape(-1), device)
    K_t = to_tensor(K, device)

    pts_cam = (R_t @ pts.T).T + T_t

    valid_z = pts_cam[:, 2] > 0
    pts_cam_z = pts_cam[valid_z]

    if pts_cam_z.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), np.zeros(len(points), dtype=bool)

    pts_proj = (K_t @ pts_cam_z.T).T
    pts_2d = pts_proj[:, :2] / pts_proj[:, 2:3]

    in_frame = (
        (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < img_w) &
        (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < img_h)
    )

    valid_mask = np.zeros(len(points), dtype=bool)
    valid_indices = np.where(valid_z.cpu().numpy())[0]
    valid_mask[valid_indices] = in_frame.cpu().numpy()

    return pts_cam_z[in_frame].cpu().numpy(), pts_2d[in_frame].cpu().numpy(), valid_mask


def voxel_downsample_gpu(points, target_n=500, device="cuda"):
    if len(points) <= target_n:
        return points

    pts = to_tensor(points, device)
    bbox = pts.max(dim=0).values - pts.min(dim=0).values
    volume = bbox.clamp(min=1e-6).prod().item()
    voxel_size = (volume / target_n) ** (1.0 / 3.0)

    for _ in range(15):
        voxel_idx = torch.floor(pts / voxel_size).to(torch.int64)
        P1, P2, P3 = 73856093, 19349663, 83492791
        h = (voxel_idx[:, 0] * P1) ^ (voxel_idx[:, 1] * P2) ^ (voxel_idx[:, 2] * P3)

        _, inv_idx = torch.unique(h, sorted=False, return_inverse=True)
        n_unique = int(inv_idx.max().item()) + 1

        first_idx = torch.zeros(n_unique, dtype=torch.long, device=device)
        first_idx.scatter_(0, inv_idx, torch.arange(len(pts), dtype=torch.long, device=device))
        unique_pts = pts[first_idx]

        if n_unique <= target_n:
            return unique_pts.cpu().numpy()

        voxel_size *= (n_unique / target_n) ** (1.0 / 3.0)

    return unique_pts.cpu().numpy()

def voxel_downsample_with_indices(points_3d, target_n=10000):
    """
    基于 xyz 的体素下采样（CPU），保留原始点的其他维度（rgb、obj_id）
    返回下采样后的 points_3d
    """
    if points_3d is None or len(points_3d) <= target_n:
        return points_3d
    pts = points_3d[:, :3]
    bbox = pts.max(axis=0) - pts.min(axis=0)
    volume = np.clip(bbox, 1e-6, None).prod()
    voxel_size = (volume / target_n) ** (1.0 / 3.0)
    voxel_idx = np.floor(pts / voxel_size).astype(np.int64)
    P1, P2, P3 = 73856093, 19349663, 83492791
    h = (voxel_idx[:, 0] * P1) ^ (voxel_idx[:, 1] * P2) ^ (voxel_idx[:, 2] * P3)
    _, first_indices = np.unique(h, return_index=True)
    return points_3d[first_indices]


def project_points_gpu(points, R, T, K, device):
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    pts = to_tensor(points, device)
    R_t = to_tensor(R, device)
    T_t = to_tensor(T.reshape(-1), device)
    K_t = to_tensor(K, device)

    pts_cam = (R_t @ pts.T).T + T_t
    valid = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid]

    if pts_cam.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)

    pts_proj = (K_t @ pts_cam.T).T
    pts_2d = pts_proj[:, :2] / pts_proj[:, 2:3]

    # 投影取整（先np.rint再转int，减少边界像素点偏移）
    pts_2d = np.rint(pts_2d.cpu().numpy()).astype(np.int32)
    return pts_2d


def points_to_mask(pts_2d, img_h, img_w):
    """
    最简单的逻辑：将点云直接映射到2D上，然后进行轻量级膨胀。
    相比凸包逻辑，这能更好地保持物体的真实形状（尤其是凹形状）。
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    if len(pts_2d) == 0:
        return mask

    # 1. 过滤越界点并直接映射到 Mask
    pts_2d = np.rint(pts_2d).astype(np.int32)
    in_frame = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < img_w) & \
               (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < img_h)
    pts_valid = pts_2d[in_frame]
    
    if len(pts_valid) > 0:
        mask[pts_valid[:, 1], pts_valid[:, 0]] = 1

    # 2. 轻量级膨胀操作
    # kernel = np.ones(DILATION_KERNEL, np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations=DILATION_ITERATIONS)

    return mask


# ============================================================================
# 辅助函数
# ============================================================================

def preprocess_depth(depth_image, min_depth=0.01, max_depth=10.0):
    """预处理深度图：过滤NaN/Inf，限定有效深度范围，保持(H, W)形状不变"""
    valid = np.isfinite(depth_image) & (depth_image > min_depth) & (depth_image < max_depth)
    return np.where(valid, depth_image, 0.0)


def median_filter_depth(depth_image, ksize=3):
    """中值滤波去除深度图飞点（flying pixels）。
    飞点通常只有 1-2 像素宽，3×3 中值即可有效去除。
    无效像素（depth=0）不参与滤波：只在有效像素邻域内取中值。
    """
    valid = depth_image > 0
    if not valid.any():
        return depth_image

    depth_f = depth_image.astype(np.float32)
    filtered = cv2.medianBlur(depth_f, ksize)

    # 中值滤波可能把有效像素变成 0（如果邻域多数是 0），恢复原始值
    became_zero = valid & (filtered == 0)
    filtered[became_zero] = depth_f[became_zero]

    # 中值滤波可能把无效像素变成非零，置回 0
    filtered[~valid] = 0.0
    return filtered


def depth_edge_mask(depth_image, relative_threshold=0.05, dilate_pixels=1):
    """检测深度跳变边缘，返回应被排除的像素 mask（True = 边缘像素，应排除）。
    使用相对梯度（grad/depth）判别真实深度跳变 vs 倾斜表面的透视梯度：
    - 真实边缘：跳变 1m/1m = 100%
    - 倾斜地板：2cm/1m = 2%（不被误判）
    relative_threshold: 默认 0.05（5%），>5% 的相对梯度视为跳变边缘。
    dilate_pixels: 膨胀边缘若干像素，确保覆盖边缘周围的飞点。
    """
    valid = depth_image > 0
    if not valid.any():
        return np.zeros(depth_image.shape, dtype=bool)

    depth_f = depth_image.astype(np.float32)
    grad_x = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 相对梯度：grad / depth，避免误判倾斜表面（地板/墙）
    relative_grad = np.zeros_like(grad_mag)
    relative_grad[valid] = grad_mag[valid] / np.maximum(depth_f[valid], 0.1)

    edge = (relative_grad > relative_threshold) & valid

    if dilate_pixels > 0:
        kernel = np.ones((2 * dilate_pixels + 1, 2 * dilate_pixels + 1), np.uint8)
        edge = cv2.dilate(edge.astype(np.uint8), kernel).astype(bool)

    return edge


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def get_mask_center(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.array([0, 0])
    return np.array([xs.mean(), ys.mean()])


def cosine_similarity(feat1, feat2):
    feat1 = feat1.flatten()
    feat2 = feat2.flatten()
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8)


def backproject_depth(depth_image, mask, intrinsics, w2c):
    """从深度图反投影生成3D点"""
    H, W = depth_image.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 3))

    depths = depth_image[ys, xs]
    valid_mask = depths > 0

    xs = xs[valid_mask]
    ys = ys[valid_mask]
    depths = depths[valid_mask]

    if len(xs) == 0:
        return np.zeros((0, 3))

    x_cam = (xs - cx) * depths / fx
    y_cam = (ys - cy) * depths / fy
    z_cam = depths

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    c2w = np.linalg.inv(w2c)
    ones = np.ones((len(points_cam), 1))
    points_homo = np.concatenate([points_cam, ones], axis=1)
    points_world = (c2w @ points_homo.T).T

    return points_world[:, :3]


def extract_rgb(rgb_image, mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 3))
    colors = rgb_image[ys, xs] / 255.0
    return colors


def backproject_depth_with_mask(depth_image, mask, intrinsics, w2c):
    """从深度图反投影生成3D点，返回3D点和对应的有效mask索引"""
    H, W = depth_image.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 3)), np.zeros((len(xs),), dtype=bool)

    depths = depth_image[ys, xs]
    valid_mask = np.isfinite(depths) & (depths>0) # 过滤无效深度值包括相机背后点/NAN/INF

    xs_valid = xs[valid_mask]
    ys_valid = ys[valid_mask]
    depths_valid = depths[valid_mask]

    if len(xs_valid) == 0:
        return np.zeros((0, 3)), valid_mask

    x_cam = (xs_valid - cx) * depths_valid / fx
    y_cam = (ys_valid - cy) * depths_valid / fy
    z_cam = depths_valid

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    c2w = np.linalg.inv(w2c)
    ones = np.ones((len(points_cam), 1))
    points_homo = np.concatenate([points_cam, ones], axis=1)
    points_world = (c2w @ points_homo.T).T

    return points_world[:, :3], valid_mask


def extract_rgb_by_mask(rgb_image, mask, valid_mask):
    """根据有效mask索引提取RGB"""
    ys, xs = np.where(mask)
    colors = rgb_image[ys, xs][valid_mask] / 255.0
    return colors


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["mydata"]:
        return MydataDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {config_dict['dataset_name']}")


# ============================================================================
# Step 2: 批量投影
# ============================================================================

def batch_project_objects(objects, intrinsics, w2c, img_h, img_w):
    """
    将每个object的点云投影到当前视角，得到每个物体的mask
    所有 object 的点拼接后一次 GPU 投影，避免逐 object 的 kernel launch 开销

    Args:
        objects: List[ObjectMemory]
        intrinsics: (3, 3)
        w2c: (4, 4) world to camera
        img_h, img_w: image size

    Returns:
        projected_masks: dict {obj_id: mask}
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    R = w2c[:3, :3]
    T = w2c[:3, 3:4]
    projected_masks = {}

    total_start = time.time()

    # 收集有点的 object 及其点数
    valid_objects = [obj for obj in objects if len(obj.points_3d) > 0]
    if len(valid_objects) == 0:
        print(f"  [BatchProject] total={time.time()-total_start:.4f}s, no objects with points")
        return projected_masks

    # 拼接所有 object 的点 + 记录边界
    all_points = np.vstack([obj.points_3d[:, :3] for obj in valid_objects])
    boundaries = np.cumsum([0] + [len(obj.points_3d) for obj in valid_objects])

    # 一次 GPU 调用：视锥剔除 + 投影
    t1 = time.time()
    pts_cam_all, pts_2d_all, valid_mask_all = frustum_culling_gpu(
        all_points, R, T, intrinsics, img_w, img_h, device
    )
    time_frustum = time.time() - t1

    # 按边界拆分，为每个 object 生成 mask
    t2 = time.time()
    for i, obj in enumerate(valid_objects):
        # valid_mask_all 是相对于 all_points 的布尔数组
        obj_valid = valid_mask_all[boundaries[i]:boundaries[i+1]]
        if obj_valid.sum() < MIN_VISIBLE_POINTS:
            continue
        # pts_2d_all 只包含 in-frustum 的点，按 valid_mask 偏移量切片
        cam_offset = valid_mask_all[:boundaries[i]].sum()
        cam_count = obj_valid.sum()
        obj_pts_2d = pts_2d_all[cam_offset:cam_offset + cam_count]
        mask = points_to_mask(obj_pts_2d, img_h, img_w)
        projected_masks[obj.id] = mask
    time_mask = time.time() - t2

    total_time = time.time() - total_start
    print(f"  [BatchProject] total={total_time:.4f}s, frustum={time_frustum:.4f}s, "
          f"mask={time_mask:.4f}s, objects={len(valid_objects)}")

    return projected_masks


# ============================================================================
# Step 3: 多模态匹配
# ============================================================================

def compute_similarity_matrix(detections, projected_masks, objects):
    """
    计算detections和projected_objects之间的相似度矩阵

    similarity = WEIGHT_IOU * IoU + WEIGHT_CENTER * center_sim + WEIGHT_CLIP * clip_sim

    优化：先用 bbox 快速排斥，跳过不可能有 IoU 的物体对。
    """
    num_det = len(detections)
    num_proj = len(projected_masks)

    if num_det == 0 or num_proj == 0:
        return np.zeros((num_det, num_proj))

    similarity = np.zeros((num_det, num_proj))
    obj_ids = list(projected_masks.keys())
    obj_by_id = {o.id: o for o in objects}

    # 预计算每个 detection mask 的 bbox (x_min, y_min, x_max, y_max)
    det_bbox_cache = {}
    for i, det in enumerate(detections):
        ys, xs = np.where(det['mask'] > 0)
        if len(xs) > 0:
            det_bbox_cache[i] = (xs.min(), ys.min(), xs.max(), ys.max())

    # 预计算每个 projected mask 的 bbox
    proj_bbox_cache = {}
    for obj_id, mask in projected_masks.items():
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            proj_bbox_cache[obj_id] = (xs.min(), ys.min(), xs.max(), ys.max())

    for i, det in enumerate(detections):
        det_center = det['center_3d']
        for j, obj_id in enumerate(obj_ids):
            obj = obj_by_id[obj_id]
            obj_center = obj.center_3d
            center_dist = np.linalg.norm(det_center - obj_center)
            center_sim = np.exp(-center_dist / max(1e-6, 0.3))
            clip_sim = cosine_similarity(det['clip_feature'], obj.clip_feature)

            # bbox 快速排斥：如果没有交集则 IoU=0，跳过全图 IoU 计算
            if i in det_bbox_cache and obj_id in proj_bbox_cache:
                dx1, dy1, dx2, dy2 = det_bbox_cache[i]
                px1, py1, px2, py2 = proj_bbox_cache[obj_id]
                if dx2 < px1 or dx1 > px2 or dy2 < py1 or dy1 > py2:
                    similarity[i, j] = WEIGHT_CENTER * center_sim + WEIGHT_CLIP * clip_sim
                    continue

            # bbox 有交集，计算完整 IoU
            iou = compute_iou(det['mask'], projected_masks[obj_id])
            similarity[i, j] = WEIGHT_IOU * iou + WEIGHT_CENTER * center_sim + WEIGHT_CLIP * clip_sim

    return similarity


def match_detections_to_objects(detections, projected_masks, objects):
    """
    最简单的匹配逻辑：遍历每个 detection，计算和 projected_mask 之间的相似度。
    如果最大相似度 < 阈值，则加入 unmatched_detections；
    如果最大相似度 >= 阈值，则将该对 (det_idx, obj_id) 加入 matched_pairs。
    """
    similarity = compute_similarity_matrix(detections, projected_masks, objects)

    matched_pairs = []
    unmatched_det = []
    unmatched_det_indices = []
    unmatched_obj = []
    obj_ids = list(projected_masks.keys())
    obj_by_id = {o.id: o for o in objects}

    print(f"\n  [MATCH DEBUG] Simple matching process (Detection-Centric):")
    for i, det in enumerate(detections):
        if len(obj_ids) == 0:
            unmatched_det.append(det)
            unmatched_det_indices.append(i)
            print(f"    det{i}: unmatched (no projected objects)")
            continue
            
        # 找到相似度最高的物体索引 j
        j = np.argmax(similarity[i])
        max_sim = similarity[i, j]
        obj_id = obj_ids[j]
        
        if max_sim >= MATCH_THRESHOLD:
            matched_pairs.append((i, obj_id))
            print(f"    det{i} vs obj{obj_id}: sim={max_sim:.4f} MATCHED")
        else:
            unmatched_det.append(det)
            unmatched_det_indices.append(i)
            print(f"    det{i}: unmatched (max_sim={max_sim:.4f} < {MATCH_THRESHOLD})")

    # 为了兼容后续逻辑，计算未匹配的物体 ID
    matched_obj_ids = set(pair[1] for pair in matched_pairs)
    unmatched_obj = [obj_by_id[oid] for oid in obj_ids if oid not in matched_obj_ids]

    return {
        'matched_pairs': matched_pairs,
        'unmatched_detections': unmatched_det,
        'unmatched_det_indices': unmatched_det_indices,
        'unmatched_objects': unmatched_obj,
        'similarity': similarity,
        'obj_ids': obj_ids
    }


def voxel_downsample_points(points, voxel_size=0.01):
    """
    使用纯 Numpy 进行超快速体素下采样。
    将 3D 空间划分为网格，每个网格（体素）内只保留第一个点。
    
    :param points: np.ndarray, shape (N, 7) [x, y, z, r, g, b, obj_idx]
    :param voxel_size: 体素大小，单位米（如 0.01 表示 1cm³ 的体素）
    :return: 下采样后的点云 np.ndarray, shape (M, 7)
    """
    if len(points) == 0:
        return points

    # 1. 提取 xyz 坐标
    xyz = points[:, :3]

    # 2. 计算每个点所属的体素索引 (x, y, z)
    # 将坐标除以 voxel_size 并向下取整
    voxel_coords = np.floor(xyz / voxel_size).astype(np.int32)

    offset = 100000
    voxel_hash = (voxel_coords[:, 0].astype(np.int64) * offset * offset +
                  voxel_coords[:, 1].astype(np.int64) * offset +
                  voxel_coords[:, 2].astype(np.int64))

    # 4. 找到所有唯一的体素，并返回它们在原数组中的第一次出现的索引 (return_index=True)
    _, unique_indices = np.unique(voxel_hash, return_index=True)

    # 5. 直接用这些唯一索引去切片原始的 (N, 7) 数组
    down_points = points[unique_indices]

    return down_points


# ============================================================================
# Step 1: 全局点云反投影模块
# ============================================================================

def backprojection_module(depth_image, rgb_image, intrinsics, w2c,
                          global_cloud, current_count, voxel_hash_to_idx,
                          voxel_size=0.02, stride=1):
    """
    全局点云反投影模块（Step 1）。
    深度预处理 → 反投影 → 帧内去重 → 全局去重(skip)+新增。
    已存在的体素跳过（保留首次观测的 rgb），不替换。
    所有 obj_idx = 0（全局点云仅用于渲染，不参与物体记忆列表构建）。

    Args:
        depth_image: (H, W) float32
        rgb_image: (H, W, 3) uint8
        intrinsics: (3, 3)
        w2c: (4, 4)
        global_cloud: (MAX_POINTS, 7) 预分配 buffer
        current_count: int, 已使用行数
        voxel_hash_to_idx: dict[int, int], 体素哈希→buffer行号
        voxel_size: 体素大小（米）
        stride: 反投影步长

    Returns:
        current_count: int
    """
    H, W = depth_image.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    offset = 100000

    # 1a: 深度预处理
    valid_depth = np.isfinite(depth_image) & (depth_image > 0.01) & (depth_image < 10.0)

    # 1a-2: 排除深度跳变边缘（多视角边缘伪影的主要来源）
    edge_mask = depth_edge_mask(depth_image, relative_threshold=0.1, dilate_pixels=1)
    valid_depth = valid_depth & ~edge_mask

    # 1b: Stride 反投影
    ys_stride, xs_stride = np.mgrid[0:H:stride, 0:W:stride]
    valid_s = valid_depth[ys_stride, xs_stride]
    ys_valid = ys_stride[valid_s]
    xs_valid = xs_stride[valid_s]
    depths_valid = depth_image[ys_valid, xs_valid]

    if len(depths_valid) == 0:
        return current_count

    x_cam = (xs_valid - cx) * depths_valid / fx
    y_cam = (ys_valid - cy) * depths_valid / fy
    z_cam = depths_valid
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    c2w = np.linalg.inv(w2c)
    ones = np.ones((len(points_cam), 1))
    points_homo = np.concatenate([points_cam, ones], axis=1)
    points_world = (c2w @ points_homo.T).T[:, :3]

    colors = rgb_image[ys_valid, xs_valid].astype(np.float32) / 255.0
    local_points = np.zeros((len(points_world), 7), dtype=np.float32)
    local_points[:, :3] = points_world
    local_points[:, 3:6] = colors

    # 1c: 帧内体素去重
    voxel_coords = np.floor(local_points[:, :3] / voxel_size).astype(np.int32)
    voxel_hash = (voxel_coords[:, 0].astype(np.int64) * offset * offset +
                  voxel_coords[:, 1].astype(np.int64) * offset +
                  voxel_coords[:, 2].astype(np.int64))
    _, first_in_frame, _ = np.unique(voxel_hash, return_index=True, return_inverse=True)
    first_hash = voxel_hash[first_in_frame]

    # 1d: 全局去重 + 新增（skip 已存在的体素，保留首次 rgb）
    first_hash_list = first_hash.tolist()
    new_mask = np.array([h not in voxel_hash_to_idx for h in first_hash_list], dtype=bool)

    K = int(new_mask.sum())
    if K > 0:
        new_local_idx = first_in_frame[new_mask]
        new_hashes = first_hash[new_mask]
        global_cloud[current_count:current_count + K] = local_points[new_local_idx]
        new_global_idx = np.arange(current_count, current_count + K, dtype=np.int32)
        for i, h in enumerate(new_hashes.tolist()):
            voxel_hash_to_idx[h] = int(new_global_idx[i])
        current_count += K

    return current_count


# ============================================================================
# 全局点云后处理
# ============================================================================

def postprocess_global_cloud(global_cloud, current_count, voxel_hash_to_idx,
                              voxel_size=0.02, min_neighbors=3):
    """单轮 26-邻域清杂 — 清除残留的零星孤立点。"""
    offset = 100000
    N = current_count
    print(f"\n[PostProcess] Starting: {N} points")

    all_hashes = np.array(list(voxel_hash_to_idx.keys()), dtype=np.int64)
    sorted_hashes = np.sort(all_hashes)

    voxel_coords = np.floor(global_cloud[:N, :3] / voxel_size).astype(np.int32)

    offsets_26 = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]
                  if not (dx == 0 and dy == 0 and dz == 0)]

    neighbor_count = np.zeros(N, dtype=np.int32)
    for dx, dy, dz in offsets_26:
        nc = voxel_coords + np.array([dx, dy, dz], dtype=np.int32)
        nh = (nc[:, 0].astype(np.int64) * offset * offset +
              nc[:, 1].astype(np.int64) * offset +
              nc[:, 2].astype(np.int64))

        pos = np.searchsorted(sorted_hashes, nh)
        pos_clipped = np.clip(pos, 0, len(sorted_hashes) - 1)
        exact_match = (pos < len(sorted_hashes)) & (sorted_hashes[pos_clipped] == nh)
        neighbor_count += exact_match.astype(np.int32)

    isolated = neighbor_count < min_neighbors
    remove_count = int(isolated.sum())

    if remove_count > 0:
        keep_mask = ~isolated
        kept = global_cloud[:N][keep_mask]
        N = len(kept)
        global_cloud[:N] = kept

        # 重建 voxel_hash_to_idx
        new_vc = np.floor(global_cloud[:N, :3] / voxel_size).astype(np.int32)
        new_hashes = (new_vc[:, 0].astype(np.int64) * offset * offset +
                      new_vc[:, 1].astype(np.int64) * offset +
                      new_vc[:, 2].astype(np.int64))
        voxel_hash_to_idx.clear()
        for i, h in enumerate(new_hashes.tolist()):
            voxel_hash_to_idx[h] = i
        print(f"  Removed {remove_count} isolated points, {N} remaining")

    print(f"[PostProcess] Done: {N} points (removed {current_count - N})")
    return N


# ============================================================================
# Step 4: 融合更新
# ============================================================================

def fuse_update_matched(objects, detections, match_result):
    """
        匹配成功的物体：更新center_3d + clip_feature + mask
        融合后立即做 2cm 体素去重，保证 obj.points_3d 与 global_cloud 同一标准
    """
    for det_idx, obj_id in match_result['matched_pairs']:
        det = detections[det_idx]
        obj = next((o for o in objects if o.id == obj_id), None)
        if obj is None:
            continue

        # 点云融合：将当前检测的3D点加入物体点云（统一设置为该物体ID）
        det['points_3d'][:, 6] = obj.id
        obj.points_3d = np.vstack([obj.points_3d, det['points_3d']])

        # object点数控制：超过阈值时进行体素下采样 (1cm³)
        if len(obj.points_3d) > MAX_POINTS_PER_OBJECT:
            original_len = len(obj.points_3d)
            obj.points_3d = voxel_downsample_points(obj.points_3d, voxel_size=0.01)
            print(f"  [DOWNSAMPLE] Obj[{obj.id}]: {original_len} -> {len(obj.points_3d)} points (voxel_size=1cm)")

        # 重新计算中心
        new_center = np.median(obj.points_3d[:, :3], axis=0)
        obj.center_3d = new_center
        # 更新clip_feature（EMA，alpha=0.3 保证新观测始终有足够权重）
        alpha = 0.3
        obj.clip_feature = (1 - alpha) * obj.clip_feature + alpha * det['clip_feature']
        obj.nums += 1  # 保留观测计数，用于调试
        # 更新物体最佳crop
        if det['view_score'] > obj.view_score:
            obj.view_score = det['view_score']
            obj.image_crop = det['image_crop']
            obj.masked_crop = det['masked_crop']
        elif obj.masked_crop is None and det.get('masked_crop') is not None:
            # 补充旧对象缺失的 masked_crop
            obj.masked_crop = det['masked_crop']
        # 更新其他属性
        obj.mask = det['mask']
        obj.class_name = det.get('class_name', 'unknown')


def add_new_objects(objects, detections, match_result):
    """未匹配的检测：新增为object，添加点云"""
    similarity = match_result.get('similarity', None)
    obj_ids = match_result.get('obj_ids', [])
    unmatched_det_indices = match_result.get('unmatched_det_indices', [])
    objects_to_add = []

    for i, det in enumerate(match_result['unmatched_detections']):
        next_obj_id = max([o.id for o in objects], default=0) + 1
        sim_row = unmatched_det_indices[i] if i < len(unmatched_det_indices) else det['idx']

        if similarity is not None and len(obj_ids) > 0:
            best_sim = np.max(similarity[sim_row])
            reason = "below threshold" if best_sim < MATCH_THRESHOLD else "best obj stolen"
            print(f"  [ADD_NEW] Det[{det['idx']}] -> NewID:{next_obj_id}, BestSim={best_sim:.4f} ({reason})")

        if len(det['points_3d'][:, :3]) < 50:
            continue

        new_obj = ObjectMemory(
            obj_id=next_obj_id,
            class_name=det['class_name'],
            clip_feature=det['clip_feature'],
            center_3d=det['center_3d'],
            mask=det['mask'],
            points_3d=det['points_3d'],
            image_crop=det['image_crop'],
            masked_crop=det['masked_crop'],
            view_score=det['view_score'],
        )
        new_obj.points_3d[:, 6] = next_obj_id
        objects.append(new_obj)
        objects_to_add.append(new_obj)

    return objects_to_add


# ============================================================================
# Step 4.5: 近距离物体融合
# ============================================================================

def merge_nearby_objects(objects, center_threshold=0.3, clip_threshold=0.7):
    """
    融合中心距离过近且外观相似的物体（解决同一物理物体被拆成多个 ObjectMemory 的问题）。

    条件（同时满足）：
    1. 两物体中心 3D 距离 < center_threshold
    2. CLIP 特征余弦相似度 > clip_threshold（防止把挨着的不同物体错误合并）

    融合策略：保留点数多的物体（更可靠），合并点云和 CLIP 特征。

    Returns:
        list of dict: [{'absorbed_id': int, 'absorber_id': int}, ...] 合并记录
    """
    if len(objects) < 2:
        return []

    merged_ids = set()
    merge_log = []
    for i in range(len(objects)):
        if objects[i].id in merged_ids:
            continue
        for j in range(i + 1, len(objects)):
            if objects[j].id in merged_ids:
                continue

            obj_i, obj_j = objects[i], objects[j]
            center_dist = np.linalg.norm(obj_i.center_3d - obj_j.center_3d)

            if center_dist >= center_threshold:
                continue

            clip_sim = cosine_similarity(obj_i.clip_feature, obj_j.clip_feature)

            if clip_sim < clip_threshold:
                continue

            # 确定主物体（点数多）和被吸收物体（点数少）
            if len(obj_i.points_3d) >= len(obj_j.points_3d):
                primary, absorbed = obj_i, obj_j
            else:
                primary, absorbed = obj_j, obj_i

            # 合并点云
            absorbed.points_3d[:, 6] = primary.id
            primary.points_3d = np.vstack([primary.points_3d, absorbed.points_3d])

            # 重新计算中心
            primary.center_3d = np.median(primary.points_3d[:, :3], axis=0)

            # CLIP 特征按点数比例加权（点数多的一方更可靠）
            n_primary = len(primary.points_3d)
            n_absorbed = len(absorbed.points_3d)
            w = n_primary / (n_primary + n_absorbed)
            primary.clip_feature = w * primary.clip_feature + (1 - w) * absorbed.clip_feature
            primary.nums += absorbed.nums  # 累计观测计数，用于调试

            # 保留最佳 view
            if absorbed.view_score > primary.view_score:
                primary.view_score = absorbed.view_score
                primary.image_crop = absorbed.image_crop
                primary.masked_crop = absorbed.masked_crop

            merged_ids.add(absorbed.id)
            merge_log.append({'absorbed_id': absorbed.id, 'absorber_id': primary.id})
            print(f"  [MERGE] obj{absorbed.id} ({absorbed.class_name}, {len(absorbed.points_3d)}pts) "
                  f"-> obj{primary.id} ({primary.class_name}, {len(primary.points_3d)}pts): "
                  f"dist={center_dist:.3f}m, clip_sim={clip_sim:.4f}")

    if merged_ids:
        objects[:] = [o for o in objects if o.id not in merged_ids]

    return merge_log


# ============================================================================
# Step 5: 删除过时物体 (DovSG风格)
# ============================================================================

def remove_stale_objects(objects, match_result, rgb_image, depth_image,
                         intrinsics, w2c, depth_threshold=DEPTH_ERROR_THRESHOLD, color_threshold=COLOR_ERROR_THRESHOLD,
                         disappear_ratio_thr=0.7, frustum_ratio_thr=0.7, visibility_ratio_thr=0.5):
    '''
    参数说明：
    - depth_threshold: 深度误差阈值（米）
    - color_threshold: 颜色误差阈值（RGB 0~255 的欧氏距离）
    - delete_rate: 点级"软删除"比例超过此阈值时执行物体"硬删除"

    改进版处理流程概要（保留点级评估并改进）：
    1) 仅针对"当前可见但未匹配"的物体进行检查（unmatched_objects）
    2) 批量视锥剔除 + 投影（一次 GPU 调用），按像素聚合选择"每像素最近点"（近似 z-buffer）
    3) 对未被遮挡的投影点，计算实际深度和颜色的误差
    4) 只有当深度误差和颜色误差同时过大时，才认为该点属于"冲突"（消失或被改变）
    5) 统计冲突点占未被遮挡点的比例，如果冲突比例（软删除比例）大于阈值，则整个物体删除
    TODO:   1、引入多帧一致性缓冲（滑窗质量/重合率），仅在连续 K 帧失败时删除；并将一致性度量扩展为"重合率"以贴近 DovSG 的 indexes 重叠率。
            2、对物体点云进行聚类，剔除离中心远的点，提高鲁棒性。
            3、换一个思路，只考虑记忆中离相机视野最近的objects（给其他物体挡住的object不再考虑），如果该object的点级软删除比例超过阈值，执行物体硬删
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    R = w2c[:3, :3]
    T = w2c[:3, 3:4]
    img_h, img_w = rgb_image.shape[:2]
    H, W = depth_image.shape

    objects_to_remove = []

    # --- Phase 1: 分离点数过少的物体（直接硬删除）和需要投影检测的物体 ---
    unmatched_objs = match_result['unmatched_objects']
    valid_unmatched = []
    for obj in unmatched_objs:
        if len(obj.points_3d) < 20:
            print(f"  [DELETE] Obj[{obj.id}] ({obj.class_name}): del_obj_points_nums={len(obj.points_3d)} < 20 -> DELETE")
            objects_to_remove.append(obj)
        else:
            valid_unmatched.append(obj)

    if len(valid_unmatched) == 0:
        # 批量删除
        for obj in objects_to_remove:
            if obj in objects:
                objects.remove(obj)
        return objects_to_remove

    # --- Phase 2: 批量 GPU 投影（所有未匹配物体一次调用） ---
    all_pts = np.vstack([obj.points_3d[:, :3] for obj in valid_unmatched])
    boundaries = np.cumsum([0] + [len(obj.points_3d) for obj in valid_unmatched])

    pts_cam_all, pts_2d_all, valid_mask_all = frustum_culling_gpu(
        all_pts, R, T, intrinsics, img_w, img_h, device
    )

    # --- Phase 3: 按边界拆分回各物体，逐物体做 z-buffer + 冲突检测 ---
    for i, obj in enumerate(valid_unmatched):
        obj_points = obj.points_3d

        obj_valid = valid_mask_all[boundaries[i]:boundaries[i+1]]
        if obj_valid.sum() < MIN_VISIBLE_POINTS:
            continue

        # 从批量结果中按 valid mask 偏移量拆分出当前物体的相机坐标和像素坐标
        cam_offset = valid_mask_all[:boundaries[i]].sum()
        cam_count = obj_valid.sum()
        pts_cam = pts_cam_all[cam_offset:cam_offset + cam_count]
        pts_2d = pts_2d_all[cam_offset:cam_offset + cam_count]

        # 像素坐标限制
        pts_2d = np.clip(np.rint(pts_2d).astype(np.int32),
                         [0, 0], [img_w-1, img_h-1])
        pts_depth = pts_cam[:, 2]

        # 按照 z-buffer 提取每像素最前面的点
        u = pts_2d[:, 0]
        v = pts_2d[:, 1]
        pixel_ids = v * W + u
        order = np.lexsort((pts_depth, pixel_ids))
        sorted_ids = pixel_ids[order]
        first_idx = np.unique(sorted_ids, return_index=True)[1]
        nearest_idx = order[first_idx]

        # 映射到原始物体点索引
        culled_idx = np.where(obj_valid)[0]
        obj_idx_near = culled_idx[nearest_idx]

        # 计算视锥占比 (Frustum Ratio)
        # 有多少点进入了理论相机的 FOV 内（即使被遮挡也算）
        total_points_num = len(obj_points)
        frustum_points_num = obj_valid.sum()
        frustum_ratio = frustum_points_num / max(1, total_points_num)

        # 提取这些最近点对应的实际深度和实际颜色
        actual_depth_near = depth_image[v[nearest_idx], u[nearest_idx]]
        depth_valid = np.isfinite(actual_depth_near) & (actual_depth_near > 0) # 过滤无效深度值包括相机背后点/负无穷NAN/正无穷INF
        if depth_valid.sum() == 0:
            continue

        # 拦截逻辑：投影到当前视角并具有有效深度的像素点过少（少于100个像素），
        # 往往是只扫到了一个极小的边缘或者全是噪点。
        # 这种情况下直接跳过，既不做物体级删除，也不做点云软删除。
        if depth_valid.sum() < 100:
            continue

        pts_depth_near = pts_depth[nearest_idx][depth_valid]
        actual_depth_valid = actual_depth_near[depth_valid]
        delta = pts_depth_near - actual_depth_valid

        # 提取颜色
        actual_color_near = rgb_image[v[nearest_idx], u[nearest_idx]].astype(np.float32)[depth_valid]
        proj_color_near = (obj_points[obj_idx_near[depth_valid]][:, 3:6] * 255.0).astype(np.float32)
        color_err_near = np.linalg.norm(proj_color_near - actual_color_near, axis=1)

        # 判断状态
        # 1. 遮挡 (Occluded)：物体点深度 > 实际深度 + 阈值
        occluded = (delta > depth_threshold)
        # 2. 可见点 (Visible)：没有被遮挡的点
        visible = ~occluded
        visible_num = int(visible.sum())

        # 可见像素点太低，认为是异常，直接跳过
        if visible_num < 100:
            continue

        # 计算可见度占比 (Visibility Ratio)
        # 投影到有效深度的像素中，有多少是没有被前方物体遮挡的
        projected_points_num = len(delta)
        visibility_ratio = visible_num / max(1, projected_points_num)

        # 在可见点（未被遮挡的点）中，寻找"冲突/错误"点
        # 冲突条件：深度严重不对（太近），并且颜色也不对
        depth_wrong = (delta[visible] < -depth_threshold)
        color_wrong = color_err_near[visible] > color_threshold

        conflict_points = depth_wrong & color_wrong
        conflict_num = int(conflict_points.sum())
        conflict_ratio = conflict_num / max(1, visible_num)

        # ====== 核心决策逻辑 ======
        should_hard_delete = False

        # 情况 2：视锥占比大于 frustum_ratio_thr (默认0.7，物体绝大部分都进入了理论视野)
        if frustum_ratio >= frustum_ratio_thr:
            # 情况 2.2：可见度占比大于 visibility_ratio_thr (默认0.5，没有被严重遮挡)
            if visibility_ratio >= visibility_ratio_thr:
                # 只有在强观测下，且冲突比例高于阈值时，才进行硬删除
                if conflict_ratio >= disappear_ratio_thr:
                    should_hard_delete = True
                    print(f"  [DELETE] Obj[{obj.id}] ({obj.class_name}): frustum={frustum_ratio:.2f} >= {frustum_ratio_thr}, visibility={visibility_ratio:.2f} >= {visibility_ratio_thr}, conflict={conflict_ratio:.3f} >= {disappear_ratio_thr} -> DELETE")

        if should_hard_delete:
            objects_to_remove.append(obj)
        else:
            # 只有在可见度大于等于软删除可见度阈值时，才进行软删除
            # 如果物体绝大部分被遮挡 (visibility_ratio < visibility_ratio_thr)，那么边缘产生的错误深度极高概率是遮挡物产生的噪点，不应进行软删除。
            if visibility_ratio >= visibility_ratio_thr:
                # 情况 1.1, 1.2, 2.1, 以及 2.2(未达删除阈值) -> 只做点级软删除
                # 【重要修改】：软删除的条件从 (depth_wrong & color_wrong) 改为仅判断 depth_wrong
                # 只要点云跑到实际物体表面前面去了（挡住了视线，形成了噪点/重影），不管颜色对不对，一律剔除！
                soft_delete_points = depth_wrong
                soft_delete_num = int(soft_delete_points.sum())

                if soft_delete_num > 0:
                    conflict_mask_full = np.zeros(len(delta), dtype=bool)
                    conflict_mask_full[visible] = soft_delete_points
                    remove_idx_near = obj_idx_near[depth_valid][conflict_mask_full]

                    # numpy boolean mask 过滤（替代 set-based 方式，更高效）
                    if len(remove_idx_near) > 0:
                        keep_mask = np.ones(len(obj_points), dtype=bool)
                        keep_mask[remove_idx_near] = False
                        obj.points_3d = obj.points_3d[keep_mask]
                        if len(obj.points_3d) >= 1:
                            obj.center_3d = np.median(obj.points_3d[:, :3], axis=0)

    # 批量删除
    for obj in objects_to_remove:
        if obj in objects:
            objects.remove(obj)

    return objects_to_remove


def estimate_center_3d_fast(mask, depth_image, intrinsics, w2c):
    """快速估计3D中心"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    center_y, center_x = int(ys.mean()), int(xs.mean())
    center_depth = depth_image[center_y, center_x]

    if center_depth > 0:
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        x_cam = (center_x - cx) * center_depth / fx
        y_cam = (center_y - cy) * center_depth / fy
        point_cam = np.array([x_cam, y_cam, center_depth, 1.0])
        c2w = np.linalg.inv(w2c)
        point_world = (c2w @ point_cam)[:3]
        return point_world
    return None


# ============================================================================
# Step 6: 可视化
# ============================================================================

def visualize_frame(rgb_image, detections, objects, match_result,
                    projected_masks, objects_to_add, deleted_objects,
                    updated_projected_masks, output_dir, frame_id):
    """
    8列可视化，拼接成2x4布局：

    第一行：
    1. Detection bbox - 检测框 + class_name + 置信度
    2. SAM mask - 每个检测框对应的分割掩码
    3. Origin objects mask - 原始物体投影mask
    4. Matched mask - 匹配上的物体mask

    第二行：
    5. Add objects mask - 新添加的物体mask（使用对应的object_id颜色）
    6. Unmatched objects mask - 未匹配的物体mask（可能是待删除的）
    7. Delete objects mask - 确定删除的物体mask（红色）
    8. Updated objects mask - 更新后的物体投影mask

    Args:
        rgb_image: (H, W, 3) RGB图像
        detections: 检测列表
        objects: 物体记忆列表
        match_result: 匹配结果
        projected_masks: 原始物体投影mask字典 {obj_id: mask}
        objects_to_add: 新添加的物体列表
        deleted_objects: 被删除的物体列表
        updated_projected_masks: 更新后的物体投影mask字典
        output_dir: 输出目录
        frame_id: 帧ID
    """
    detection_dir = os.path.join(output_dir, "detection")
    os.makedirs(detection_dir, exist_ok=True)

    def add_title(img, title):
        """在图片顶部添加标题栏"""
        h, w = img.shape[:2]
        title_bg = np.zeros((40, w, 3), dtype=np.uint8)
        (text_w, text_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_x = (w - text_w) // 2
        text_y = 20 + text_h // 2
        cv2.putText(title_bg, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return np.vstack((title_bg, img))

    def put_text_with_bg(img, text, x, y, color, scale=0.6, thickness=2):
        """在图片上添加带背景的文本"""
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(img, (x, y - text_h - 5), (x + text_w, y), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    colors = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3] * 255

    matched_obj_ids = set(obj_id for _, obj_id in match_result['matched_pairs'])
    all_obj_ids = set(projected_masks.keys())
    unmatched_obj_ids = all_obj_ids - matched_obj_ids

    # 第一列：检测框 + class_name + 置信度
    det_bbox_viz = rgb_image.copy()
    det_bbox_viz = cv2.cvtColor(det_bbox_viz, cv2.COLOR_RGB2BGR)
    for det_idx, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        class_name = det.get('class_name', 'unknown')
        conf = det.get('confidence', 0.0)
        # 为每个检测分配不同颜色
        color = colors[det_idx % 20].astype(np.uint8).tolist()
        cv2.rectangle(det_bbox_viz, (x1, y1), (x2, y2), color, 2)
        text = f"D{det_idx}:{class_name}:{conf:.2f}"
        put_text_with_bg(det_bbox_viz, text, x1, y1 - 5, color)
    det_bbox_viz = add_title(det_bbox_viz, f"1.Detections({len(detections)})")

    # 第二列：SAM mask - 每个检测框的分割掩码
    sam_mask_viz = rgb_image.copy()
    sam_mask_viz = cv2.cvtColor(sam_mask_viz, cv2.COLOR_RGB2BGR)
    for det_idx, det in enumerate(detections):
        color = colors[det_idx % 20].astype(np.uint8).tolist()
        mask_bool = det['mask'] > 0
        sam_mask_viz[mask_bool] = (np.array(color) * 0.5 + sam_mask_viz[mask_bool] * 0.5).astype(np.uint8)
    sam_mask_viz = add_title(sam_mask_viz, f"2.SAM({len(detections)})")

    # 第三列：原始物体投影mask
    origin_viz = rgb_image.copy()
    origin_viz = cv2.cvtColor(origin_viz, cv2.COLOR_RGB2BGR)
    for obj_id, proj_mask in projected_masks.items():
        color = colors[int(obj_id) % 20].astype(np.uint8).tolist()
        mask_bool = proj_mask > 0
        origin_viz[mask_bool] = (np.array(color) * 0.5 + origin_viz[mask_bool] * 0.5).astype(np.uint8)
        obj_center = np.where(proj_mask > 0)
        if len(obj_center[0]) > 0:
            cx, cy = int(obj_center[1].mean()), int(obj_center[0].mean())
            text = f"ID:{obj_id}"
            put_text_with_bg(origin_viz, text, cx, cy, color, scale=0.5)
    origin_viz = add_title(origin_viz, f"3.OriginObjs({len(projected_masks)})")

    # 第四列：匹配上的物体mask
    matched_viz = rgb_image.copy()
    matched_viz = cv2.cvtColor(matched_viz, cv2.COLOR_RGB2BGR)
    for det_idx, obj_id in match_result['matched_pairs']:
        if obj_id in projected_masks:
            proj_mask = projected_masks[obj_id]
            color = colors[int(obj_id) % 20].astype(np.uint8).tolist()
            mask_bool = proj_mask > 0
            matched_viz[mask_bool] = (np.array(color) * 0.5 + matched_viz[mask_bool] * 0.5).astype(np.uint8)
            obj_center = np.where(proj_mask > 0)
            if len(obj_center[0]) > 0:
                cx, cy = int(obj_center[1].mean()), int(obj_center[0].mean())
                text = f"ID:{obj_id}"
                put_text_with_bg(matched_viz, text, cx, cy, color, scale=0.5)
    matched_viz = add_title(matched_viz, f"4.Matched({len(match_result['matched_pairs'])})")

    # 第五列：新增的物体mask（使用对应的object_id颜色）
    add_viz = rgb_image.copy()
    add_viz = cv2.cvtColor(add_viz, cv2.COLOR_RGB2BGR)
    for obj in objects_to_add:
        proj_mask = obj.mask
        color = colors[int(obj.id) % 20].astype(np.uint8).tolist()
        mask_bool = proj_mask > 0
        add_viz[mask_bool] = (np.array(color) * 0.5 + add_viz[mask_bool] * 0.5).astype(np.uint8)
        obj_center = np.where(proj_mask > 0)
        if len(obj_center[0]) > 0:
            cx, cy = int(obj_center[1].mean()), int(obj_center[0].mean())
            text = f"ID:{obj.id}"
            put_text_with_bg(add_viz, text, cx, cy, color, scale=0.5)
    add_viz = add_title(add_viz, f"5.AddObjs({len(objects_to_add)})")

    # 第六列：未匹配的物体mask（可能需要删除的）
    unmatched_viz = rgb_image.copy()
    unmatched_viz = cv2.cvtColor(unmatched_viz, cv2.COLOR_RGB2BGR)
    for obj_id in unmatched_obj_ids:
        if obj_id in projected_masks:
            proj_mask = projected_masks[obj_id]
            color = colors[int(obj_id) % 20].astype(np.uint8).tolist()
            mask_bool = proj_mask > 0
            unmatched_viz[mask_bool] = (np.array(color) * 0.5 + unmatched_viz[mask_bool] * 0.5).astype(np.uint8)
            obj_center = np.where(proj_mask > 0)
            if len(obj_center[0]) > 0:
                cx, cy = int(obj_center[1].mean()), int(obj_center[0].mean())
                text = f"ID:{obj_id}"
                put_text_with_bg(unmatched_viz, text, cx, cy, color, scale=0.5)
    unmatched_viz = add_title(unmatched_viz, f"6.UnmatchedObjs({len(unmatched_obj_ids)})")

    # 第七列：确定删除的物体mask
    delete_viz = rgb_image.copy()
    delete_viz = cv2.cvtColor(delete_viz, cv2.COLOR_RGB2BGR)
    for obj in deleted_objects:
        if obj.id in projected_masks:
            proj_mask = projected_masks[obj.id]
            color = colors[int(obj.id) % 20].astype(np.uint8).tolist()
            mask_bool = proj_mask > 0
            delete_viz[mask_bool] = (np.array(color) * 0.5 + delete_viz[mask_bool] * 0.5).astype(np.uint8)
            obj_center = np.where(proj_mask > 0)
            if len(obj_center[0]) > 0:
                cx, cy = int(obj_center[1].mean()), int(obj_center[0].mean())
                text = f"ID:{obj.id}"
                put_text_with_bg(delete_viz, text, cx, cy, color, scale=0.5)
    delete_viz = add_title(delete_viz, f"7.DeleteObjs({len(deleted_objects)})")

    # 第八列：更新后的物体投影mask
    updated_viz = rgb_image.copy()
    updated_viz = cv2.cvtColor(updated_viz, cv2.COLOR_RGB2BGR)
    for obj_id, proj_mask in updated_projected_masks.items():
        color = colors[int(obj_id) % 20].astype(np.uint8).tolist()
        mask_bool = proj_mask > 0
        updated_viz[mask_bool] = (np.array(color) * 0.5 + updated_viz[mask_bool] * 0.5).astype(np.uint8)
        obj_center = np.where(proj_mask > 0)
        if len(obj_center[0]) > 0:
            cx, cy = int(obj_center[1].mean()), int(obj_center[0].mean())
            text = f"ID:{obj_id}"
            put_text_with_bg(updated_viz, text, cx, cy, color, scale=0.5)
    updated_viz = add_title(updated_viz, f"8.UpdatedObjs({len(updated_projected_masks)})")

    # 2x4布局拼接：先拼第一行，再拼第二行，最后垂直拼接
    row1 = np.hstack([det_bbox_viz, sam_mask_viz, origin_viz, matched_viz])
    row2 = np.hstack([add_viz, unmatched_viz, delete_viz, updated_viz])
    combined = np.vstack([row1, row2])

    save_path = os.path.join(detection_dir, f"{frame_id:04d}.jpg")
    cv2.imwrite(save_path, combined)


# ============================================================================
# 检测模块
# ============================================================================

def calculate_view_score(mask_area, xyxy, img_width, img_height, margin_px=10):
    """
    评估当前帧 crop 质量，用于选择最佳 image_crop。

    标准：物体完整出现在画面内，且占画面比例尽可能大。
    - 完整出现在画面内（bbox 不触边）→ 得分 = mask 面积占比
    - 触边（物体被裁切）→ 得分 = mask 面积占比 × 0.1（大幅降分但仍可比）
    - 触边时得分永远低于任何完整帧，确保完整帧优先
    - 同为触边帧时面积大的仍然优先，避免边缘物体永远选不到 crop

    Args:
        mask_area: mask 像素面积（int）
        xyxy: (x1, y1, x2, y2) 检测框坐标
        img_width, img_height: 图像尺寸
        margin_px: bbox 距画面边缘的最小像素距离（默认 10px）
    """
    x1, y1, x2, y2 = xyxy
    image_area = img_width * img_height
    area_ratio = mask_area / image_area

    # 触边惩罚：分数 ×0.1，保证低于任何完整帧，但触边帧之间仍可比较
    if x1 < margin_px or y1 < margin_px or \
       x2 > img_width - margin_px or y2 > img_height - margin_px:
        return area_ratio * 0.1

    return area_ratio

def filter_point_cloud_fast(points_3d, points_rgb, depth_valid_mask, std_ratio=2.0, min_points=10):
    """
    超快速点云去噪：基于 MAD（中位数绝对偏差）的鲁棒统计滤波。
    剔除距离物体中值中心过远的离群点（如 Mask 溢出导致的背景点）。

    MAD 相比 mean+std 的优势：不受离群点污染，即使 50% 的点是离群点也不影响阈值。

    :param points_3d: (N, 3) 3D 坐标
    :param points_rgb: (N, 3) 颜色
    :param depth_valid_mask: 原始深度有效掩码
    :param std_ratio: MAD 乘数。2.0 表示剔除距离中位数超过 2 倍 MAD 的点
    """
    if len(points_3d) < min_points:
        return points_3d, points_rgb, depth_valid_mask

    # 1. 鲁棒中心（中位数不受离群点影响）
    center = np.median(points_3d, axis=0)

    # 2. 计算所有点到中心的欧氏距离
    distances = np.linalg.norm(points_3d - center, axis=1)

    # 3. 鲁棒统计量：中位数 + MAD
    #    MAD = median(|d_i - median(d)|)
    #    1.4826 × MAD ≈ std（对正态分布的一致性因子）
    median_dist = np.median(distances)
    mad = np.median(np.abs(distances - median_dist))
    robust_threshold = median_dist + std_ratio * 1.4826 * mad + 1e-6

    # 4. 找出核心点
    keep_indices = np.where(distances <= robust_threshold)[0]

    # 5. 保护：过滤后点太少则保留原样
    if len(keep_indices) < min_points:
        return points_3d, points_rgb, depth_valid_mask

    # 6. 更新返回数据
    filtered_points_3d = points_3d[keep_indices]
    filtered_points_rgb = points_rgb[keep_indices]

    # 7. 同步更新深度掩码
    new_depth_valid_mask = np.zeros_like(depth_valid_mask)
    valid_positions = np.where(depth_valid_mask)[0]
    new_depth_valid_mask[valid_positions[keep_indices]] = True

    return filtered_points_3d, filtered_points_rgb, new_depth_valid_mask

def add_lb_points_to_global_cloud(world_points, rgb_image_hwc, global_cloud,
                                   current_count, stride=STRIDE_DOWNSAMPLE):
    """将 lingbot-map 当前帧有效点逐帧 stride 降采样后追加到全局点云。

    Args:
        world_points: (H_lb, W_lb, 3) 世界坐标映射表
        rgb_image_hwc: (H_dgsg, W_dgsg, 3) 原始RGB图像
        global_cloud: (N, 7) 全局点云buffer
        current_count: 当前已使用行数
        stride: 降采样步长

    Returns:
        new_count: 更新后的 current_count
    """
    valid = np.any(world_points != 0, axis=-1)
    if valid.sum() == 0:
        return current_count

    pts_world = world_points[valid]
    h_lb, w_lb = world_points.shape[:2]
    rgb_lb = cv2.resize(rgb_image_hwc, (w_lb, h_lb))
    pts_rgb = (rgb_lb[valid] / 255.0).astype(np.float32)

    # 逐帧 stride 降采样（与先拼后采效果一致，节省 8× 内存）
    pts_world = pts_world[::stride]
    pts_rgb = pts_rgb[::stride]

    n = len(pts_world)
    if n > 0:
        # 防止 buffer 溢出
        if current_count + n > MAX_POINTS:
            print(f"  [WARNING] Global cloud buffer overflow: {current_count} + {n} > {MAX_POINTS}, truncating")
            n = MAX_POINTS - current_count
            if n <= 0:
                return current_count
        global_cloud[current_count:current_count + n, :3] = pts_world[:n]
        global_cloud[current_count:current_count + n, 3:6] = pts_rgb[:n]
        global_cloud[current_count:current_count + n, 6] = 0
        current_count += n
    return current_count


def detect_objects(rgb_image, depth_image, yolo_model, sam_model, clip_model,
                         clip_preprocess, obj_classes, cfg, frame_id, intrinsics, w2c, world_points=None):
    """
        YOLO + SAM + CLIP 检测
        通过 world_points 映射表直接索引获取物体3D点
    """
    det_timing = {'yolo': 0.0, 'sam': 0.0, 'clip': 0.0, 'backproj': 0.0}
    detections = []
    img_height, img_width = rgb_image.shape[:2]
    image_area = img_height * img_width
    # 从cfg中获取配置参数
    mask_conf_threshold = cfg.get('mask_conf_threshold', 0.4)
    max_bbox_area_ratio = cfg.get('max_bbox_area_ratio', 0.6)
    mask_area_threshold = cfg.get('mask_area_threshold', 10)
    box_overlap_threshold = cfg.get('box_overlap_threshold', 0.95)
    bg_classes = obj_classes.get_bg_classes_arr()


    _t0 = time.time()
    yolo_results = yolo_model(rgb_image, conf=0.1, verbose=False)
    det_timing['yolo'] = time.time() - _t0
    if len(yolo_results) == 0 or len(yolo_results[0].boxes) == 0:
        print(f"Frame {frame_id}: No detections")
        return detections, det_timing
    boxes = yolo_results[0].boxes

    # 步骤1: YOLO后过滤（过滤掉低置信度、高检测框比、背景物体）
    valid_boxes = []
    valid_indices = []
    for i, box in enumerate(boxes):
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0])
        class_name = obj_classes.get_classes_arr()[cls_id]
        # 置信度过滤
        if conf < mask_conf_threshold:
            continue
        # 背景物体过滤
        if class_name in bg_classes:
            continue
        # 检测框过大过滤
        bbox_area = (x2 - x1) * (y2 - y1)
        bbox_area_ratio = bbox_area / image_area
        if bbox_area_ratio > max_bbox_area_ratio:
            continue
        valid_boxes.append(box.xyxy[0].cpu().numpy().astype(int))
        valid_indices.append(i)

    if len(valid_boxes) == 0:
        print(f"Frame {frame_id}: No valid YOLO detections")
        return detections, det_timing
    
    # YOLO BBox 两两去重 (基于 IoU/IoM)
    # 如果两个框重叠度极高(>0.95)，认为是 YOLO 对同一个物体识别了多个框，保留置信度高的那个
    n_boxes = len(valid_boxes)
    keep_box_flags = np.ones(n_boxes, dtype=bool)
    confs = np.array([float(boxes[valid_indices[i]].conf[0]) for i in range(n_boxes)])
    
    for i in range(n_boxes):
        if not keep_box_flags[i]:
            continue
        x1_i, y1_i, x2_i, y2_i = valid_boxes[i]
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        
        for j in range(i + 1, n_boxes):
            if not keep_box_flags[j]:
                continue
                
            x1_j, y1_j, x2_j, y2_j = valid_boxes[j]
            area_j = (x2_j - x1_j) * (y2_j - y1_j)
            
            # 计算交集
            inter_x1 = max(x1_i, x1_j)
            inter_y1 = max(y1_i, y1_j)
            inter_x2 = min(x2_i, x2_j)
            inter_y2 = min(y2_i, y2_j)
            
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue
                
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            
            # 计算 Intersection over Minimum (IoM)
            # 使用 IoM 而不是 IoU，可以防止"一个大框完全包住一个小框"时漏掉
            min_area = min(area_i, area_j)
            overlap_ratio = inter_area / float(min_area)
            
            if overlap_ratio > box_overlap_threshold:
                # 发现高度重叠框，抑制置信度较低的那个
                if confs[i] >= confs[j]:
                    keep_box_flags[j] = False
                else:
                    keep_box_flags[i] = False
                    break
                    
    # 根据过滤结果更新 valid_boxes
    valid_boxes = [valid_boxes[i] for i in range(n_boxes) if keep_box_flags[i]]
    valid_indices = [valid_indices[i] for i in range(n_boxes) if keep_box_flags[i]]

    # 步骤2: SAM分割，过滤掉mask像素面积过小的mask
    _t1 = time.time()
    sam_results = sam_model(rgb_image, bboxes=valid_boxes, verbose=False)
    if len(sam_results) == 0 or sam_results[0].masks is None:
        print(f"Frame {frame_id}: No SAM detections")
        return detections, det_timing
    masks = sam_results[0].masks.data.cpu().numpy()
    det_timing['sam'] = time.time() - _t1

    crops = []
    crops_bgr = []
    masked_crops = []
    valid_mask_indices = []
    for i, (bbox, mask) in enumerate(zip(valid_boxes, masks)):
        if mask.sum() < mask_area_threshold:
            print(f"    [MASK_FILTER_{frame_id}] Mask[{i}]: area={mask.sum()} < {MIN_MASK_AREA} -> SKIP")
            continue

        x1, y1, x2, y2 = bbox
        pad = 15
        cx1, cy1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
        cx2, cy2 = min(img_width, int(x2) + pad), min(img_height, int(y2) + pad)
        crop = rgb_image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        # 生成 masked crop：mask外区域置黑
        mask_in_crop = mask[cy1:cy2, cx1:cx2]
        m_crop = crop.copy()
        m_crop[~mask_in_crop] = 0

        crops_bgr.append(crop)  # 必须在 size 检查之后添加，保证索引对齐
        masked_crops.append(m_crop)
        crop_pil = Image.fromarray(crop)
        crops.append(clip_preprocess(crop_pil))
        valid_mask_indices.append(i)

    if len(crops) == 0:
        print(f"Frame {frame_id}: No valid crops")
        return detections, det_timing

    crops_tensor = torch.stack(crops).cuda()
    _t2 = time.time()
    with torch.no_grad():
        clip_features = clip_model.encode_image(crops_tensor)
        clip_features = clip_features.cpu().numpy()
    det_timing['clip'] = time.time() - _t2

    for i, mask_idx in enumerate(valid_mask_indices):
        box_idx = valid_indices[mask_idx]  
        box = boxes[box_idx]

        cls_id = int(box.cls[0])
        class_name = obj_classes.get_classes_arr()[cls_id] if cls_id < len(obj_classes.get_classes_arr()) else f"object_{cls_id}"
        conf = float(box.conf[0])

        # 计算视角得分
        view_score = calculate_view_score(masks[mask_idx].sum(), valid_boxes[mask_idx], img_width, img_height)

        # 通过 world_points 映射表或 sensor depth 反投影获取物体3D点
        _t3 = time.time()
        points_3d = np.zeros((0, 3), dtype=np.float32)
        points_rgb = np.zeros((0, 3), dtype=np.float32)
        depth_valid_mask = np.zeros(masks[mask_idx].shape, dtype=bool)
        if world_points is not None:
            # 缩放 mask 到 lingbot-map 分辨率，索引映射表
            h_lb, w_lb = world_points.shape[:2]
            mask_resized = cv2.resize(
                masks[mask_idx].astype(np.uint8), (w_lb, h_lb),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            pts_world = world_points[mask_resized]
            valid = np.any(pts_world != 0, axis=-1)
            points_3d = pts_world[valid]
            if len(points_3d) > 0:
                rgb_lb = cv2.resize(rgb_image, (w_lb, h_lb))
                points_rgb = (rgb_lb[mask_resized][valid] / 255.0).astype(np.float32)
                depth_valid_mask = np.ones(len(points_3d), dtype=bool)
        else:
            # fallback: 传感器深度反投影
            points_3d, depth_valid_mask = backproject_depth_with_mask(
                depth_image, masks[mask_idx], intrinsics, w2c
            )
            points_rgb = extract_rgb_by_mask(rgb_image, masks[mask_idx], depth_valid_mask)
            # MAD 统计滤波
            points_3d, points_rgb, depth_valid_mask = filter_point_cloud_fast(
                points_3d, points_rgb, depth_valid_mask, std_ratio=2.0, min_points=10
            )
        det_timing['backproj'] += time.time() - _t3

        det_points_3d = np.zeros((len(points_3d), 7))
        det_points_3d[:, :3] = points_3d
        det_points_3d[:, 3:6] = points_rgb
        det_points_3d[:, 6] = 0

        if len(points_3d) == 0: continue
        center_3d = np.median(points_3d[:, :3], axis=0)

        detection = {
            'idx': i,                                       # 物体的索引
            'bbox': valid_boxes[mask_idx],                  # 物体的边界框
            'mask': masks[mask_idx],                        # 物体的SAM掩码
            'class_name': class_name,                       # 物体的类别名称
            'clip_feature': clip_features[i],               # 物体的CLIP特征
            'confidence': conf,                             # 物体的置信度
            'image_crop': crops_bgr[i],                     # 物体裁剪图（bbox+padding）
            'masked_crop': masked_crops[i],                 # mask遮罩裁剪图（VLM输入）
            'view_score': view_score,                       # 视角得分
            'points_3d': det_points_3d,                     # 物体的3d点云 (K, 7)
            'depth_valid_mask': depth_valid_mask,           # 深度有效掩码
            'center_3d': center_3d,                         # 物体的中心点（中值）
        }
        detections.append(detection)
    
    print(f"Frame {frame_id}: {len(detections)} detections")
    return detections, det_timing

# ============================================================================
# 保存模块
# ============================================================================

def save_results(global_cloud, current_count, objects, output_dir):
    """保存点云和物体 — vstack 合并：背景(obj_idx=0) + 物体点云(obj_idx=obj.id)"""
    background = global_cloud[:current_count]
    all_obj_points = [obj.points_3d for obj in objects if len(obj.points_3d) > 0]

    if all_obj_points:
        point_cloud = np.vstack([background] + all_obj_points)
    else:
        point_cloud = background

    params_dict = {
        'means3D': point_cloud[:, :3],
        'rgb_colors': point_cloud[:, 3:6],
        'object_idx': point_cloud[:, 6].astype(np.int32),
        'log_scales': np.full((len(point_cloud), 3), -5.0, dtype=np.float32),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (len(point_cloud), 1)).astype(np.float32),
        'logit_opacities': np.full((len(point_cloud), 1), 100, dtype=np.float32),
    }

    save_path = os.path.join(output_dir, 'params_with_idx.npz')
    np.savez(save_path, **params_dict)
    n_bg = len(background)
    n_obj = len(point_cloud) - n_bg
    print(f"Saved point cloud to {save_path} (background={n_bg}, objects={n_obj}, total={len(point_cloud)})")

    objects_list = []
    for obj in objects:
        obj_dict = {
            'idx': obj.id,
            'class_name': obj.class_name,
            'clip_feature': obj.clip_feature,
            'center_3d': obj.center_3d,
            'image_crops': obj.image_crop,
            'masked_crops': getattr(obj, 'masked_crop', None),
        }
        objects_list.append(obj_dict)

    save_path = os.path.join(output_dir, 'objects.pkl.gz')
    with gzip.open(save_path, 'wb') as f:
        pickle.dump(objects_list, f)
    print(f"Saved {len(objects_list)} objects to {save_path}")

def save_memory(global_cloud, current_count, objects, output_dir, voxel_hash_to_idx=None):
    """保存完整的空间场景记忆"""
    valid_cloud = global_cloud[:current_count]
    params_dict = {
        'means3D': valid_cloud[:, :3],
        'rgb_colors': valid_cloud[:, 3:6],
        'object_idx': valid_cloud[:, 6].astype(np.int32),
    }
    pc_path = os.path.join(output_dir, 'memory_point_cloud.npz')
    np.savez(pc_path, **params_dict)
    print(f"Saved memory point cloud ({current_count} points) to {pc_path}")

    save_path = os.path.join(output_dir, 'memory_objects.pkl.gz')
    with gzip.open(save_path, 'wb') as f:
        pickle.dump(objects, f)
    print(f"Saved memory objects ({len(objects)} objects) to {save_path}")

    if voxel_hash_to_idx is not None:
        vhm_path = os.path.join(output_dir, 'memory_voxel_hash_to_idx.pkl.gz')
        with gzip.open(vhm_path, 'wb') as f:
            pickle.dump(voxel_hash_to_idx, f)
        print(f"Saved voxel_hash_to_idx ({len(voxel_hash_to_idx)} entries) to {vhm_path}")


def load_memory(memory_dir):
    """加载之前保存的空间场景记忆

    Returns:
        point_cloud: (N, 7)
        objects: list
        voxel_hash_to_idx: dict[int, int]
    """
    pc_path_new = os.path.join(memory_dir, 'memory_point_cloud.npz')
    pc_path_old = os.path.join(memory_dir, 'params_with_idx.npz')
    if os.path.exists(pc_path_new):
        data = np.load(pc_path_new)
    elif os.path.exists(pc_path_old):
        data = np.load(pc_path_old)
    else:
        raise FileNotFoundError(f"No point cloud file found in {memory_dir}")

    point_cloud = np.zeros((len(data['means3D']), 7), dtype=np.float32)
    point_cloud[:, :3] = data['means3D']
    point_cloud[:, 3:6] = data['rgb_colors']
    point_cloud[:, 6] = data['object_idx']

    obj_path_new = os.path.join(memory_dir, 'memory_objects.pkl.gz')
    obj_path_old = os.path.join(memory_dir, 'objects.pkl.gz')
    if os.path.exists(obj_path_new):
        obj_path = obj_path_new
    elif os.path.exists(obj_path_old):
        obj_path = obj_path_old
    else:
        raise FileNotFoundError(f"No objects file found in {memory_dir}")

    with gzip.open(obj_path, 'rb') as f:
        objects = pickle.load(f)

    # 加载体素哈希映射表
    vhm_path = os.path.join(memory_dir, 'memory_voxel_hash_to_idx.pkl.gz')
    if os.path.exists(vhm_path):
        with gzip.open(vhm_path, 'rb') as f:
            voxel_hash_to_idx = pickle.load(f)
        print(f"Loaded voxel_hash_to_idx: {len(voxel_hash_to_idx)} entries")
    else:
        # 从点云重建
        voxel_coords = np.floor(point_cloud[:, :3] / 0.02).astype(np.int32)
        offset = 100000
        hashes = (voxel_coords[:, 0] * offset * offset +
                  voxel_coords[:, 1] * offset + voxel_coords[:, 2])
        voxel_hash_to_idx = {int(h): i for i, h in enumerate(hashes)}
        print(f"Rebuilt voxel_hash_to_idx: {len(voxel_hash_to_idx)} entries")

    print(f"Loaded memory: {len(point_cloud)} points, {len(objects)} objects from {memory_dir}")
    return point_cloud, objects, voxel_hash_to_idx


# ============================================================================
# Main
# ============================================================================

def main(config):
    """主流程"""
    print("=" * 80)
    print("DGSG Refactor V2: Dynamic Point Cloud and Scene Graph Builder")
    print("=" * 80)
    total_start = time.time()

    seed_everything(config['seed'])

    output_dir = os.path.join(config["workdir"], config["run_name"])
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(config["primary_device"])

    print("\nLoading Dataset...")
    dataset_config = config["data"]

    # 从当前场景目录加载内参文件
    scene_dir = os.path.join(dataset_config["basedir"], dataset_config["sequence"])
    intrinsics_path = os.path.join(scene_dir, "intrinsics.yaml")
    if not os.path.exists(intrinsics_path):
        print(f"ERROR: 未找到内参文件 {intrinsics_path}")
        print(f"请先运行 capture 脚本生成相机内参")
        sys.exit(1)
    gradslam_data_cfg = load_dataset_config(intrinsics_path)

    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
    )

    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    print(f"Dataset loaded: {num_frames} frames")

    print("\n[1/4] Loading Models...")
    lang_config = config['lang']

    obj_classes = ObjectClasses(
        classes_file_path=lang_config['classes_file'],
        bg_classes=lang_config['bg_classes'],
        skip_bg=lang_config['skip_bg']
    )

    yolo_model = YOLO(lang_config['yolo_model_path'])
    yolo_model.set_classes(obj_classes.get_classes_arr())

    sam_model = SAM(lang_config['sam_model_path'])

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained=lang_config['clip_model_path']
    )
    clip_model = clip_model.to(device).eval()

    print("Models loaded successfully")

    print("\n[2/4] Initializing Data Structures...")
    global_cloud = np.zeros((MAX_POINTS, 7), dtype=np.float32)
    current_count = 0
    voxel_hash_to_idx = {}
    objects = []
    # 注：不再累积 objects_deleted，物体点云独立维护在 ObjectMemory.points_3d 中

    # 自动检测是否存在历史记忆文件
    memory_pc_path = os.path.join(output_dir, 'memory_point_cloud.npz')
    memory_obj_path = os.path.join(output_dir, 'memory_objects.pkl.gz')
    if os.path.exists(memory_pc_path) and os.path.exists(memory_obj_path):
        print(f"  Found existing memory in output directory, loading...")
        loaded_cloud, objects, loaded_vhi = load_memory(output_dir)
        current_count = len(loaded_cloud)
        global_cloud[:current_count] = loaded_cloud
        voxel_hash_to_idx = loaded_vhi
        # 兼容旧数据：确保加载的对象有 masked_crop 属性
        for obj in objects:
            if not hasattr(obj, 'masked_crop'):
                obj.masked_crop = None

    color, depth, intrinsics, gt_pose = dataset[0]
    intrinsics = intrinsics[:3, :3].cpu().numpy()
    first_frame_w2c = torch.linalg.inv(gt_pose).cpu().numpy()
    image_size = (dataset_config["desired_image_height"], dataset_config["desired_image_width"])

    print(f"Image size: {image_size}")
    print(f"Intrinsics:\n{intrinsics}")

    print(f"\n[3/4] Processing {num_frames} frames...")
    print("=" * 80)

    # 计时累加器
    time_accum = {
        'detect': 0.0, 'yolo': 0.0, 'sam': 0.0, 'clip': 0.0, 'backproj': 0.0,
        'backproj_global': 0.0,
        'project': 0.0, 'match': 0.0, 'update': 0.0, 'delete': 0.0,
        'reproject': 0.0, 'visual': 0.0, 'total': 0.0,
    }

    for time_idx in tqdm(range(num_frames), desc="Processing frames"):
        frame_start_time = time.time()

        color, depth, _, gt_pose = dataset[time_idx]

        rgb_image_hwc = color.cpu().numpy().astype(np.uint8)

        depth_image = depth.cpu().numpy()
        if depth_image.ndim == 3 and depth_image.shape[2] == 1:
            depth_image = depth_image[:, :, 0]

        gt_w2c = torch.linalg.inv(gt_pose).cpu().numpy()

        # step 0: load lingbot-map pre-computed world point cloud
        frame_stem = dataset.color_paths[time_idx].split("/")[-1].rsplit(".", 1)[0]
        lb_point_path = os.path.join(scene_dir, "point", f"{frame_stem}.npy")
        world_points = np.load(lb_point_path)  # (H_lb, W_lb, 3)

        # step 1: add valid points to global cloud + voxel dedup
        bp_start = time.time()
        current_count = add_lb_points_to_global_cloud(
            world_points, rgb_image_hwc, global_cloud, current_count
        )
        time_accum["backproj_global"] += time.time() - bp_start

        if time_idx % 10 == 0:
            print(f"  Frame {time_idx}: global_cloud={current_count} pts")

        # step 2: detect objects via world_points indexing
        detect_start = time.time()
        detections, det_timing = detect_objects(
            rgb_image_hwc, depth_image, yolo_model, sam_model, clip_model,
            clip_preprocess, obj_classes, lang_config, time_idx,
            intrinsics, gt_w2c, world_points=world_points
        )
        detect_time = time.time() - detect_start

        # Step 3: 批量投影（将objects对应的每个物体object点云投影到当前视角，得到每个物体的mask）
        project_start = time.time()
        projected_masks = batch_project_objects(
            objects, intrinsics, gt_w2c, image_size[0], image_size[1]
        )
        project_time = time.time() - project_start

        # Step 4: 多模态匹配（将detections中的每个检测框与objects中的每个物体object进行匹配）
        match_start = time.time()
        match_result = match_detections_to_objects(detections, projected_masks, objects)
        match_time = time.time() - match_start

        # Step 5: 融合更新（detection-mask匹配上了就融合更新object，没有匹配上就添加成为新的object，新增点云并更新point_cloud）
        update_start = time.time()
        fuse_update_matched(objects, detections, match_result)
        objects_to_add = add_new_objects(objects, detections, match_result)
        merge_nearby_objects(objects, center_threshold=0.3, clip_threshold=0.7)
        update_time = time.time() - update_start

        # Step 6: 删除过时的object（前面没有被匹配上的object，将其投影到2d平面上，计算每个像素z值与实际深度的平均差值，以及彩色和实际彩色的平均差值，超过阈值就删除掉）
        delete_start = time.time()
        objects_to_remove = remove_stale_objects(
            objects, match_result, rgb_image_hwc, depth_image,
            intrinsics, gt_w2c, DEPTH_ERROR_THRESHOLD, COLOR_ERROR_THRESHOLD, disappear_ratio_thr=0.5, frustum_ratio_thr=0.7, visibility_ratio_thr=0.7
        )
        delete_time = time.time() - delete_start

        # 第二次投影：获取更新后的物体masks
        reproject_start = time.time()
        updated_projected_masks = batch_project_objects(
            objects, intrinsics, gt_w2c, image_size[0], image_size[1]
        )
        reproject_time = time.time() - reproject_start

        # Step 7: 保存可视化（简化版，无投影物体）
        visual_start = time.time()
        visualize_frame(rgb_image_hwc, detections, objects, match_result,
                      projected_masks, objects_to_add, objects_to_remove,
                      updated_projected_masks, output_dir, time_idx)
        visual_time = time.time() - visual_start

        # 打印统计信息（打印每个阶段的耗时）
        frame_time = time.time() - frame_start_time
        print(f"Frame {time_idx}: Det={len(detections)}, Match={len(match_result['matched_pairs'])}, "
              f"Total={len(objects)}, Points={current_count}")
        print(f"  Time: detect={detect_time:.4f}s (yolo={det_timing['yolo']:.4f}, sam={det_timing['sam']:.4f}, "
              f"clip={det_timing['clip']:.4f}, backproj={det_timing['backproj']:.4f}), "
              f"proj={project_time:.4f}s, match={match_time:.4f}s, "
              f"update={update_time:.4f}s, delete={delete_time:.4f}s, "
              f"reproj={reproject_time:.4f}s, visual={visual_time:.4f}s, total={frame_time:.4f}s")
        print("\n")

        # 累加计时
        time_accum['detect'] += detect_time
        time_accum['yolo'] += det_timing['yolo']
        time_accum['sam'] += det_timing['sam']
        time_accum['clip'] += det_timing['clip']
        time_accum['backproj'] += det_timing['backproj']
        time_accum['project'] += project_time
        time_accum['match'] += match_time
        time_accum['update'] += update_time
        time_accum['delete'] += delete_time
        time_accum['reproject'] += reproject_time
        time_accum['visual'] += visual_time
        time_accum['total'] += frame_time

    # ========== 计时汇总统计 ==========
    N = num_frames
    if N > 0:
        print("\n" + "=" * 80)
        print("各模块耗时汇总统计（{} 帧）".format(N))
        print("=" * 80)
        print(f"{'模块':<24} {'总耗时(s)':<14} {'单帧平均(ms)':<14} {'占比(%)':<10}")
        print("-" * 62)

        modules = [
            ('全局反投影+下采样', 'backproj_global'),
            ('YOLO 检测',         'yolo'),
            ('SAM 分割',          'sam'),
            ('CLIP 特征提取',     'clip'),
            ('深度反投影(检测)',   'backproj'),
            ('检测模块合计',      'detect'),
            ('历史点云投影',      'project'),
            ('多特征匹配',        'match'),
            ('融合更新(新增+更新)','update'),
            ('删除更新',          'delete'),
            ('二次投影',          'reproject'),
            ('可视化保存',        'visual'),
        ]
        for label, key in modules:
            t = time_accum[key]
            avg = t / N * 1000
            pct = t / max(time_accum['total'], 1e-9) * 100
            print(f"  {label:<22} {t:<14.3f} {avg:<14.1f} {pct:<10.1f}")

        print("-" * 62)
        total_t = time_accum['total']
        avg_frame = total_t / N * 1000
        print(f"  {'单帧总耗时(不含IO)':<22} {total_t:<14.3f} {avg_frame:<14.1f} {'100.0':<10}")
        print("=" * 80)

    # Step 4: 保存结果
    print(f"\n[4/4] Saving Results...")
    save_results(global_cloud, current_count, objects, output_dir)
    save_memory(global_cloud, current_count, objects, output_dir, voxel_hash_to_idx)

    total_time = time.time() - total_start
    print(f"\n[Done]")
    print(f"Output directory: {output_dir}")
    print(f"Total objects: {len(objects)}")
    print(f"Total points: {current_count}")
    print(f"Total time: {total_time:.4f}s")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--num_frames", type=int, default=-1, help="Number of frames to process")
    args = parser.parse_args()

    config_path = args.config
    config_module = SourceFileLoader(os.path.basename(config_path), config_path).load_module()
    config = config_module.config

    if args.num_frames > 0:
        config['data']['num_frames'] = args.num_frames

    main(config)
