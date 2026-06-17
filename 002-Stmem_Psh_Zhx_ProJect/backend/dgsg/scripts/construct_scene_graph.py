"""
构建场景图（Lingbot增强版）：加载物体数据，生成描述，用几何方法计算空间关系

改进点：
  1. 优先使用 masked_crops（mask遮罩后的裁剪图）消除背景干扰
  2. 类别和描述独立预测（拆分prompt）
  3. 用增强版3D几何计算替代VLM关系预测（6方向+距离+on/contain/attach）
  4. CLIP验证闭环：验证Moondream输出的类别是否合理

流程：
  1. 加载 params_with_idx.npz 和 objects.pkl.gz
  2. 调用 Moondream 生成物体类别和描述（独立预测）
  3. CLIP 验证类别，不通过则 fallback 到 YOLO class_name
  4. 用 3D 几何计算增强版空间关系（替代 VLM 关系预测）
  5. 构建 scene_graph.json

用法:
    python scripts/construct_scene_graph_lingbot.py configs/mydata/lingbot.py
"""

import argparse
import os
import sys
import time
import gzip
import pickle
import json
import numpy as np
import cv2

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from importlib.machinery import SourceFileLoader
from utils.common_utils import seed_everything
from utils.slam_classes import MapObjectList
from vlm_utils.moondream_local import MoondreamVLM
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import open_clip


def get_spatial_relation_enhanced(center1, center2, bbox1, bbox2, contact_threshold=0.05):
    """
    增强版空间关系：6方向 + 距离 + 接触关系(on/contain/attach)

    Args:
        center1, center2: 物体中心 (3,)
        bbox1, bbox2: 物体 bbox [[min_x,min_y,min_z], [max_x,max_y,max_z]]
        contact_threshold: 接触判定距离阈值（米），默认5cm
    Returns:
        dict: {spatial, distance, contact}
    """
    diff = center2 - center1
    dist = np.linalg.norm(diff)

    # --- 空间方向（基于中心差值的主轴） ---
    abs_diff = np.abs(diff)
    max_axis = np.argmax(abs_diff)
    if max_axis == 0:
        direction = "right" if diff[0] > 0 else "left"
    elif max_axis == 1:
        direction = "above" if diff[1] > 0 else "below"
    else:
        direction = "front" if diff[2] > 0 else "back"

    # --- 接触关系（基于 3D bbox 重叠） ---
    min1, max1 = np.array(bbox1[0]), np.array(bbox1[1])
    min2, max2 = np.array(bbox2[0]), np.array(bbox2[1])

    # bbox 体积
    vol1 = max(np.prod(max1 - min1), 1e-6)
    vol2 = max(np.prod(max2 - min2), 1e-6)

    # 重叠检测
    overlap_min = np.maximum(min1, min2)
    overlap_max = np.minimum(max1, max2)
    overlap_size = np.maximum(overlap_max - overlap_min, 0)
    overlap_vol = np.prod(overlap_size)

    contact = "none"

    if overlap_vol > 0:
        # 有实际重叠
        if overlap_vol / vol2 > 0.5:
            contact = "contain"
        elif overlap_vol / vol1 > 0.5:
            contact = "contain"
        else:
            # 检测 on（上方接触）
            y_gap = min2[1] - max1[1]
            if abs(y_gap) < contact_threshold and direction == "above":
                contact = "on"
            else:
                contact = "attach"
    else:
        # 无重叠，检查最近轴距离
        gap = overlap_max - overlap_min  # 负值表示有间隙
        min_gap = np.max(gap)  # 最大的间隙（最近轴方向）
        if min_gap < contact_threshold:
            if abs(min2[1] - max1[1]) < contact_threshold and direction == "above":
                contact = "on"
            elif abs(max2[1] - min1[1]) < contact_threshold and direction == "below":
                contact = "on"
            else:
                contact = "attach"

    return {
        'spatial': direction,
        'distance': round(float(dist), 3),
        'contact': contact
    }


def verify_category_with_clip(image_crop, category, clip_model, clip_preprocess, clip_tokenizer, threshold=0.22):
    """
    用 CLIP 验证 Moondream 输出的类别是否合理。
    如果 CLIP 相似度低于阈值，返回 None。
    """
    if image_crop is None or category is None or category == "Unknown":
        return category

    try:
        if isinstance(image_crop, np.ndarray):
            image_pil = Image.fromarray(image_crop)
        else:
            image_pil = image_crop

        image_input = clip_preprocess(image_pil).unsqueeze(0).cuda()
        text_input = clip_tokenizer([f"a {category}", "a object"]).cuda()

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        similarity = (image_features @ text_features.T).squeeze(0)
        clip_score = similarity[0].item()

        print(f"    [CLIP_VERIFY] category='{category}', score={clip_score:.4f} ({'PASS' if clip_score >= threshold else 'FAIL'})")

        if clip_score < threshold:
            return None
        return category
    except Exception as e:
        print(f"    [CLIP_VERIFY] Error: {e}")
        return category


def get_objects_descriptions(objects: MapObjectList, lf_config, output_dir,
                             clip_model=None, clip_preprocess=None, clip_tokenizer=None):
    """
    Phase 1: 用 Moondream 生成物体类别和描述（独立预测 + CLIP 验证）
    """
    total_time_start = time.time()

    # Initialize Moondream
    model_path = lf_config.get('moondream_model_path', "models/moondream2")
    load_4bit = lf_config.get('moondream_4bit', False)
    moondream_vlm = MoondreamVLM(model_path=model_path, load_in_4bit=load_4bit)

    objects_img_crop_save_path = Path(output_dir) / "objects_img_crop"
    objects_img_crop_save_path.mkdir(parents=True, exist_ok=True)

    # Prompt 定义（独立预测）
    category_prompt = (
        "What is this object? Answer with exactly ONE word: the object category.\n"
        "Examples: chair, table, monitor, bottle, box, wall, door, pillow, plant, cup, keyboard, trash can.\n"
        "Answer with ONLY the category name, nothing else."
    )

    description_prompt = (
        "Describe this object briefly in one sentence.\n"
        "Focus on color, material, and shape.\n"
        "Answer in under 10 words."
    )

    # --- Phase 1: Object Descriptions ---
    print("=== Phase 1: Object Descriptions (Masked Crop + Split Prompts + CLIP Verify) ===")
    description_time_start = time.time()

    # 计时累加器
    timing_stats = {
        'category_total': 0.0, 'category_count': 0,
        'description_total': 0.0, 'description_count': 0,
        'clip_verify_total': 0.0, 'clip_verify_count': 0,
    }

    for i, obj in enumerate(tqdm(objects, desc="Processing Objects")):
        obj_time_start = time.time()

        # 优先使用 masked_crops，fallback 到 image_crops
        if 'masked_crops' in obj and obj['masked_crops'] is not None:
            img_crop = np.array(obj['masked_crops'])
            crop_source = "masked"
        else:
            img_crop = np.array(obj['image_crops'])
            crop_source = "raw"

        # 保存 crop 图（用于 Viewer 显示）
        img_crop_save = np.array(obj['image_crops'])
        img_crop_save = cv2.cvtColor(img_crop_save, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{objects_img_crop_save_path}/{obj['idx']}.jpg", img_crop_save)

        image_pil = Image.fromarray(img_crop)

        # 独立预测 1: 类别
        _t_cat_start = time.time()
        category_response = moondream_vlm.generate_content(image_pil, category_prompt)
        category = category_response.strip().rstrip('.')
        _t_cat = time.time() - _t_cat_start
        timing_stats['category_total'] += _t_cat
        timing_stats['category_count'] += 1

        # 清理 category
        category = category.strip().rstrip('.')
        category = category.strip('[]{}()')

        # CLIP 验证
        if clip_model is not None:
            _t_clip_start = time.time()
            verified = verify_category_with_clip(
                img_crop, category, clip_model, clip_preprocess, clip_tokenizer
            )
            _t_clip = time.time() - _t_clip_start
            timing_stats['clip_verify_total'] += _t_clip
            timing_stats['clip_verify_count'] += 1
            if verified is None:
                original_class = obj.get('class_name', 'unknown')
                print(f"    [CLIP_FALLBACK] '{category}' -> '{original_class}'")
                category = original_class

        # 场景描述词 fallback
        scene_keywords = ['scene', 'image', 'indoor', 'outdoor', 'room', 'view', 'picture', 'photo', 'the', 'this', 'it']
        if any(kw == category.lower() for kw in scene_keywords):
            category = obj.get('class_name', 'unknown')

        # 独立预测 2: 描述
        _t_desc_start = time.time()
        description_response = moondream_vlm.generate_content(image_pil, description_prompt)
        description = description_response.strip()
        _t_desc = time.time() - _t_desc_start
        timing_stats['description_total'] += _t_desc
        timing_stats['description_count'] += 1

        obj['category'] = category
        obj['description'] = description

        obj_time = time.time() - obj_time_start
        print(f"Object {i} [{crop_source}] - Category: {obj['category']}, Description: {description}")
        print(f"    Timing: category={_t_cat:.3f}s, description={_t_desc:.3f}s, total={obj_time:.3f}s")

    description_time_end = time.time()
    total_time_end = time.time()

    # Helper function to format seconds
    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        if h > 0:
            return f"{h}h {m}m {s:.2f}s"
        elif m > 0:
            return f"{m}m {s:.2f}s"
        else:
            return f"{s:.2f}s"

    # 计时统计汇总
    n_cat = timing_stats['category_count']
    n_desc = timing_stats['description_count']
    n_clip = timing_stats['clip_verify_count']
    print(f"\n{'='*60}")
    print(f"Phase 1 Timing Summary")
    print(f"{'='*60}")
    print(f"  类别预测:   总计={timing_stats['category_total']:.3f}s, 次数={n_cat}, 平均={timing_stats['category_total']/max(n_cat,1)*1000:.1f}ms")
    print(f"  描述预测:   总计={timing_stats['description_total']:.3f}s, 次数={n_desc}, 平均={timing_stats['description_total']/max(n_desc,1)*1000:.1f}ms")
    print(f"  CLIP验证:   总计={timing_stats['clip_verify_total']:.3f}s, 次数={n_clip}, 平均={timing_stats['clip_verify_total']/max(n_clip,1)*1000:.1f}ms")
    print(f"  Phase 1 总计: {format_time(description_time_end - description_time_start)}")
    print(f"  总耗时 (含初始化): {format_time(total_time_end - total_time_start)}")
    print(f"{'='*60}")

    return objects


def construct_graph(config: dict):
    input_dir = os.path.join(config["workdir"], config["run_name"])
    output_dir = input_dir
    param_path = os.path.join(input_dir, "params_with_idx.npz")
    objects_path = os.path.join(input_dir, "objects.pkl.gz")
    graph_path = os.path.join(output_dir, "scene_graph.json")

    # 加载高斯球数据
    print(f"Loading parameters from {param_path}...")
    params = np.load(param_path)
    means3D = params['means3D']
    obj_idx_arr = params['object_idx']
    obj_idx_arr = np.asarray(obj_idx_arr).reshape(-1)

    # 加载物体数据
    print(f"Loading objects from {objects_path}...")
    with gzip.open(objects_path, "rb") as f:
        objects_list = pickle.load(f)

    # 转换为 MapObjectList
    objects = MapObjectList(objects_list)

    # 初始化 CLIP（用于验证，从本地权重加载）
    clip_model = None
    clip_preprocess = None
    clip_tokenizer = None
    clip_model_path = config['lang'].get('clip_model_path', './models/open_clip_pytorch_model.bin')
    try:
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained=clip_model_path
        )
        clip_model = clip_model.cuda().eval()
        clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        print(f"CLIP model loaded from {clip_model_path}")
    except Exception as e:
        print(f"Warning: CLIP model not available, skipping verification: {e}")

    # 获取物体描述（Phase 1: Moondream + CLIP 验证）
    objects = get_objects_descriptions(
        objects, config['lang'], output_dir,
        clip_model=clip_model, clip_preprocess=clip_preprocess, clip_tokenizer=clip_tokenizer
    )

    # 保存 objects.pkl.gz
    objects_save_path = os.path.join(output_dir, "objects.pkl.gz")
    print(f"Saving objects to {objects_save_path}...")
    with gzip.open(objects_save_path, "wb") as f:
        pickle.dump(objects.to_serializable(), f)

    # 距离过滤阈值
    distance_threshold = config['lang'].get('relation_distance_threshold', 2.5)

    # 构建场景图
    scene_graph = {
        "nodes": [],
        "edges": []
    }

    # === Phase 2: 几何空间关系计算（替代 VLM 关系预测）===
    print(f"=== Phase 2: Geometric Spatial Relations (threshold: {distance_threshold}m) ===")
    relation_time_start = time.time()

    # Process Nodes + 计算 bbox
    obj_centers = {}
    obj_bboxes = {}
    for obj in objects:
        idx = obj['idx']
        mask = (obj_idx_arr == idx)
        if not np.any(mask):
            center = np.zeros(3, dtype=float)
            min_bound = np.zeros(3, dtype=float)
            max_bound = np.zeros(3, dtype=float)
        else:
            obj_points = means3D[mask]
            center = np.mean(obj_points, axis=0)
            min_bound = np.min(obj_points, axis=0)
            max_bound = np.max(obj_points, axis=0)
        bbox = [np.round(min_bound, 3).tolist(), np.round(max_bound, 3).tolist()]

        obj_centers[idx] = center
        obj_bboxes[idx] = bbox

        rounded_center = np.round(center, 3).tolist() if isinstance(center, np.ndarray) else [round(c, 3) for c in center]

        node = {
            "idx": idx,
            "category": obj.get('category', 'unknown'),
            "description": obj.get('description', ''),
            "center": rounded_center,
            "bbox": bbox
        }
        scene_graph["nodes"].append(node)

    # Process Edges（几何空间关系）
    total_pairs = len(objects) * (len(objects) - 1) // 2
    edge_count = 0
    skipped_count = 0

    for i in range(len(scene_graph["nodes"])):
        for j in range(i + 1, len(scene_graph["nodes"])):
            node1 = scene_graph["nodes"][i]
            node2 = scene_graph["nodes"][j]
            idx1, idx2 = node1["idx"], node2["idx"]

            if idx1 not in obj_centers or idx2 not in obj_centers:
                skipped_count += 1
                continue

            # 距离过滤
            dist = np.linalg.norm(obj_centers[idx1] - obj_centers[idx2])
            if dist > distance_threshold:
                skipped_count += 1
                continue

            relation = get_spatial_relation_enhanced(
                obj_centers[idx1], obj_centers[idx2],
                obj_bboxes[idx1], obj_bboxes[idx2]
            )

            edge = {
                "obj1": idx1,
                "obj2": idx2,
                "spatial_relation": relation['spatial'],
                "distance": relation['distance'],
                "contact_relation": relation['contact'],
            }
            scene_graph["edges"].append(edge)
            edge_count += 1

    relation_time_end = time.time()
    relation_time = relation_time_end - relation_time_start

    print(f"[Spatial Relations] Total pairs: {total_pairs}, Edges: {edge_count}, "
          f"Skipped: {skipped_count} ({100*skipped_count/max(total_pairs,1):.1f}%)")
    print(f"Phase 2 Time: {relation_time:.4f}s")
    if edge_count > 0:
        print(f"  关系预测:   总计={relation_time*1000:.1f}ms, 次数={edge_count}, 平均={relation_time/edge_count*1000:.3f}ms/对")

    print(f"Saving scene graph to {graph_path}...")
    with open(graph_path, "w") as f:
        json.dump(scene_graph, f, indent=4)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    args = parser.parse_args()
    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])

    construct_graph(experiment.config)
