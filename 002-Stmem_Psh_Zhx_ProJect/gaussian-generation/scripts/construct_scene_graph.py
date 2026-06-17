"""
构建场景图：加载物体数据，生成描述和关系，构建场景图JSON

流程：
  1. 加载 params_with_idx.npz 和 objects.pkl.gz
  2. 调用 Moondream 生成物体描述和关系
  3. 保存 objects.pkl.gz, relations.pkl.gz
  4. 构建 scene_graph.json

用法:
    python scripts/construct_scene_graph.py configs/mydata/dgsg.py
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





def get_objects_relations(objects: MapObjectList, lf_config, output_dir,
                            means3D=None, obj_idx_arr=None, distance_threshold=2.0):
    """
    Generate object and relations using Moondream.
    Phase 1: Moondream with single object image (generate category and description)
    Phase 2: Moondream with concatenated object images (generate relations)

    Args:
        objects: MapObjectList of objects
        lf_config: Language config dict
        output_dir: Output directory path
        means3D: 3D points array (N, 3) for distance filtering
        obj_idx_arr: Object index array (N,) for each point
        distance_threshold: Max distance (meters) to consider relationship (default: 2.0m)
    """
    total_time_start = time.time()
    
    # Initialize Moondream
    model_path = lf_config.get('moondream_model_path', "models/moondream2")
    load_4bit = lf_config.get('moondream_4bit', False)
    moondream_vlm = MoondreamVLM(model_path=model_path, load_in_4bit=load_4bit)

    objects_img_crop_save_path = Path(output_dir) / "objects_img_crop"
    objects_img_crop_save_path.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Object Descriptions ---
    print("=== Phase 1: Object Descriptions with Moondream ===")
    description_time_start = time.time()
    for i, obj in enumerate(tqdm(objects, desc="Processing Objects")):
        moondream_time_start = time.time()

        # 保存 crop 图用于 Phase 2 关系预测
        img_crop = np.array(obj['image_crops'])
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{objects_img_crop_save_path}/{obj['idx']}.jpg", img_crop)

        image_pil = Image.fromarray(np.array(obj['image_crops']))
        prompt = (
            "What is this object? Answer in this exact format:\n"
            "Category: <one or two word noun>\n"
            "Description: <short description under ten words>\n\n"
            "Rules:\n"
            "- Category must be a simple noun: chair, table, monitor, bottle, box, wall, door, pillow, plant, etc.\n"
            "- Describe only the object's appearance and material\n"
        )

        # Use Moondream
        response = moondream_vlm.generate_content(image_pil, prompt)
        
        # Parse
        category = "Unknown"
        description = response
        if "Category:" in response and "Description:" in response:
            try:
                parts = response.split("Description:")
                cat_part = parts[0].split("Category:")[1].strip()
                desc_part = parts[1].strip()
                category = cat_part
                description = desc_part
            except:
                pass
        # Clean up category
        category = category.strip().rstrip('.')
        category = category.strip('[]{}()')
        # 如果 category 包含场景描述词，用 class_name fallback
        scene_keywords = ['scene', 'image', 'indoor', 'outdoor', 'room', 'view', 'picture', 'photo']
        if any(kw in category.lower() for kw in scene_keywords):
            category = obj.get('class_name', 'unknown')
        obj['category'] = category
        obj['description'] = description

        moondream_time_end = time.time()
        print(f"Object {i} - Category: {obj['category']}, Description: {description}")
        print(f"Moondream Time: {moondream_time_end - moondream_time_start:.4f}s")
    description_time_end = time.time()

    # --- Phase 2: Relations with Concatenated Images ---
    print("=== Phase 2: Relationship Prediction with Moondream (Concatenated Images) ===")
    relation_time_start = time.time()
    relations = []

    # Compute object centers for distance filtering
    obj_centers = {}
    if means3D is not None and obj_idx_arr is not None:
        print(f"Computing object centers for distance filtering (threshold: {distance_threshold}m)...")
        for obj in objects:
            idx = obj['idx']
            mask = (obj_idx_arr == idx)
            if np.any(mask):
                obj_centers[idx] = np.mean(means3D[mask], axis=0)
            else:
                obj_centers[idx] = None

    # Count for statistics
    total_pairs = len(objects) * (len(objects) - 1) // 2
    filtered_pairs = 0

    # Helper function to concatenate images horizontally
    def concat_images(img1, img2, padding=10):
        """Concatenate two images horizontally with padding"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        # Resize to same height
        target_h = max(h1, h2)
        if h1 < target_h:
            img1 = cv2.resize(img1, (int(w1 * target_h / h1), target_h))
        if h2 < target_h:
            img2 = cv2.resize(img2, (int(w2 * target_h / h2), target_h))
        # Add padding
        padding_img = np.ones((target_h, padding, 3), dtype=np.uint8) * 255
        # Concatenate
        concat_img = np.hstack([img1, padding_img, img2])
        return concat_img
    
    for i, obj1 in enumerate(tqdm(objects, desc="Analyzing Relations")):
        for j, obj2 in enumerate(objects[i + 1:], start=i + 1):
            # Skip same type objects (e.g., chair + chair)
            cat1_clean = obj1['category'].replace('[', '').replace(']', '').split('/')[0].strip().lower()
            cat2_clean = obj2['category'].replace('[', '').replace(']', '').split('/')[0].strip().lower()
            if cat1_clean == cat2_clean:
                continue

            # Distance filtering
            if obj_centers:
                center1 = obj_centers.get(obj1['idx'])
                center2 = obj_centers.get(obj2['idx'])
                if center1 is not None and center2 is not None:
                    dist = np.linalg.norm(center1 - center2)
                    if dist > distance_threshold:
                        continue  # Skip this pair
            filtered_pairs += 1

            moondream_time_start = time.time()
            rel = {}
            rel['obj1_id'] = obj1['idx']
            rel['obj2_id'] = obj2['idx']
            
            # Load images
            img1_path = f"{objects_img_crop_save_path}/{obj1['idx']}.jpg"
            img2_path = f"{objects_img_crop_save_path}/{obj2['idx']}.jpg"
            
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            # Concatenate images
            concat_img = concat_images(img1, img2, padding=20)
            concat_img_pil = Image.fromarray(cv2.cvtColor(concat_img, cv2.COLOR_BGR2RGB))
            
            # Construct Prompt with actual object categories
            # Clean category names (remove brackets, take first if multiple)
            cat1 = obj1['category'].replace('[', '').replace(']', '').split('/')[0].strip()
            cat2 = obj2['category'].replace('[', '').replace(']', '').split('/')[0].strip()

            prompt = (
                f"A {cat1} and a {cat2}.\n"
                f"What does the {cat1} DO to the {cat2}?\n"
                f"Only output 1-2 words, nothing else.\n\n"
                f"Examples:\n"
                f"chair + desk → sit at\n"
                f"fan + room → cool\n"
                f"lamp + book → light\n"
                f"door + wall → block\n"
                f"light + ceiling → illuminate\n"
                f"same type objects → none\n"
                f"unrelated objects → none\n\n"
                f"Answer:"
            )
            
            # Use Moondream with concatenated image
            response = moondream_vlm.generate_content(concat_img_pil, prompt)
            response = response.strip().lower()

            # Clean response: remove arrow format (e.g., "chair + desk → sit at" -> "sit at")
            if '→' in response:
                response = response.split('→')[-1].strip()
            # Remove "none" related patterns
            if 'none' in response or 'unrelated' in response:
                continue


            rel['pretential_relation'] = response
            relations.append(rel)
            moondream_time_end = time.time()
            print(f"Relation {obj1['category']}: ({obj1['idx']}) - {obj2['category']}:({obj2['idx']}): {response}")
            print(f"Moondream Time: {moondream_time_end - moondream_time_start:.4f}s")
    relation_time_end = time.time()
    total_time_end = time.time()

    # Print filtering statistics
    if obj_centers:
        print(f"\n[Distance Filtering] Total pairs: {total_pairs}, Filtered pairs: {filtered_pairs}, "
              f"Skipped: {total_pairs - filtered_pairs} ({100*(total_pairs-filtered_pairs)/total_pairs:.1f}%)")

    # Helper function to format seconds to HH:MM:SS
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
    print(f"Total Time - Save Objects: {format_time(description_time_end - description_time_start)}")
    print(f"Total Time - Save Relations: {format_time(relation_time_end - relation_time_start)}")
    print(f"Total Time - Save Objects & Relations: {format_time(total_time_end - total_time_start)}")

    return objects, relations


def get_spatial_relation(center1, center2):
    """
    Compute spatial relation between two centers.
    Returns a string describing the relative position of obj2 with respect to obj1.
    """
    diff = center2 - center1
    dist = np.linalg.norm(diff)
    
    # Determine dominant axis
    abs_diff = np.abs(diff)
    max_axis = np.argmax(abs_diff)
    
    if max_axis == 0:
        rel = "right" if diff[0] > 0 else "left"
    elif max_axis == 1:
        rel = "bottom" if diff[1] > 0 else "top" # Assuming Y increases downwards (image coords), adjust if needed
    else:
        rel = "front" if diff[2] > 0 else "back"
        
    return f"{rel} (distance: {dist:.2f})"

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

    # 距离过滤阈值（从配置读取，默认 2.0 米）
    distance_threshold = config['lang'].get('relation_distance_threshold', 2.0)

    # 获取物体数据和关系数据（调用Moondream生成描述和关系）
    objects, relations = get_objects_relations(
        objects, config['lang'], output_dir,
        means3D=means3D, obj_idx_arr=obj_idx_arr, distance_threshold=distance_threshold
    )

    # 保存 objects.pkl.gz 和 relations.pkl.gz
    objects_save_path = os.path.join(output_dir, "objects.pkl.gz")
    relations_save_path = os.path.join(output_dir, "relations.pkl.gz")

    print(f"Saving objects to {objects_save_path}...")
    with gzip.open(objects_save_path, "wb") as f:
        pickle.dump(objects.to_serializable(), f)

    print(f"Saving relations to {relations_save_path}...")
    with gzip.open(relations_save_path, "wb") as f:
        pickle.dump(relations, f)

    scene_graph = {
        "nodes": [],
        "edges": []
    }
    
    # Process Nodes（构建节点）
    obj_centers = {} # idx -> center
    for obj in objects:
        idx = obj['idx']
        # Filter gaussians for this object
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
        # 对边界框坐标进行四舍五入保留后3位小数并转换为 list
        bbox = [np.round(min_bound, 3).tolist(), np.round(max_bound, 3).tolist()]
            
        obj_centers[idx] = center
        
        # 对中心点坐标进行四舍五入保留后3位小数并转换为 list
        rounded_center = np.round(center, 3).tolist() if isinstance(center, np.ndarray) else [round(c, 3) for c in center]
        
        node = {
            "idx": idx,
            "category": obj.get('category', 'unknown'),
            "description": obj.get('description', ''),
            "center": rounded_center,
            "bbox": bbox
        }
        scene_graph["nodes"].append(node)
        
    # Process Edges（构建边）
    for rel in relations:
        obj1_id = rel.get('obj1_id')
        obj2_id = rel.get('obj2_id')
        
        if obj1_id is None or obj2_id is None:
            continue
            
        pretential_relation = rel.get('pretential_relation', rel.get('outputs', ''))
        
        location_relation = "unknown"
        if obj1_id in obj_centers and obj2_id in obj_centers:
            center1 = obj_centers[obj1_id]
            center2 = obj_centers[obj2_id]
            location_relation = get_spatial_relation(center1, center2)
            
        edge = {
            "obj1": obj1_id,
            "obj2": obj2_id,
            "pretential_relation": pretential_relation,
            "location_relation": location_relation
        }
        scene_graph["edges"].append(edge)
        
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
