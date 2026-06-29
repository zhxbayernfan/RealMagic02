"""
StreamSceneGraph: 流式场景图状态管理模块
管节点/边状态、CLIP 类别名决策、几何关系计算、VLM 描述生成
"""
import os
import json
import gzip
import pickle
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from collections import Counter

# --- sys.path 设置（与 dgsg_refactor_lingbot.py 相同模式）---
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SCRIPTS_DIR)
_SERVICE_DIR = os.path.dirname(_BASE_DIR)
import sys
sys.path.insert(0, _BASE_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _SERVICE_DIR)

# --- moondream2 源码在 models/ 目录下，需加入 sys.path ---
# 目录结构: stmem-psh/ -> {models/moondream2/, backend/dgsg/scripts/scene_graph_stream.py}
# _SCRIPTS_DIR = .../backend/dgsg/scripts/ -> 上3级到 stmem-psh/
_MODELS_DIR = os.path.normpath(os.path.join(_SCRIPTS_DIR, "..", "..", "..", "models"))
if os.path.isdir(_MODELS_DIR) and _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

from dgsg_refactor_lingbot import cosine_similarity
from construct_scene_graph_lingbot import (
    get_spatial_relation_enhanced,
    verify_category_with_clip,
)
from vlm_utils.minicpm_local import MiniCPMVLM
from PIL import Image


# ============================================================================
# 数据结构
# ============================================================================

class SGNode:
    """场景图节点"""
    def __init__(self, obj_id, category, center, bbox, image_crop=None, masked_crop=None, view_score=0.0):
        self.obj_id = obj_id
        self.category = category
        self.description = ""
        self.center = center          # [x, y, z]
        self.bbox = bbox              # [[min_x,min_y,min_z], [max_x,max_y,max_z]]
        self.seen_classes = {category: 1}  # {"chair": 4, "sofa": 1}
        self.best_text_sim = -1.0
        self.image_crop = image_crop
        self.masked_crop = masked_crop
        self.view_score = view_score


class SGEdge:
    """场景图边"""
    def __init__(self, obj1, obj2, spatial_relation, distance, contact_relation):
        self.obj1 = obj1
        self.obj2 = obj2
        self.spatial_relation = spatial_relation
        self.distance = distance
        self.contact_relation = contact_relation


# ============================================================================
# StreamSceneGraph 类
# ============================================================================

class StreamSceneGraph:
    """流式场景图状态管理"""

    def __init__(self, clip_model, clip_tokenizer, clip_preprocess, obj_classes, distance_threshold=2.5):
        """
        Args:
            clip_model: CLIP 模型实例（已加载到 GPU）
            clip_tokenizer: CLIP tokenizer
            clip_preprocess: CLIP 预处理函数
            obj_classes: ObjectClasses 实例
            distance_threshold: 关系距离阈值（米）
        """
        self._clip_model = clip_model
        self._clip_tokenizer = clip_tokenizer
        self._clip_preprocess = clip_preprocess
        self._obj_classes = obj_classes
        self._distance_threshold = distance_threshold

        self._nodes = {}               # obj_id → SGNode
        self._edges = []               # List[SGEdge]
        self._text_feat_cache = {}     # class_name → np.ndarray (CLIP text feature)
        self._version = 0

        # Moondream 延迟加载
        self._moondream_vlm = None
        self._moondream_config = {}

    # --- CLIP text embedding 缓存 ---

    def _get_text_feature(self, class_name):
        """CLIP text embedding 缓存。同一个类别名只编码一次。"""
        if class_name not in self._text_feat_cache:
            text_tokens = self._clip_tokenizer([f"a {class_name}"]).cuda()
            with torch.no_grad():
                text_feat = self._clip_model.encode_text(text_tokens)
            self._text_feat_cache[class_name] = text_feat.cpu().numpy().flatten()
        return self._text_feat_cache[class_name]

    # --- CLIP 类别名决策 ---

    def _resolve_class_name(self, node, obj):
        """
        用 CLIP image-text 相似度决定物体的最佳类别名。
        只有 seen_classes >= 2 种时才触发 CLIP 计算。
        """
        if len(node.seen_classes) < 2:
            # 只有一种类别，直接用
            node.category = list(node.seen_classes.keys())[0]
            return

        # 取物体的累积 CLIP image feature
        image_feat = obj.clip_feature.flatten()
        image_feat = image_feat / (np.linalg.norm(image_feat) + 1e-8)

        best_class = None
        best_sim = -1.0
        for class_name in node.seen_classes:
            text_feat = self._get_text_feature(class_name)
            text_feat = text_feat / (np.linalg.norm(text_feat) + 1e-8)
            sim = float(np.dot(image_feat, text_feat))
            if sim > best_sim:
                best_sim = sim
                best_class = class_name

        node.category = best_class
        node.best_text_sim = best_sim

    # --- 几何关系计算 ---

    @staticmethod
    def _compute_bbox(obj):
        """从 ObjectMemory.points_3d 计算 3D bbox"""
        if len(obj.points_3d) == 0:
            return [[0, 0, 0], [0, 0, 0]]
        min_b = np.round(np.min(obj.points_3d[:, :3], axis=0), 3).tolist()
        max_b = np.round(np.max(obj.points_3d[:, :3], axis=0), 3).tolist()
        return [min_b, max_b]

    def _compute_edges_for(self, obj, all_objects):
        """计算 obj 与 all_objects 中所有其他物体的几何关系"""
        edges = []
        for other in all_objects:
            if other.id == obj.id:
                continue
            dist = np.linalg.norm(obj.center_3d - other.center_3d)
            if dist > self._distance_threshold:
                continue
            relation = get_spatial_relation_enhanced(
                obj.center_3d, other.center_3d,
                self._compute_bbox(obj), self._compute_bbox(other)
            )
            edges.append(SGEdge(
                obj1=obj.id, obj2=other.id,
                spatial_relation=relation['spatial'],
                distance=relation['distance'],
                contact_relation=relation['contact'],
            ))
        return edges

    def _recompute_edges_for(self, obj_id, all_objects):
        """先删旧边，再加新边"""
        self._edges = [e for e in self._edges
                       if e.obj1 != obj_id and e.obj2 != obj_id]
        obj = next((o for o in all_objects if o.id == obj_id), None)
        if obj:
            self._edges.extend(self._compute_edges_for(obj, all_objects))

    # --- 事件处理 ---

    def on_frame_processed(self, objects, events):
        """每帧调用：处理 4 类事件（added/merged/removed/updated）"""

        # 1. 新增物体 → 创建节点 + 计算新边
        for obj in events.get('added', []):
            node = SGNode(
                obj_id=obj.id,
                category=obj.class_name,
                center=np.round(obj.center_3d, 3).tolist(),
                bbox=self._compute_bbox(obj),
                image_crop=obj.image_crop,
                masked_crop=getattr(obj, 'masked_crop', None),
                view_score=obj.view_score,
            )
            self._nodes[obj.id] = node
            self._edges.extend(self._compute_edges_for(obj, objects))

        # 2. 合并事件 → 合并 seen_classes + 删除被吸收节点
        for merge_info in events.get('merged', []):
            absorbed_id = merge_info['absorbed_id']
            absorber_id = merge_info['absorber_id']
            if absorbed_id in self._nodes and absorber_id in self._nodes:
                absorber_node = self._nodes[absorber_id]
                absorbed_node = self._nodes[absorbed_id]
                for cls, cnt in absorbed_node.seen_classes.items():
                    absorber_node.seen_classes[cls] = (
                        absorber_node.seen_classes.get(cls, 0) + cnt
                    )
                # 重新决策类别
                absorber_obj = next((o for o in objects if o.id == absorber_id), None)
                if absorber_obj:
                    self._resolve_class_name(absorber_node, absorber_obj)
            # 删除被吸收的节点和边
            if absorbed_id in self._nodes:
                del self._nodes[absorbed_id]
            self._edges = [e for e in self._edges
                           if e.obj1 != absorbed_id and e.obj2 != absorbed_id]

        # 3. 删除物体 → 删除节点 + 清理边
        for obj in events.get('removed', []):
            obj_id = obj.id
            if obj_id in self._nodes:
                del self._nodes[obj_id]
            self._edges = [e for e in self._edges
                           if e.obj1 != obj_id and e.obj2 != obj_id]

        # 4. 匹配更新（类别名可能变了）→ CLIP 类别决策 + 重算边
        for upd in events.get('updated', []):
            obj_id = upd['obj_id']
            new_class = upd['new_class']
            obj = upd['obj']

            node = self._nodes.get(obj_id)
            if node is None:
                continue

            # 更新 seen_classes
            node.seen_classes[new_class] = node.seen_classes.get(new_class, 0) + 1

            # 更新 crop（视角更好时）
            if obj.view_score > node.view_score:
                node.view_score = obj.view_score
                node.image_crop = obj.image_crop
                node.masked_crop = getattr(obj, 'masked_crop', None)

            # CLIP 类别名决策
            self._resolve_class_name(node, obj)

            # 更新 center / bbox
            node.center = np.round(obj.center_3d, 3).tolist()
            node.bbox = self._compute_bbox(obj)

            # 重算该物体的边
            self._recompute_edges_for(obj_id, objects)

        self._version += 1

    # --- 快照 ---

    def get_snapshot(self):
        """返回当前场景图状态（给下游大模型用）"""
        nodes = []
        for n in self._nodes.values():
            nodes.append({
                'idx': n.obj_id,
                'category': n.category,
                'description': n.description,
                'center': n.center,
                'bbox': n.bbox,
            })
        edges = []
        for e in self._edges:
            edges.append({
                'obj1': e.obj1,
                'obj2': e.obj2,
                'spatial_relation': e.spatial_relation,
                'distance': e.distance,
                'contact_relation': e.contact_relation,
            })
        return {
            'nodes': nodes,
            'edges': edges,
            'stats': {
                'total_nodes': len(self._nodes),
                'total_edges': len(self._edges),
            }
        }

    # --- finish() 相关 ---

    def run_vlm_descriptions(self, objects, moondream_config):
        """finish() 调用：Moondream VLM 统一生成描述"""
        # 延迟加载 Moondream
        if self._moondream_vlm is None:
            model_path = moondream_config.get('moondream_model_path', './models/moondream2')
            load_4bit = moondream_config.get('moondream_4bit', False)
            print(f"[VLM] Loading Moondream from {model_path}...")
            self._moondream_vlm = MiniCPMVLM(model_path=model_path, load_in_4bit=load_4bit)
            print("[VLM] Moondream loaded.")

        category_prompt = (
            "What is this object? Answer with exactly ONE word: the object category.\n"
            "Examples: chair, table, monitor, bottle, box, wall, door, pillow, plant, cup, keyboard, trash can.\n"
            "Answer with ONLY the category name, nothing else."
        )
        description_prompt = (
            "Describe this object in one sentence for identification.\n"
            "Include: color, material, and distinctive features.\n"
            "Examples: 'a white ceramic mug with a handle', "
            "'a black leather office chair with armrests', "
            "'a brown wooden rectangular table'.\n"
            "Answer in under 20 words."
        )

        obj_by_id = {o.id: o for o in objects}

        for node in list(self._nodes.values()):
            # 检查强制停止标志
            if getattr(self, '_stop_event', None) and self._stop_event.is_set():
                print("[VLM] 收到停止信号，中断描述生成")
                return

            obj = obj_by_id.get(node.obj_id)
            if obj is None:
                continue

            # 取 masked_crop 或 image_crop
            crop = getattr(obj, 'masked_crop', None)
            if crop is None:
                crop = getattr(obj, 'image_crop', None)
            if crop is None:
                continue

            image_pil = Image.fromarray(np.array(crop))

            # 类别预测
            category_response = self._moondream_vlm.generate_content(image_pil, category_prompt)
            category = category_response.strip().rstrip('.')
            category = category.strip('[]{}()')

            # CLIP 验证
            verified = verify_category_with_clip(
                crop, category, self._clip_model,
                self._clip_preprocess, self._clip_tokenizer
            )
            if verified is None:
                category = node.category  # fallback 到 CLIP 校准后的 YOLO 类名

            # 场景描述词 fallback
            scene_keywords = ['scene', 'image', 'indoor', 'outdoor', 'room', 'view', 'picture', 'photo', 'the', 'this', 'it']
            if any(kw == category.lower() for kw in scene_keywords):
                category = node.category

            # 描述预测
            description = self._moondream_vlm.generate_content(image_pil, description_prompt).strip()

            node.category = category
            node.description = description
            print(f"  [VLM] obj{node.obj_id}: category='{category}', desc='{description[:50]}'")

    def finalize_relations(self, objects):
        """finish() 调用：清空所有边，基于最终 bbox 全量重算"""
        self._edges = []
        for i, obj_i in enumerate(objects):
            for obj_j in objects[i + 1:]:
                dist = np.linalg.norm(obj_i.center_3d - obj_j.center_3d)
                if dist > self._distance_threshold:
                    continue
                relation = get_spatial_relation_enhanced(
                    obj_i.center_3d, obj_j.center_3d,
                    self._compute_bbox(obj_i), self._compute_bbox(obj_j)
                )
                self._edges.append(SGEdge(
                    obj1=obj_i.id, obj2=obj_j.id,
                    spatial_relation=relation['spatial'],
                    distance=relation['distance'],
                    contact_relation=relation['contact'],
                ))
        print(f"  [Finalize] {len(self._edges)} edges computed from {len(objects)} objects")

    def save_json(self, output_dir):
        """写 scene_graph.json（Viewer 兼容格式）"""
        snapshot = self.get_snapshot()
        graph_path = os.path.join(output_dir, "scene_graph.json")
        with open(graph_path, "w") as f:
            json.dump(snapshot, f, indent=4)
        print(f"Saved scene graph ({len(snapshot['nodes'])} nodes, {len(snapshot['edges'])} edges) to {graph_path}")

    def save_crops(self, output_dir):
        """写 objects_img_crop/{idx}.jpg"""
        crop_dir = os.path.join(output_dir, "objects_img_crop")
        os.makedirs(crop_dir, exist_ok=True)
        for node in self._nodes.values():
            if node.image_crop is not None:
                crop_bgr = cv2.cvtColor(np.array(node.image_crop), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(crop_dir, f"{node.obj_id}.jpg"), crop_bgr)
        print(f"Saved {len(self._nodes)} crops to {crop_dir}")
