"""
StreamingDGSG: 流式 DGSG 管线
lingbot-map 每帧调用 process_frame()，获取点云+物体记忆快照。
所有帧处理完后调用 finish() 跑 VLM + 保存结果。
"""
import os
import sys
import time
import gzip
import pickle
import numpy as np
import torch
import open_clip

# --- sys.path 设置 ---
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SCRIPTS_DIR)
_SERVICE_DIR = os.path.dirname(_BASE_DIR)
sys.path.insert(0, _BASE_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _SERVICE_DIR)

from dgsg_refactor_lingbot import (
    ObjectMemory,
    add_lb_points_to_global_cloud,
    detect_objects,
    batch_project_objects,
    match_detections_to_objects,
    fuse_update_matched,
    add_new_objects,
    merge_nearby_objects,
    remove_stale_objects,
    save_results,
    save_memory,
    visualize_frame,
    MAX_POINTS,
    DEPTH_ERROR_THRESHOLD,
    COLOR_ERROR_THRESHOLD,
)
from scene_graph_stream import StreamSceneGraph
from utils.object_helpers import ObjectClasses
from utils.common_utils import seed_everything
from ultralytics import SAM, YOLO


class StreamingDGSG:
    """流式 DGSG 管线"""

    def __init__(self, config, save_every_n_frames=0, save_visualization=True):
        """
        Args:
            config: 配置字典（与 dgsg_refactor_lingbot.py 相同格式）
            save_every_n_frames: 每隔几帧写一次磁盘（0=不写）
            save_visualization: 是否保存每帧 2x4 可视化图（默认开启）
        """
        print("=" * 60)
        print("StreamingDGSG 初始化")
        print("=" * 60)
        total_start = time.time()

        seed_everything(config['seed'])

        self.config = config
        self.lang_config = config['lang']
        self.save_every_n_frames = save_every_n_frames
        self.save_visualization = save_visualization

        self.output_dir = os.path.join(config["workdir"], config["run_name"])
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = torch.device(config["primary_device"])

        # 1. 加载模型
        print("  [1] 加载模型...")
        self.obj_classes = ObjectClasses(
            classes_file_path=self.lang_config['classes_file'],
            bg_classes=self.lang_config['bg_classes'],
            skip_bg=self.lang_config['skip_bg']
        )
        self.yolo_model = YOLO(self.lang_config['yolo_model_path'])
        self.yolo_model.set_classes(self.obj_classes.get_classes_arr())
        self.sam_model = SAM(self.lang_config['sam_model_path'])
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained=self.lang_config['clip_model_path']
        )
        self.clip_model = self.clip_model.to(self.device).eval()
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

        # 清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. 初始化 DGSG 状态
        print("  [2] 初始化状态...")
        self.global_cloud = np.zeros((MAX_POINTS, 7), dtype=np.float32)
        self.current_count = 0
        self.objects = []

        # 3. 内参（稍后由 set_intrinsics 设置）
        self._intrinsics = None
        self._image_size = (
            config['data'].get('desired_image_height', 378),
            config['data'].get('desired_image_width', 518),
        )

        # 4. 创建 StreamSceneGraph
        distance_threshold = self.lang_config.get('relation_distance_threshold', 2.5)
        self.scene_graph = StreamSceneGraph(
            self.clip_model, self.clip_tokenizer, self.clip_preprocess,
            self.obj_classes, distance_threshold
        )

        init_time = time.time() - total_start
        print(f"  初始化完成: {init_time:.1f}s")
        print("=" * 60)

    def set_intrinsics(self, intrinsics):
        """设置相机内参 (3,3)。必须在第一个 process_frame 之前调用。"""
        self._intrinsics = intrinsics
        print(f"  内参已设置:\n{intrinsics}")

    def process_frame(self, rgb_image, depth_image, world_points, pose, frame_id=0):
        """
        处理单帧。

        Args:
            rgb_image:    (H, W, 3) uint8 RGB 图像
            depth_image:  (H, W) float32 深度图（米）
            world_points: (H_lb, W_lb, 3) float32 lingbot-map 世界坐标映射
            pose:         (4, 4) float64 camera-to-world 位姿（c2w）
            frame_id:     int 帧编号

        Returns:
            dict: {frame_id, point_cloud, objects, changes, stats}
        """
        assert self._intrinsics is not None, "请先调用 set_intrinsics()"
        frame_start = time.time()

        w2c = np.linalg.inv(pose)
        img_h, img_w = rgb_image.shape[:2]
        intrinsics = self._intrinsics

        # Step 0: 快照旧状态（用于检测变更）
        old_class_names = {obj.id: obj.class_name for obj in self.objects}

        # Step 1: 全局点云追加
        self.current_count = add_lb_points_to_global_cloud(
            world_points, rgb_image, self.global_cloud, self.current_count
        )

        # Step 2: 检测
        # 在检测前清理 CUDA 缓存，避免内存碎片
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        detections, det_timing = detect_objects(
            rgb_image, depth_image, self.yolo_model, self.sam_model,
            self.clip_model, self.clip_preprocess, self.obj_classes,
            self.lang_config, frame_id, intrinsics, w2c,
            world_points=world_points
        )

        # Step 3: 批量投影
        projected_masks = batch_project_objects(
            self.objects, intrinsics, w2c, img_h, img_w
        )

        # Step 4: 匹配
        match_result = match_detections_to_objects(
            detections, projected_masks, self.objects
        )

        # Step 5: 融合 + 新增 + 合并
        fuse_update_matched(self.objects, detections, match_result)
        objects_to_add = add_new_objects(self.objects, detections, match_result)
        merge_log = []  # merge 暂不启用

        # Step 6: 删除过时物体
        objects_to_remove = remove_stale_objects(
            self.objects, match_result, rgb_image, depth_image,
            intrinsics, w2c,
            DEPTH_ERROR_THRESHOLD, COLOR_ERROR_THRESHOLD,
            disappear_ratio_thr=0.5, frustum_ratio_thr=0.7, visibility_ratio_thr=0.7
        )

        # Step 6.5: 二次投影 + 可视化
        if self.save_visualization:
            updated_projected_masks = batch_project_objects(
                self.objects, intrinsics, w2c, img_h, img_w
            )
            visualize_frame(
                rgb_image, detections, self.objects, match_result,
                projected_masks, objects_to_add, objects_to_remove,
                updated_projected_masks, self.output_dir, frame_id
            )

        # Step 7: 检测变更 + 通知场景图
        updated_events = []
        for det_idx, obj_id in match_result['matched_pairs']:
            obj = next((o for o in self.objects if o.id == obj_id), None)
            if obj is None:
                continue
            old_name = old_class_names.get(obj_id, None)
            new_name = obj.class_name
            if old_name != new_name:
                updated_events.append({
                    'obj_id': obj_id,
                    'old_class': old_name,
                    'new_class': new_name,
                    'obj': obj,
                })

        events = {
            'added': objects_to_add,
            'removed': objects_to_remove,
            'updated': updated_events,
            'merged': merge_log,
        }
        self.scene_graph.on_frame_processed(self.objects, events)

        # Step 8: 构建返回值
        snapshot = self._build_snapshot(frame_id, events)

        # Step 9: 可选写磁盘
        if self.save_every_n_frames > 0 and frame_id % self.save_every_n_frames == 0:
            self._save_frame_results(snapshot, frame_id)

        # 日志
        frame_time = time.time() - frame_start
        sg = self.scene_graph.get_snapshot()
        print(f"  [Frame {frame_id}] objs={len(self.objects)}, pts={self.current_count}, "
              f"nodes={sg['stats']['total_nodes']}, edges={sg['stats']['total_edges']}, "
              f"+{len(objects_to_add)}/-{len(objects_to_remove)}/~{len(updated_events)}, "
              f"time={frame_time:.2f}s")

        return snapshot

    def _build_snapshot(self, frame_id, events=None):
        """构建返回 dict。

        所有数组通过 np.ascontiguousarray 强制独立连续内存，
        防止任何 view/slice 共享 global_cloud 或 obj.points_3d 的底层 buffer。
        """
        background = np.ascontiguousarray(self.global_cloud[:self.current_count])
        all_obj_points = [
            np.ascontiguousarray(obj.points_3d)
            for obj in self.objects if len(obj.points_3d) > 0
        ]

        if all_obj_points:
            point_cloud = np.vstack([background] + all_obj_points)
        else:
            point_cloud = background

        N = len(point_cloud)
        # 强制独立副本 — ascontiguousarray 确保拥有独立内存
        means = np.ascontiguousarray(point_cloud[:, :3])
        colors = np.ascontiguousarray(point_cloud[:, 3:6])
        obj_idx = np.ascontiguousarray(point_cloud[:, 6].astype(np.int32))

        # 运行时验证：检查颜色值范围
        c_min, c_max = float(colors.min()), float(colors.max())
        c_finite = bool(np.isfinite(colors).all())
        if not c_finite or c_min < -0.1 or c_max > 1.1:
            print(f"  [WARN] _build_snapshot frame={frame_id}: RGB range=[{c_min:.4f}, {c_max:.4f}], "
                  f"finite={c_finite}, N={N} — 颜色数据异常!")

        params_dict = {
            'means3D': means,
            'rgb_colors': colors,
            'object_idx': obj_idx,
            'log_scales': np.full((N, 3), -5.0, dtype=np.float32),
            'unnorm_rotations': np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            'logit_opacities': np.full((N, 1), 100, dtype=np.float32),
        }

        # 物体记忆列表（精简 dict）
        objects_list = []
        for obj in self.objects:
            objects_list.append({
                'idx': obj.id,
                'class_name': obj.class_name,
                'clip_feature': obj.clip_feature,
                'center_3d': obj.center_3d.tolist() if isinstance(obj.center_3d, np.ndarray) else obj.center_3d,
                'image_crops': obj.image_crop,
                'masked_crops': getattr(obj, 'masked_crop', None),
            })

        sg_snapshot = self.scene_graph.get_snapshot()

        # changes
        changes = {
            'added_obj_ids': [o.id for o in (events.get('added', []) if events else [])],
            'removed_obj_ids': [o.id for o in (events.get('removed', []) if events else [])],
            'updated_obj_ids': [u['obj_id'] for u in (events.get('updated', []) if events else [])],
        }

        return {
            'frame_id': frame_id,
            'point_cloud': params_dict,
            'objects': objects_list,
            'scene_graph': sg_snapshot,
            'changes': changes,
            'stats': {
                'total_objects': len(self.objects),
                'total_points': N,
                'global_cloud_points': self.current_count,  # 背景点云（单调递增）
            },
        }

    def _save_frame_results(self, snapshot, frame_id):
        """每帧写磁盘（覆盖写同一文件）"""
        # params_with_idx.npz
        save_path = os.path.join(self.output_dir, 'params_with_idx.npz')
        np.savez(save_path, **snapshot['point_cloud'])

        # objects.pkl.gz
        obj_save_path = os.path.join(self.output_dir, 'objects.pkl.gz')
        with gzip.open(obj_save_path, 'wb') as f:
            pickle.dump(snapshot['objects'], f)

        print(f"  [Save] Frame {frame_id}: saved to {self.output_dir}")

    def finish(self):
        """所有帧处理完后调用：VLM + 全量重算关系 + 保存"""
        print("\n" + "=" * 60)
        print("finish(): VLM 描述 + 最终保存")
        print("=" * 60)

        # 1. 对所有节点做最终 CLIP 类别决策（确保基于完整 clip_feature）
        for node in self.scene_graph._nodes.values():
            obj = next((o for o in self.objects if o.id == node.obj_id), None)
            if obj:
                self.scene_graph._resolve_class_name(node, obj)

        # 1.5 释放不再需要的模型显存（YOLO/SAM），为 Moondream 腾空间
        print("\n[1.5/4] 释放检测模型显存...")
        if hasattr(self, 'yolo_model'):
            del self.yolo_model
        if hasattr(self, 'sam_model'):
            del self.sam_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("  已释放 YOLO/SAM，当前显存: "
                  f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB / "
                  f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("  已释放 YOLO/SAM")

        # 2. VLM 统一描述
        print("\n[2/4] 运行 VLM 描述生成...")
        self.scene_graph.run_vlm_descriptions(self.objects, self.lang_config)

        # 3. 全量重算关系
        print("\n[3/4] 全量重算几何关系...")
        self.scene_graph.finalize_relations(self.objects)

        # 4. 保存结果（先清空旧文件，避免残留）
        print("\n[4/4] 保存结果...")
        import shutil as _shutil
        if os.path.exists(self.output_dir):
            _shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        save_results(self.global_cloud, self.current_count, self.objects, self.output_dir)
        save_memory(self.global_cloud, self.current_count, self.objects, self.output_dir)
        self.scene_graph.save_json(self.output_dir)
        self.scene_graph.save_crops(self.output_dir)

        print("\n[4/4] 完成!")
        sg = self.scene_graph.get_snapshot()
        result = {
            'output_dir': self.output_dir,
            'total_objects': len(self.objects),
            'total_points': self.current_count,
            'scene_graph_nodes': sg['stats']['total_nodes'],
            'scene_graph_edges': sg['stats']['total_edges'],
        }
        print(f"  输出目录: {self.output_dir}")
        print(f"  物体数: {result['total_objects']}")
        print(f"  点云数: {result['total_points']}")
        print(f"  场景图: {result['scene_graph_nodes']} nodes, {result['scene_graph_edges']} edges")
        print("=" * 60)

        return result
