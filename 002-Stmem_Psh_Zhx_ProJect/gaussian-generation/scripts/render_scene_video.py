"""
渲染场景视频：三列横排布局
  第一列：原始 RGB + 场景图叠加（节点 category + 边 pretential_relation，仅未遮挡物体）
  第二列：物体点云投影（原色）
  第三列：纯色物体投影

用法：
  python scripts/render_scene_video.py experiments/mydata/office1_static1 [--output video.mp4] [--fps 5]
"""

import argparse
import os
import json
import glob
import subprocess
import shutil
import tempfile

import cv2
import numpy as np
import yaml
from natsort import natsorted
from tqdm import tqdm


def load_poses(pose_dir):
    pose_files = natsorted(glob.glob(os.path.join(pose_dir, "*.txt")))
    poses = []
    for f in pose_files:
        values = []
        with open(f) as fh:
            for line in fh:
                values.extend(map(float, line.split()))
        poses.append(np.array(values).reshape(4, 4))
    return poses


def load_intrinsics(yaml_path):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cp = cfg["camera_params"]
    K = np.array([
        [cp["fx"], 0, cp["cx"]],
        [0, cp["fy"], cp["cy"]],
        [0, 0, 1],
    ], dtype=np.float64)
    H, W = cp["image_height"], cp["image_width"]
    return K, H, W


def make_object_colors(obj_ids):
    """黄金比例分布的 HSV 纯色，视觉区分度好。"""
    unique_ids = np.unique(obj_ids)
    unique_ids = unique_ids[unique_ids > 0]
    colors = {}
    for uid in unique_ids:
        hue = (uid * 0.618033988749895) % 1.0
        h_i = int(hue * 6)
        f = hue * 6 - h_i
        s, v = 0.85, 0.95
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        rgb_map = {
            0: (v, t, p), 1: (q, v, p), 2: (p, v, t),
            3: (p, q, v), 4: (t, p, v), 5: (v, p, q),
        }
        r, g, b = rgb_map[h_i % 6]
        colors[int(uid)] = (int(b * 255), int(g * 255), int(r * 255))  # BGR for OpenCV
    return colors


def project_object_points(means3D, rgb_colors, obj_ids, c2w, K, H, W):
    """投影物体点云（obj_id > 0），一次投影返回所有渲染所需数据。"""
    mask_obj = obj_ids > 0
    pts = means3D[mask_obj]
    ids = obj_ids[mask_obj]
    rgbs = rgb_colors[mask_obj]

    if len(pts) == 0:
        empty2d = np.zeros((0, 2), np.int32)
        return empty2d, np.zeros(0, np.float64), np.zeros(0, np.int32), np.zeros((0, 3), np.uint8)

    w2c = np.linalg.inv(c2w)
    R, T = w2c[:3, :3], w2c[:3, 3]
    pts_cam = (R @ pts.T).T + T

    valid_z = pts_cam[:, 2] > 0.01
    pts_cam = pts_cam[valid_z]
    ids = ids[valid_z]
    rgbs = rgbs[valid_z]

    if len(pts_cam) == 0:
        empty2d = np.zeros((0, 2), np.int32)
        return empty2d, np.zeros(0, np.float64), np.zeros(0, np.int32), np.zeros((0, 3), np.uint8)

    pts_proj = (K @ pts_cam.T).T
    pts_2d = np.rint(pts_proj[:, :2] / pts_proj[:, 2:3]).astype(np.int32)
    depths = pts_cam[:, 2]

    in_frame = (
        (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W) &
        (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H)
    )
    rgb_bgr = (rgbs[in_frame, ::-1] * 255).astype(np.uint8)
    return pts_2d[in_frame], depths[in_frame], ids[in_frame], rgb_bgr


def render_point_cloud(pts_2d, depths, colors_per_point, H, W, point_size=1):
    """Painter's algorithm 渲染点云到黑色背景。"""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    if len(pts_2d) == 0:
        return canvas

    order = np.argsort(-depths)
    xs = pts_2d[order, 0]
    ys = pts_2d[order, 1]
    cols = colors_per_point[order]

    if point_size <= 1:
        canvas[ys, xs] = cols
    else:
        r = point_size // 2
        for x, y, c in zip(xs, ys, cols):
            cv2.circle(canvas, (int(x), int(y)), r, (int(c[0]), int(c[1]), int(c[2])), -1)
    return canvas


def get_visible_objects(obj_ids, threshold=30):
    """统计当前帧每个物体的可见点数，返回可见物体 id 集合。"""
    if len(obj_ids) == 0:
        return set()
    unique, counts = np.unique(obj_ids, return_counts=True)
    return set(int(u) for u, c in zip(unique, counts) if c >= threshold)


def project_center(center_3d, c2w, K):
    """投影单个 3D 中心点到 2D，同时返回相机坐标系深度。"""
    w2c = np.linalg.inv(c2w)
    R, T = w2c[:3, :3], w2c[:3, 3]
    pt_cam = R @ center_3d + T
    if pt_cam[2] <= 0.01:
        return None, None
    pt_proj = K @ pt_cam
    px = int(round(pt_proj[0] / pt_proj[2]))
    py = int(round(pt_proj[1] / pt_proj[2]))
    return (px, py), pt_cam[2]


def filter_unoccluded_objects(visible_ids, pts_2d, depths, c2w, K, H, W, node_map,
                              occlude_thr=0.5):
    """基于 z-buffer 遮挡判定，返回离镜头最近且未被遮挡的物体 id 集合。"""
    if len(pts_2d) == 0 or not visible_ids:
        return set()

    # 构建深度图：每个像素取最近深度（向量化）
    depth_map = np.full((H, W), np.inf, dtype=np.float64)
    order = np.argsort(depths)  # 近处优先覆盖
    xs, ys, ds = pts_2d[order, 0], pts_2d[order, 1], depths[order]
    # 只保留每个像素最近的点（用 min 赋值）
    for x, y, d in zip(xs, ys, ds):
        if d < depth_map[y, x]:
            depth_map[y, x] = d

    # 检查每个可见物体的中心是否被遮挡
    unoccluded = set()
    for nid in visible_ids:
        if nid not in node_map:
            continue
        center = np.array(node_map[nid]["center"], dtype=np.float64)
        pt, center_depth = project_center(center, c2w, K)
        if pt is None:
            continue
        px, py = pt
        if not (0 <= px < W and 0 <= py < H):
            continue
        nearest_depth = depth_map[py, px]
        if nearest_depth == np.inf:
            continue
        # 中心深度与最近表面差距 < 阈值 → 未被遮挡
        if center_depth - nearest_depth < occlude_thr:
            unoccluded.add(nid)
    return unoccluded


def draw_label(img, text, pos, font_scale=0.4, thickness=1, bg_color=(0, 0, 0),
               text_color=(255, 255, 255)):
    """在 pos 处画带半透明背景的文字标签。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    x = max(0, min(x, img.shape[1] - tw - 4))
    y = max(th + 2, min(y, img.shape[0] - 4))

    overlay = img.copy()
    cv2.rectangle(overlay, (x - 2, y - th - 2), (x + tw + 2, y + baseline + 2), bg_color, -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_scene_graph(img, scene_graph, visible_ids, c2w, K, obj_colors):
    """在 RGB 图上叠加场景图：节点 category + 边 pretential_relation（仅未遮挡物体）。"""
    node_map = {n["idx"]: n for n in scene_graph["nodes"]}

    # 投影可见节点的中心
    projected = {}
    for nid in visible_ids:
        if nid in node_map:
            center = np.array(node_map[nid]["center"], dtype=np.float64)
            pt, _ = project_center(center, c2w, K)
            if pt is not None and 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
                projected[nid] = pt

    # 画边
    for edge in scene_graph["edges"]:
        o1, o2 = edge["obj1"], edge["obj2"]
        if o1 in projected and o2 in projected:
            p1, p2 = projected[o1], projected[o2]
            cv2.line(img, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            rel_text = edge.get("pretential_relation", "")
            if rel_text:
                draw_label(img, rel_text, (mx, my), font_scale=0.28, thickness=1,
                           bg_color=(60, 60, 60))

    # 画节点标签
    for nid, pt in projected.items():
        cat = node_map[nid].get("category", f"obj{nid}")
        color = obj_colors.get(nid, (200, 200, 200))
        cv2.circle(img, pt, 4, color, -1, cv2.LINE_AA)
        draw_label(img, cat, (pt[0] + 6, pt[1] - 4), font_scale=0.35, thickness=1,
                   bg_color=color)


def encode_video_ffmpeg(tmp_dir, out_path, fps, n_frames):
    """用 ffmpeg 将临时目录中的帧图片编码为 H.264 MP4。"""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(tmp_dir, "%06d.png"),
        "-frames:v", str(n_frames),
        "-c:v", "libx264", "-profile:v", "high", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-movflags", "+faststart",
        out_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"ffmpeg stderr: {result.stderr.decode()[-500:]}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="渲染场景视频")
    parser.add_argument("experiment_dir", help="实验输出目录，如 experiments/mydata/office1_static1")
    parser.add_argument("--output", default=None, help="输出视频路径（默认 <experiment_dir>/scene_video.mp4）")
    parser.add_argument("--fps", type=int, default=5, help="视频帧率（默认 5）")
    parser.add_argument("--min_points", type=int, default=30, help="物体可见最少投影点数（默认 30）")
    parser.add_argument("--point_size", type=int, default=1, help="点云渲染点大小（默认 1）")
    parser.add_argument("--occlude_thr", type=float, default=0.5, help="遮挡判定深度阈值（米，默认 0.5）")
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.experiment_dir)
    parts = exp_dir.rstrip("/").split(os.sep)
    scene_name = parts[-1]
    group_name = parts[-2]
    data_dir = os.path.join("data", group_name, scene_name)
    print(f"实验目录: {exp_dir}")
    print(f"数据目录: {data_dir}")

    # --- 加载数据 ---
    print("加载点云...")
    npz = np.load(os.path.join(exp_dir, "params_with_idx.npz"))
    means3D = npz["means3D"]
    rgb_colors = npz["rgb_colors"]
    obj_ids = npz["object_idx"].astype(np.int32)
    print(f"  点数: {len(means3D)}, 物体点: {np.sum(obj_ids > 0)}")

    print("加载场景图...")
    with open(os.path.join(exp_dir, "scene_graph.json")) as f:
        scene_graph = json.load(f)
    node_map = {n["idx"]: n for n in scene_graph["nodes"]}
    print(f"  节点: {len(scene_graph['nodes'])}, 边: {len(scene_graph['edges'])}")

    print("加载内参...")
    K, H, W = load_intrinsics(os.path.join(data_dir, "intrinsics.yaml"))
    print(f"  分辨率: {W}x{H}")

    print("加载位姿...")
    poses = load_poses(os.path.join(data_dir, "poses"))
    first_inv = np.linalg.inv(poses[0])
    poses = [first_inv @ p for p in poses]
    print(f"  位姿数: {len(poses)}")

    print("加载 RGB 图像列表...")
    rgb_files = natsorted(glob.glob(os.path.join(data_dir, "rgb", "*.png")))
    n_frames = min(len(rgb_files), len(poses))
    print(f"  总帧数: {n_frames}")

    obj_solid_colors = make_object_colors(obj_ids)
    print(f"  物体数: {len(obj_solid_colors)}")

    # --- 输出设置 ---
    out_path = args.output or os.path.join(exp_dir, "scene_video.mp4")
    out_W, out_H = W * 3, H
    has_ffmpeg = shutil.which("ffmpeg") is not None

    if has_ffmpeg:
        tmp_dir = tempfile.mkdtemp(prefix="dgsg_video_")
        print(f"输出视频: {out_path} ({out_W}x{out_H}, {args.fps}fps) [H.264 via ffmpeg]")
        print(f"临时帧目录: {tmp_dir}")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cv_writer = cv2.VideoWriter(out_path, fourcc, args.fps, (out_W, out_H))
        print(f"输出视频: {out_path} ({out_W}x{out_H}, {args.fps}fps) [mp4v]")
        print("  建议安装 ffmpeg 以获得 Mac 兼容的 H.264 编码: apt install ffmpeg")

    # --- 逐帧渲染 ---
    print("开始渲染...")
    written_frames = 0
    for i in tqdm(range(n_frames), desc="渲染帧"):
        rgb = cv2.imread(rgb_files[i])
        if rgb is None:
            print(f"  跳过无法读取的帧: {rgb_files[i]}")
            continue

        c2w = poses[i]

        # 一次投影
        pts_2d, depths, ids, rgb_bgr = project_object_points(
            means3D, rgb_colors, obj_ids, c2w, K, H, W)

        # --- 第一行：RGB + 场景图 ---
        row1 = rgb.copy()
        cv2.putText(row1, f"Frame {i}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 1, cv2.LINE_AA)

        visible = get_visible_objects(ids, threshold=args.min_points)
        unoccluded = filter_unoccluded_objects(
            visible, pts_2d, depths, c2w, K, H, W, node_map,
            occlude_thr=args.occlude_thr)
        draw_scene_graph(row1, scene_graph, unoccluded, c2w, K, obj_solid_colors)

        # --- 第二行：原色点云投影 ---
        row2 = render_point_cloud(pts_2d, depths, rgb_bgr, H, W, args.point_size)
        cv2.putText(row2, "Point Cloud (RGB)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 1, cv2.LINE_AA)

        # --- 第三行：纯色物体投影（无场景图） ---
        if len(pts_2d) > 0:
            solid_bgr = np.array(
                [obj_solid_colors.get(int(oid), (128, 128, 128)) for oid in ids],
                dtype=np.uint8)
        else:
            solid_bgr = np.zeros((0, 3), dtype=np.uint8)
        row3 = render_point_cloud(pts_2d, depths, solid_bgr, H, W, args.point_size)
        cv2.putText(row3, "Instance Segmentation", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 1, cv2.LINE_AA)

        frame = np.concatenate([row1, row2, row3], axis=1)

        if has_ffmpeg:
            frame_path = os.path.join(tmp_dir, f"{written_frames:06d}.png")
            cv2.imwrite(frame_path, frame)
        else:
            cv_writer.write(frame)
        written_frames += 1

    # --- 编码输出 ---
    if has_ffmpeg:
        print(f"编码 H.264 视频 ({written_frames} 帧)...")
        ok = encode_video_ffmpeg(tmp_dir, out_path, args.fps, written_frames)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if not ok:
            print("ffmpeg 编码失败！")
    else:
        cv_writer.release()

    print(f"完成！视频已保存到: {out_path}")
    print(f"  总帧数: {written_frames}, 分辨率: {out_W}x{out_H}, FPS: {args.fps}")


if __name__ == "__main__":
    main()
