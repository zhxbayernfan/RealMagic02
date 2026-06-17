"""3D Scene Graph Point Cloud Viewer - Flask Backend"""

import os
import time
import argparse
import pickle
import gzip
import json
import traceback
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify, Response, abort

BASE_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"

app = Flask(__name__)

_data_cache = {}
_cache_mtime = {}


def _key_files(exp_dir):
    """Return list of files whose mtime determines cache validity."""
    return [
        exp_dir / "params_with_idx.npz",
        exp_dir / "scene_graph.json",
        exp_dir / "objects.pkl.gz",
    ]


def _is_cache_stale(exp_path):
    """Check if any key file was modified after caching."""
    if exp_path not in _cache_mtime:
        return True
    exp_dir = EXPERIMENTS_DIR / exp_path
    for f in _key_files(exp_dir):
        if f.exists() and f.stat().st_mtime > _cache_mtime[exp_path]:
            return True
    return False


def find_experiments():
    """Scan experiments/ for directories containing params_with_idx.npz."""
    results = []
    if not EXPERIMENTS_DIR.exists():
        return results
    for root, dirs, files in os.walk(EXPERIMENTS_DIR):
        if "params_with_idx.npz" in files:
            rel = os.path.relpath(root, EXPERIMENTS_DIR)
            results.append(rel)
    return sorted(results)


def _load_experiment(exp_path):
    """Load and cache experiment data. Returns (data_dict, error_msg)."""
    if exp_path in _data_cache and not _is_cache_stale(exp_path):
        return _data_cache[exp_path], None

    exp_dir = EXPERIMENTS_DIR / exp_path
    if not exp_dir.exists():
        return None, f"Experiment directory not found: {exp_path}"

    npz_path = exp_dir / "params_with_idx.npz"
    sg_path = exp_dir / "scene_graph.json"
    obj_path = exp_dir / "objects.pkl.gz"

    if not npz_path.exists():
        return None, f"Point cloud file not found: params_with_idx.npz"

    # Load point cloud
    try:
        data = np.load(npz_path)
        required_keys = {"means3D", "rgb_colors", "object_idx"}
        missing = required_keys - set(data.keys())
        if missing:
            return None, f"Point cloud format error: missing keys {missing} in params_with_idx.npz"
        means3D = data["means3D"].astype(np.float32)
        rgb_colors = data["rgb_colors"].astype(np.float32)
        object_idx = data["object_idx"]
        if means3D.ndim != 2 or means3D.shape[1] != 3:
            return None, f"Point cloud format error: means3D shape should be (N,3), got {means3D.shape}"
    except Exception as e:
        return None, f"Failed to load point cloud: {e}"

    points = {}
    for uid in np.unique(object_idx):
        uid = int(uid)
        mask = object_idx == uid
        points[uid] = {
            "xyz": np.ascontiguousarray(means3D[mask]),
            "rgb": np.ascontiguousarray(rgb_colors[mask]),
            "count": int(mask.sum()),
        }

    # Load scene graph
    scene_graph = None
    if sg_path.exists():
        try:
            with open(sg_path) as f:
                scene_graph = json.load(f)
            if not isinstance(scene_graph, dict) or "nodes" not in scene_graph:
                return None, f"Scene graph format error: expected {{'nodes': [...], 'edges': [...]}}"
        except json.JSONDecodeError:
            return None, f"Scene graph format error: invalid JSON in scene_graph.json"
    else:
        return None, f"Scene graph not found: scene_graph.json"

    # Load objects metadata
    objects_meta = []
    has_crops = os.path.isdir(exp_dir / "objects_img_crop")
    if obj_path.exists():
        try:
            with gzip.open(obj_path, "rb") as f:
                objects_list = pickle.load(f)
        except Exception as e:
            return None, f"Failed to load objects: {e}"

        node_map = {}
        for node in scene_graph.get("nodes", []):
            node_map[node["idx"]] = node
        for obj in objects_list:
            idx = obj["idx"]
            meta = {
                "id": idx,
                "category": obj.get("category", ""),
                "description": obj.get("description", ""),
                "point_count": points.get(idx, {}).get("count", 0),
                "has_crop": has_crops and os.path.exists(exp_dir / "objects_img_crop" / f"{idx}.jpg"),
            }
            if idx in node_map:
                n = node_map[idx]
                meta["center"] = n.get("center")
                meta["bbox"] = n.get("bbox")
            objects_meta.append(meta)

    # Add background entry for obj_idx=0 points
    if 0 in points:
        objects_meta.insert(0, {
            "id": 0,
            "category": "__background__",
            "description": "",
            "point_count": points[0]["count"],
            "has_crop": False,
        })

    cache = {
        "points": points,
        "scene_graph": scene_graph,
        "objects_meta": objects_meta,
        "exp_dir": str(exp_dir),
    }
    _data_cache[exp_path] = cache
    _cache_mtime[exp_path] = time.time()
    return cache, None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/experiments")
def api_experiments():
    return jsonify(find_experiments())


@app.route("/api/check")
def api_check():
    """Check if an experiment has all required files, return error if not."""
    exp = request.args.get("exp", "")
    data, error = _load_experiment(exp)
    if error:
        return jsonify({"ok": False, "error": error})
    return jsonify({"ok": True, "object_count": len(data["objects_meta"])})


@app.route("/api/scene_graph")
def api_scene_graph():
    exp = request.args.get("exp", "")
    data, error = _load_experiment(exp)
    if error:
        return jsonify({"error": error}), 400
    return jsonify(data["scene_graph"])


@app.route("/api/objects")
def api_objects():
    exp = request.args.get("exp", "")
    data, error = _load_experiment(exp)
    if error:
        return jsonify({"error": error}), 400
    return jsonify(data["objects_meta"])


@app.route("/api/points/<int:obj_id>")
def api_points(obj_id):
    exp = request.args.get("exp", "")
    data, error = _load_experiment(exp)
    if error:
        return jsonify({"error": error}), 400
    obj = data["points"].get(obj_id)
    if obj is None:
        abort(404)
    xyz = obj["xyz"]
    rgb = obj["rgb"]
    n = np.uint32(xyz.shape[0])
    buf = n.tobytes() + xyz.tobytes() + rgb.tobytes()
    return Response(buf, mimetype="application/octet-stream")


@app.route("/api/crops/<int:obj_id>.jpg")
def api_crop(obj_id):
    exp = request.args.get("exp", "")
    data, error = _load_experiment(exp)
    if error:
        abort(404)
    crop_path = os.path.join(data["exp_dir"], "objects_img_crop", f"{obj_id}.jpg")
    if not os.path.exists(crop_path):
        abort(404)
    return Response(open(crop_path, "rb").read(), mimetype="image/jpeg")


@app.route("/api/overlap")
def api_overlap():
    """Compute point cloud overlap between two objects using voxel hashing."""
    exp = request.args.get("exp", "")
    id1 = request.args.get("id1", type=int)
    id2 = request.args.get("id2", type=int)
    if id1 is None or id2 is None:
        return jsonify({"error": "id1 and id2 required"}), 400

    data, error = _load_experiment(exp)
    if error:
        return jsonify({"error": error}), 400

    obj1 = data["points"].get(id1)
    obj2 = data["points"].get(id2)
    if obj1 is None or obj2 is None:
        return jsonify({"error": "Object not found"}), 404

    VOXEL_SIZE = 0.02
    INV = 1.0 / VOXEL_SIZE

    def _voxel_hash(xyz_row):
        xi = int(xyz_row[0] * INV)
        yi = int(xyz_row[1] * INV)
        zi = int(xyz_row[2] * INV)
        return xi * 73856093 ^ yi * 19349669 ^ zi * 83492791

    # Build voxel set for object 2
    xyz2 = obj2["xyz"]
    vox2 = set()
    for i in range(xyz2.shape[0]):
        vox2.add(_voxel_hash(xyz2[i]))

    # Check object 1 points against voxel set
    xyz1 = obj1["xyz"]
    overlap_indices_1 = []
    for i in range(xyz1.shape[0]):
        if _voxel_hash(xyz1[i]) in vox2:
            overlap_indices_1.append(i)

    # Build voxel set for object 1 and check object 2
    vox1 = set()
    for i in range(xyz1.shape[0]):
        vox1.add(_voxel_hash(xyz1[i]))

    overlap_indices_2 = []
    for i in range(xyz2.shape[0]):
        if _voxel_hash(xyz2[i]) in vox1:
            overlap_indices_2.append(i)

    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]

    return jsonify({
        "overlap_ratio_1": len(overlap_indices_1) / n1 if n1 > 0 else 0,
        "overlap_ratio_2": len(overlap_indices_2) / n2 if n2 > 0 else 0,
        "overlap_count_1": len(overlap_indices_1),
        "overlap_count_2": len(overlap_indices_2),
        "total_1": n1,
        "total_2": n2,
        "indices_1": overlap_indices_1,
        "indices_2": overlap_indices_2,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    print(f"Viewer running at http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)
