"""StreamingDGSG 配置文件 — lingbot-map 流式管线专用

与 dgsg_refactor_lingbot.py 的 config 格式完全兼容。
模型路径从环境变量推导，与现有部署规范一致。
"""
import os

_STMEM_HOME = os.environ.get("STMEM_HOME",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
_MODEL_DIR = os.environ.get("STMEM_DGSG_MODEL_DIR", os.path.join(_STMEM_HOME, "models"))
_WORKDIR = os.environ.get("STMEM_DGSG_EXP_DIR",
    os.path.join(_STMEM_HOME, "backend", "dgsg", "experiments", "mydata"))

config = {
    "workdir": _WORKDIR,
    "run_name": "stream",
    "seed": 42,
    "primary_device": "cuda" if __import__('torch').cuda.is_available() else "cpu",
    "data": {
        "basedir": "",
        "sequence": "stream",
        "desired_image_height": 378,
        "desired_image_width": 518,
        "start": 0,
        "end": -1,
        "stride": 1,
        "num_frames": -1,
    },
    "lang": {
        "yolo_model_path": os.path.join(_MODEL_DIR, "yolov8s-world.pt"),
        "sam_model_path": os.path.join(_MODEL_DIR, "sam2.1_b.pt"),
        "clip_model_path": os.path.join(_MODEL_DIR, "open_clip_pytorch_model.bin"),
        "moondream_model_path": os.path.join(_MODEL_DIR, "moondream2"),
        "classes_file": os.path.join(_STMEM_HOME, "backend", "dgsg", "configs", "mydata", "office1_classes.txt"),
        "bg_classes": ["wall", "floor", "ceiling", "door", "window", "cabinet", "counter", "curtain"],
        "skip_bg": False,
        "relation_distance_threshold": 2.5,
        "mask_conf_threshold": 0.4,
        "max_bbox_area_ratio": 0.6,
        "mask_area_threshold": 500,
        "box_overlap_threshold": 0.95,
    },
    "viz": {
        "variables_path": "",
        "keyframe_list_path": "",
    },
}
