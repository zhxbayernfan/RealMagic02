"""
Moondream VLM 本地加载器 — 完全不走 ~/.cache

直接从 models/moondream2 目录 import 模型代码并加载 safetensors 权重，
不依赖 transformers 的 trust_remote_code / dynamic module 缓存机制。
"""

import os
import sys
import importlib
import torch
from pathlib import Path
from PIL import Image

# 将 models/ 目录加入 sys.path，以包方式 import moondream2
# 优先使用环境变量，部署时 models 可能在项目根目录而非 dgsg/ 下
_MODELS_DIR = os.environ.get("STMEM_DGSG_MODEL_DIR",
    str(Path(__file__).resolve().parent.parent.parent.parent / "models"))
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

# 确保 moondream2 包的 __init__.py 存在
_PKG_INIT = os.path.join(_MODELS_DIR, "moondream2", "__init__.py")
if not os.path.exists(_PKG_INIT):
    os.makedirs(os.path.dirname(_PKG_INIT), exist_ok=True)
    open(_PKG_INIT, "w").close()


def _load_safetensors(model_path):
    """从本地 safetensors 文件加载权重到模型"""
    from safetensors.torch import load_file
    safetensors_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Weights not found: {safetensors_path}")
    return load_file(safetensors_path)


class MoondreamVLM:
    def __init__(self, model_path="models/moondream2", device="cuda", load_in_4bit=False):
        self.device = device
        model_path = str(Path(model_path).resolve())

        print(f"Loading MoondreamVLM: {model_path} (4bit={load_in_4bit})...")

        # 以包方式 import，支持 moondream2 内部的相对 import
        hf_mod = importlib.import_module("moondream2.hf_moondream")
        HfMoondream = hf_mod.HfMoondream
        HfConfig = hf_mod.HfConfig

        # 构建 config + 模型
        config = HfConfig()
        model = HfMoondream(config)

        # 加载权重
        state_dict = _load_safetensors(model_path)
        model.load_state_dict(state_dict, strict=False)

        # 4bit 量化（可选）
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            # 需要通过 transformers 量化包装
            from transformers import quantize_model
            model = quantize_model(model, quantization_config)

        # 移到 GPU
        model = model.to(device)
        model.eval()
        self.model = model

        # tokenizer（从 moondream 内部获取）
        self.tokenizer = model.model.tokenizer

        print("MoondreamVLM loaded (fully local, no .cache).")

    def generate_content(self, image_input, prompt, max_new_tokens=200):
        """
        Generate content from image and prompt.

        Args:
            image_input: PIL Image
            prompt: Text prompt
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated text response
        """
        enc_image = self.model.encode_image(image_input)

        with torch.no_grad():
            response = self.model.answer_question(enc_image, prompt, self.tokenizer)

        return response.strip()
