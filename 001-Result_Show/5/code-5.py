#!/usr/bin/python3
"""
Test script for LocateAnything-3B model.
Locates objects in images based on natural language queries.

Usage:
    python Test_LocateAnything.py

Requirements:
    pip install transformers==4.46.0 torch torchvision pillow accelerate peft
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用 GPU 1

import re
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoProcessor

# ==================== 配置 ====================
model_path = "/home/zhanghexiang/LocateAnything-3B"
image_path = "/home/zhanghexiang/zhx/RealMagic02/001-Result_Show/original-5.jpeg"
question = "Find the text in the image."  # 自然语言查询
max_side = 1008  # 图片最大边长，控制显存占用
# ==============================================

# 加载处理器
print("加载处理器...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 加载模型
print("加载模型...")
device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES=1 使这里对应物理 GPU 1
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to(device).eval()
torch.cuda.reset_peak_memory_stats()

# 加载并缩放图片
print(f"加载图片: {image_path}")
image = Image.open(image_path).convert("RGB")
orig_w, orig_h = image.size
print(f"原始尺寸: {orig_w}x{orig_h}")

if max(orig_w, orig_h) > max_side:
    scale = max_side / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    print(f"缩放后尺寸: {new_w}x{new_h}")
else:
    new_w, new_h = orig_w, orig_h

# 构建输入
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question}
    ]
}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos = processor.process_vision_info(messages)

inputs = processor(
    text=[text],
    images=images,
    videos=videos,
    return_tensors="pt"
).to(model.device)

pixel_values = inputs["pixel_values"].to(torch.bfloat16)

# 生成输出
print("开始生成...")
with torch.no_grad():
    outputs = model.generate(
        pixel_values=pixel_values,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_grid_hws=inputs.get("image_grid_hws", None),
        tokenizer=processor,
        max_new_tokens=128,
        generation_mode="hybrid",
        do_sample=False,
        use_cache=True,
    )

# 输出结果
if isinstance(outputs, str):
    result = outputs
else:
    result = processor.decode(outputs[0], skip_special_tokens=True)
print("模型输出：", result)

# 解析坐标并绘制框体
print("\n解析坐标并生成带框体的图片...")
boxes = re.findall(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>', result)

if boxes:
    # 去重：将坐标相近的框合并（容差50像素）
    unique_boxes = []
    tol = 50
    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        is_dup = False
        for ux1, uy1, ux2, uy2 in unique_boxes:
            if abs(x1 - ux1) < tol and abs(y1 - uy1) < tol and abs(x2 - ux2) < tol and abs(y2 - uy2) < tol:
                is_dup = True
                break
        if not is_dup:
            unique_boxes.append((x1, y1, x2, y2))
    print(f"原始 {len(boxes)} 个框，去重后 {len(unique_boxes)} 个")

    # 过滤全图框：去掉面积超过图片90%的框
    image_area = new_w * new_h
    filtered_boxes = []
    for (x1, y1, x2, y2) in unique_boxes:
        box_area = (x2 - x1) * (y2 - y1)
        if box_area < 0.9 * image_area:
            filtered_boxes.append((x1, y1, x2, y2))
        else:
            print(f"已过滤全图框: ({x1},{y1},{x2},{y2}) 面积占比 {box_area/image_area:.1%}")
    unique_boxes = filtered_boxes
    print(f"过滤全图框后: {len(unique_boxes)} 个")

    # 坐标是相对于缩放后图片的，需要映射回原始尺寸
    scale_x = orig_w / new_w
    scale_y = orig_h / new_h

    # 加载原始图片
    orig_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(orig_image)

    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]
    for i, (x1, y1, x2, y2) in enumerate(unique_boxes):
        # 映射回原始尺寸
        x1_orig = int(float(x1) * scale_x)
        y1_orig = int(float(y1) * scale_y)
        x2_orig = int(float(x2) * scale_x)
        y2_orig = int(float(y2) * scale_y)

        color = colors[i % len(colors)]
        draw.rectangle([x1_orig, y1_orig, x2_orig, y2_orig], outline=color, width=3)
        draw.text((x1_orig, y1_orig - 25), "text", fill=color, font=font)

    output_path = "/home/zhanghexiang/zhx/RealMagic02/001-Result_Show/result-5.jpeg"
    orig_image.save(output_path, quality=95)
    print(f"已保存带框体的图片: {output_path}")
    print(f"共检测到 {len(boxes)} 个框")
else:
    print("未检测到任何框")
gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
print(f"\nGPU_MEM: {gpu_mem:.0f}MB")