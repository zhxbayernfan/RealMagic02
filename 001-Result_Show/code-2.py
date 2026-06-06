#!/usr/bin/python3
"""
Multi-Subject Detection — detect all players in a team photo.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
script_dir = os.path.dirname(os.path.abspath(__file__))

import re
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoProcessor

# ==================== 配置 ====================
model_path = "/home/zhanghexiang/LocateAnything-3B"
image_path = os.path.join(script_dir, "original-2.jpeg")
max_side = 1008

# ==============================================

print("加载处理器...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

print("加载模型...")
device = torch.device("cuda:0")
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to(device).eval()

print(f"加载图片: {image_path}")
image = Image.open(image_path).convert("RGB")
orig_w, orig_h = image.size
print(f"原始尺寸: {orig_w}x{orig_h}")

if max(orig_w, orig_h) > max_side:
    scale = max_side / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    scaled_image = image.resize((new_w, new_h), Image.LANCZOS)
    print(f"缩放后尺寸: {new_w}x{new_h}")
else:
    new_w, new_h = orig_w, orig_h
    scaled_image = image

scale_x = orig_w / new_w
scale_y = orig_h / new_h

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
except:
    font = ImageFont.load_default()
    font_large = font

def query_model(question):
    """返回模型的原始文本输出"""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": scaled_image},
            {"type": "text", "text": question}
        ]
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(model.device)
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)

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
    result = outputs if isinstance(outputs, str) else processor.decode(outputs[0], skip_special_tokens=True)
    print("模型输出：", result)
    return result

def detect_objects(question):
    """用模型检测目标并返回去重后的边界框"""
    result = query_model(question)
    boxes = re.findall(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>', result)
    valid = []
    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        if (x1 == 0 and y1 == 0 and x2 >= 990 and y2 >= 990):
            continue
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
        valid.append((x1, y1, x2, y2))

    # 去重 + 过滤小框 + 修复不完整的框
    unique = []
    for box in valid:
        x1, y1, x2, y2 = box
        if y2 - y1 < 50:
            continue
        if not any(abs(x1-ux1)<50 and abs(y1-uy1)<50 and abs(x2-ux2)<50 and abs(y2-uy2)<50 for ux1,uy1,ux2,uy2 in unique):
            unique.append(box)

    # 修复过矮的框：扩展到中位高度
    if len(unique) >= 2:
        heights = [y2 - y1 for x1, y1, x2, y2 in unique]
        median_h = sorted(heights)[len(heights) // 2]
        fixed = []
        for x1, y1, x2, y2 in unique:
            h = y2 - y1
            if h < median_h * 0.6:
                y1 = max(0, y2 - int(median_h))
            fixed.append((x1, y1, x2, y2))
        unique = fixed

    # 头顶扩展：框往上延伸8%以包含完整头部，保底20px留标签空间
    headroom = []
    for x1, y1, x2, y2 in unique:
        h = y2 - y1
        extend = int(h * 0.08)
        y1_new = max(20, y1 - extend)
        headroom.append((x1, y1_new, x2, y2))
    unique = headroom
    return unique

# ========== 任务1：检测球员 ==========
print("\n" + "="*50)
print("任务: 检测球员 (带球衣号和姓名)")
print("="*50)

player_boxes = detect_objects("Find all the players in the image.")
print(f"检测到 {len(player_boxes)} 个球员")

# 按 x1 从左到右排序
player_boxes.sort(key=lambda b: b[0])

# 绘制球员标注
orig_image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(orig_image)

colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

for i, (x1, y1, x2, y2) in enumerate(player_boxes):
    x1_orig = int(x1 * scale_x)
    y1_orig = int(y1 * scale_y)
    x2_orig = int(x2 * scale_x)
    y2_orig = int(y2 * scale_y)

    c = colors[i % len(colors)]
    draw.rectangle([x1_orig, y1_orig, x2_orig, y2_orig], outline=c, width=4)
    draw.text((x1_orig, y1_orig - 30), f"player-{i+1}", fill=c, font=font_large)
    print(f"  player-{i+1}")

output_path = os.path.join(script_dir, "result-2.jpeg")
orig_image.save(output_path, quality=95)
print(f"已保存球员标注图: {output_path}")

print("\n全部完成！")