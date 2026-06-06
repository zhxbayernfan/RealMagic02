#!/usr/bin/python3
"""
Trophies script: Q2结构化查询获取四个奖杯原始框，加标签输出。
"""
import os, re, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
script_dir = os.path.dirname(os.path.abspath(__file__))

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoProcessor

model_path = "/home/zhanghexiang/LocateAnything-3B"
image_path = os.path.join(script_dir, "original-3.jpeg")
max_side = 1008

print("加载模型...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
device = torch.device("cuda:0")
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

image = Image.open(image_path).convert("RGB")
orig_w, orig_h = image.size
scale = max_side / max(orig_w, orig_h)
new_w, new_h = int(orig_w * scale), int(orig_h * scale)
scaled_image = image.resize((new_w, new_h), Image.LANCZOS)
scale_x, scale_y = orig_w / new_w, orig_h / new_h

def query_model(question):
    messages = [{"role": "user", "content": [{"type": "image", "image": scaled_image}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, videos=vids, return_tensors="pt").to(model.device)
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(pixel_values=pixel_values, input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=processor, max_new_tokens=128, generation_mode="hybrid", do_sample=False, use_cache=True)
    return outputs if isinstance(outputs, str) else processor.decode(outputs[0], skip_special_tokens=True)

# 结构化查询：一次获取四个奖杯
trophy_names = ["trophy-1","trophy-2","trophy-3","trophy-4","trophy-5","trophy-6","trophy-7","trophy-8"]
trophy_colors = ["gold", "orange", "cyan", "yellow"]

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
except:
    font = ImageFont.load_default()

result = query_model(
    "List the four trophies from left to right: 1.silver cup 2.trophy with big ears 3.round shield 4.gold cup. Output a bounding box for each."
)
boxes = re.findall(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>', result)
print(f"检测到 {len(boxes)} 个奖杯")

draw = ImageDraw.Draw(image)

for i, (x1, y1, x2, y2) in enumerate(boxes):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    ox1 = int(x1 / 1000 * new_w * scale_x)
    oy1 = int(y1 / 1000 * new_h * scale_y)
    ox2 = int(x2 / 1000 * new_w * scale_x)
    oy2 = int(y2 / 1000 * new_h * scale_y)

    name = trophy_names[i] if i < len(trophy_names) else f"trophy-{i+1}"
    c = trophy_colors[i % len(trophy_colors)]
    draw.rectangle([ox1, oy1, ox2, oy2], outline=c, width=4)
    draw.text((ox1, oy1 - 32), name, fill=c, font=font)
    print(f"  {name}: 模型框({x1},{y1},{x2},{y2}) -> 原图({ox1},{oy1},{ox2},{oy2})")

output_path = os.path.join(script_dir, "result-3.jpeg")
image.save(output_path, quality=95)
print(f"\n奖杯标注图已保存: {output_path}")
