#!/usr/bin/python3
import os, re, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoProcessor

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = "/home/zhanghexiang/LocateAnything-3B"
image_path = os.path.join(script_dir, "original-1.jpeg")
question = "Find all the cats in the image."
label = "cat"
max_side = 1500

print("加载模型...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
device = torch.device("cuda:0")
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
torch.cuda.reset_peak_memory_stats()

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

scale_x = orig_w / new_w
scale_y = orig_h / new_h

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
except:
    font = ImageFont.load_default()

def query_model(question):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, videos=vids, return_tensors="pt").to(model.device)
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(pixel_values=pixel_values, input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=processor, max_new_tokens=128, generation_mode="hybrid", do_sample=False, use_cache=True)
    result = outputs if isinstance(outputs, str) else processor.decode(outputs[0], skip_special_tokens=True)
    print("模型输出：", result)
    return result

result = query_model(question)
boxes = re.findall(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>', result)

if boxes:
    unique_boxes = []
    tol = 50
    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        if (x1 == 0 and y1 == 0 and x2 >= 990 and y2 >= 990):
            continue
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
        is_dup = False
        for ux1, uy1, ux2, uy2 in unique_boxes:
            if abs(x1 - ux1) < tol and abs(y1 - uy1) < tol and abs(x2 - ux2) < tol and abs(y2 - uy2) < tol:
                is_dup = True
                break
        if not is_dup:
            unique_boxes.append((x1, y1, x2, y2))
    print(f"原始 {len(boxes)} 个框，去重后 {len(unique_boxes)} 个")

    image_area = new_w * new_h
    filtered_boxes = []
    for (x1, y1, x2, y2) in unique_boxes:
        box_area = (x2 - x1) * (y2 - y1)
        if box_area < 0.9 * image_area:
            filtered_boxes.append((x1, y1, x2, y2))
    unique_boxes = filtered_boxes

    orig_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(orig_image)
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]

    for i, (x1, y1, x2, y2) in enumerate(unique_boxes):
        x1_orig = int(float(x1) / 1000 * new_w * scale_x)
        y1_orig = int(float(y1) / 1000 * new_h * scale_y)
        x2_orig = int(float(x2) / 1000 * new_w * scale_x)
        y2_orig = int(float(y2) / 1000 * new_h * scale_y)
        color = colors[i % len(colors)]
        draw.rectangle([x1_orig, y1_orig, x2_orig, y2_orig], outline=color, width=3)
        draw.text((x1_orig, y1_orig - 25), label, fill=color, font=font)

    output_path = os.path.join(script_dir, "result-1.jpeg")
    orig_image.save(output_path, quality=95)
    print(f"已保存: {output_path}")
    print(f"共检测到 {len(boxes)} 个框")
else:
    print("未检测到任何框")

gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
print(f"GPU_MEM: {gpu_mem:.0f}")
