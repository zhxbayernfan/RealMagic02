#!/usr/bin/python3
import os, re, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoProcessor
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = "/home/zhanghexiang/LocateAnything-3B"
image_path = os.path.join(script_dir, "original-011.png")
question = "Find all people in the image."
label = "person"
max_side = 1500
print("加载模型...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
device = torch.device("cuda:0")
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
torch.cuda.reset_peak_memory_stats()
image = Image.open(image_path).convert("RGB")
orig_w, orig_h = image.size
print(f"原始尺寸: {orig_w}x{orig_h}")
if max(orig_w, orig_h) > max_side:
    scale = max_side / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
else:
    new_w, new_h = orig_w, orig_h
scale_x = orig_w / new_w; scale_y = orig_h / new_h
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
except:
    font = ImageFont.load_default()
def query_model(question):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, videos=vids, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(pixel_values=pixel_values, input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=processor, max_new_tokens=128, generation_mode="hybrid", do_sample=False, use_cache=True)
    result = outputs if isinstance(outputs, str) else processor.decode(outputs[0], skip_special_tokens=True)
    return result
result = query_model(question)
boxes = re.findall(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>', result)
if boxes:
    unique_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        if (x1 == 0 and y1 == 0 and x2 >= 990): continue
        if x2-x1<10 or y2-y1<10: continue
        dup = False
        for u in unique_boxes:
            if abs(x1-u[0])<50 and abs(y1-u[1])<50 and abs(x2-u[2])<50 and abs(y2-u[3])<50: dup=True; break
        if not dup: unique_boxes.append((x1,y1,x2,y2))
    orig_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(orig_image)
    colors = ["red","blue","green","orange","purple","cyan","magenta","yellow"]
    for i,(x1,y1,x2,y2) in enumerate(unique_boxes):
        ox1=int(x1/1000*new_w*scale_x); oy1=int(y1/1000*new_h*scale_y)
        ox2=int(x2/1000*new_w*scale_x); oy2=int(y2/1000*new_h*scale_y)
        draw.rectangle([ox1,oy1,ox2,oy2], outline=colors[i%8], width=3)
        draw.text((ox1,oy1-25), label, fill=colors[i%8], font=font)
    output_path = os.path.join(script_dir, "result-011.png")
    orig_image.save(output_path, quality=95)
    print(f"已保存: {output_path}")
    print(f"共检测到 {len(boxes)} 个框, 去重后 {len(unique_boxes)} 个")
else:
    print("未检测到任何框")
gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
print(f"GPU_MEM: {gpu_mem:.0f}")
