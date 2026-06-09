#!/usr/bin/python3
"""Test MiniCPM-V 4.6 on original-008."""
import os, torch, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

model_path = "/home/zhanghexiang/MiniCPM-4_6"
work_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(work_dir, "original-008.")

print("加载模型...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_path, trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="cuda:0"
).eval()
print(f"模型就绪，加载耗时 {time.time()-t0:.1f}s")

image = Image.open(image_path).convert("RGB")
print(f"图片尺寸: {image.size}")

question = "用中文描述这张图片的内容。"
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question}
    ]}
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device, dtype=model.dtype)

print("推理中...")
t1 = time.time()
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(f"推理耗时 {time.time()-t1:.1f}s")

result = processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(f"\n问题: {question}")
print(f"回答: {result}")

output_file = os.path.join(work_dir, "original-008.")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Image: original-008.jpg\nQuestion: {question}\nAnswer: {result}\n")
print(f"\n结果已保存到 original-008.")
