#!/usr/bin/env python3
"""补全缺失帧的描述（不删已有数据）"""
import sqlite3, os, json, urllib.request, base64, time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:e2b"
PROMPT = "第一行用一句有温度、有画面感的话描述这张图片（20字以内，不要以「这是一张」开头），空一行后第二段开始用流畅的中文详细描述：场景、人物、物体、动作。"
FRAMES_DIR = os.path.expanduser("~/lingbot-map/lingbot-jszn/data/frames")
DB_PATH = os.path.expanduser("~/lingbot-map/lingbot-jszn/data/memory.sqlite")

def call_ollama(image_b64):
    data = json.dumps({
        "model": MODEL, "prompt": PROMPT, "images": [image_b64],
        "stream": False, "options": {"num_predict": 1024, "temperature": 0.7}
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=180)
        return json.loads(resp.read()).get("response", "").strip()
    except Exception as e:
        print(f"\n  [API错误] {e}")
        return ""

def main():
    # 找缺失的帧
    frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])
    db = sqlite3.connect(DB_PATH)
    existing = set(str(r[0]) for r in db.execute("SELECT id FROM memories").fetchall())
    todo = []
    for fname in frames:
        fid = fname.replace('frame_', '').replace('.jpg', '')
        if fid not in existing:
            todo.append((fid, fname))
    db.close()

    if not todo:
        print("全部帧已有描述，无需处理")
        return

    print(f"共 {len(frames)} 帧，{len(existing)} 帧已有描述，需要补 {len(todo)} 帧")

    db = sqlite3.connect(DB_PATH)
    ok = 0; t0 = time.time()
    for i, (fid, fname) in enumerate(todo, 1):
        path = os.path.join(FRAMES_DIR, fname)
        print(f"[{i}/{len(todo)}] {fname}", end="", flush=True)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        desc = call_ollama(b64)
        if not desc: desc = "(生成失败)"
        else: ok += 1
        mtime = os.path.getmtime(path)
        cap_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mtime))
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        db.execute("INSERT INTO memories (id, frame_path, description, timestamp, capture_time, model) VALUES (?, ?, ?, ?, ?, ?)",
            (fid, f"data/frames/{fname}", desc, now, cap_time, f"ollama/{MODEL}"))
        db.commit()
        print(f" ({len(desc)}字)")
    elapsed = time.time() - t0
    db.close()
    print(f"\n完成！{ok}/{len(todo)} 帧，耗时 {elapsed:.0f} 秒")

if __name__ == "__main__":
    main()
