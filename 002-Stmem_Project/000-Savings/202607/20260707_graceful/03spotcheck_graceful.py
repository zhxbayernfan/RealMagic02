#!/usr/bin/env python3
"""03spotcheck.py — 单独重新描述指定帧
用法: python3 03spotcheck.py 57
      python3 03spotcheck.py 1 2 3
      python3 03spotcheck.py 10-15"""
import sqlite3, os, json, urllib.request, base64, sys, time, re

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
        print(f"  [API错误] {e}")
        return ""

def parse_args():
    ids = []
    for arg in sys.argv[1:]:
        m = re.match(r'^(\d+)-(\d+)$', arg)
        if m:
            ids.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ids.append(int(arg))
    return ids

def main():
    ids = parse_args()
    if not ids:
        print("用法: python3 03spotcheck.py 57")
        print("      python3 03spotcheck.py 1 2 3")
        print("      python3 03spotcheck.py 10-15")
        sys.exit(1)

    try:
        urllib.request.urlopen("http://localhost:11434", timeout=3)
    except:
        print("错误：Ollama 未运行"); sys.exit(1)

    db = sqlite3.connect(DB_PATH)
    ok = 0
    for fid in ids:
        fname = f"frame_{fid:03d}.jpg"
        path = os.path.join(FRAMES_DIR, fname)
        if not os.path.exists(path):
            print(f"❌ {fname} 不存在")
            continue
        print(f"🔄 {fname}", end=" ", flush=True)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        desc = call_ollama(b64)
        if not desc:
            desc = "(生成失败)"
        else:
            ok += 1
        mtime = os.path.getmtime(path)
        cap_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mtime))
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        db.execute("INSERT OR REPLACE INTO memories (id, frame_path, description, timestamp, capture_time, model) VALUES (?, ?, ?, ?, ?, ?)",
            (str(fid), f"data/frames/{fname}", desc, now, cap_time, f"ollama/{MODEL}"))
        db.commit()
        print(f"✅ ({len(desc)}字)")
    db.close()
    print(f"完成：{ok}/{len(ids)}")

if __name__ == "__main__":
    main()
