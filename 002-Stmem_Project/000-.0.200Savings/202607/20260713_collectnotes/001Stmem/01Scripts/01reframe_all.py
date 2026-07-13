#!/usr/bin/env python3
"""01reframe.py — 批量生成全部帧的描述（全量模式）"""
import sqlite3, os, json, urllib.request, base64, sys, time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:e2b"
PROMPT = "第一行用一句温暖具体、有画面感的话描述这张图片（20字以内，包含人物或具体动作，不要以「这是一张」开头），空一行后第二段开始用流畅的中文详细描述（500字以上）：场景、人物、物体、动作。描述中需自然融入氛围形容词如「安静」「专注」「温暖」「宁静」「悠闲」「热闹」等。"
FRAMES_DIR = os.path.expanduser("~/lingbot-map/lingbot-jszn/data/frames")
DB_PATH = os.path.expanduser("~/lingbot-map/lingbot-jszn/data/memory.sqlite")

def call_ollama(image_b64, retries=2):
    for attempt in range(retries + 1):
        data = json.dumps({
            "model": MODEL, "prompt": PROMPT, "images": [image_b64],
            "stream": False, "options": {"num_predict": 1024, "temperature": 0.7}
        }).encode()
        req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=180)
            desc = json.loads(resp.read()).get("response", "").strip()
            if len(desc) >= 50 or attempt >= retries:
                return desc
            print(f"  [重试{attempt+1}: 仅{len(desc)}字]")
        except Exception as e:
            print(f"\n  [API错误] {e}")
            return ""

def main():
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=3)
    except:
        print("错误：Ollama 未运行"); sys.exit(1)

    frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])
    if not frames:
        print("未找到帧文件"); return

    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("DELETE FROM memories")
    db.commit()
    print("已清空旧记忆，开始生成...")

    total = len(frames)
    ok = 0; t0 = time.time()
    for i, fname in enumerate(frames, 1):
        fid = fname.replace('frame_', '').replace('.jpg', '')
        path = os.path.join(FRAMES_DIR, fname)
        print(f"[{i}/{total}] {fname}", end="", flush=True)
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
        mark = "✅" if len(desc) >= 550 else "❌"
        print(f" ({len(desc)}字){mark}")
    elapsed = time.time() - t0
    db.close()
    print(f"\n完成！{ok}/{total} 帧，耗时 {elapsed:.0f} 秒")

if __name__ == "__main__":
    main()
