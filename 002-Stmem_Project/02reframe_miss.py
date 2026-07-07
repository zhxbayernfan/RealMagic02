#!/usr/bin/env python3
"""补全缺失帧的描述（不删已有数据）"""
import sqlite3, os, json, subprocess, base64, time, tempfile

API_URL = "http://192.168.0.100:8000/v1/chat/completions"
MODEL = "MiniCPM-V-4.6"
PROMPT = "用流畅的中文描述这张图片：场景、人物、物体、动作。"
FRAMES_DIR = "/Users/rm010/Desktop/zhx/RealMagic02/002-Stmem_Project/001-Data/frames"
DB_PATH = "/Users/rm010/Desktop/zhx/RealMagic02/002-Stmem_Project/001-Data/memory.sqlite"

def call_minicpm(image_b64):
    try:
        payload = json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
                {"type": "text", "text": PROMPT}
            ]}],
            "max_tokens": 2048
        })
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json.loads(payload), f)
            tmpfile = f.name
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", API_URL,
             "-H", "Content-Type: application/json",
             "-d", f"@{tmpfile}"],
            capture_output=True, text=True, timeout=300
        )
        os.unlink(tmpfile)
        body = json.loads(result.stdout)
        return body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except:
        return ""

def main():
    frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    db = sqlite3.connect(DB_PATH)
    existing = set(str(r[0]) for r in db.execute("SELECT id FROM memories").fetchall())
    todo = []
    for fname in frames:
        fid = fname.replace('frame_', '').rsplit('.', 1)[0]
        if fid not in existing:
            todo.append((fid, fname))
    db.close()
    if not todo:
        print("全部帧已有描述，无需处理"); return
    print(f"共 {len(frames)} 帧，{len(existing)} 帧已有描述，需要补 {len(todo)} 帧")
    db = sqlite3.connect(DB_PATH)
    ok = 0; t0 = time.time()
    for i, (fid, fname) in enumerate(todo, 1):
        path = os.path.join(FRAMES_DIR, fname)
        print(f"[{i}/{len(todo)}] {fname}", end="", flush=True)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        desc = call_minicpm(b64)
        if not desc: desc = "(生成失败)"
        else: ok += 1
        mtime = os.path.getmtime(path)
        cap_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mtime))
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        db.execute("INSERT INTO memories (id, frame_path, description, timestamp, capture_time, model) VALUES (?, ?, ?, ?, ?, ?)",
            (fid, fname, desc, now, cap_time, MODEL))
        db.commit()
        print(f" ({len(desc)}字)")
    elapsed = time.time() - t0
    db.close()
    print(f"\n完成！{ok}/{len(todo)} 帧，耗时 {elapsed:.0f} 秒")

if __name__ == "__main__":
    main()
