#!/usr/bin/env python3
"""03spotcheck.py — 单独重新描述指定帧
用法: python3 03spotcheck.py 57
      python3 03spotcheck.py 1 2 3
      python3 03spotcheck.py 10-15"""
import sqlite3, os, json, subprocess, base64, sys, time, re, tempfile

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
        print("用法: python3 03spotcheck.py 57"); print("      python3 03spotcheck.py 1 2 3"); print("      python3 03spotcheck.py 10-15")
        sys.exit(1)
    db = sqlite3.connect(DB_PATH)
    ok = 0
    for fid in ids:
        fname = None
        for ext in ['jpg', 'jpeg', 'png']:
            p = os.path.join(FRAMES_DIR, f"frame_{fid:03d}.{ext}")
            if os.path.exists(p):
                fname = f"frame_{fid:03d}.{ext}"; path = p; break
        if not fname:
            print(f"❌ frame_{fid:03d} 不存在"); continue
        print(f"🔄 {fname}", end=" ", flush=True)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        desc = call_minicpm(b64)
        if not desc: desc = "(生成失败)"
        else: ok += 1
        mtime = os.path.getmtime(path)
        cap_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mtime))
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        db.execute("INSERT OR REPLACE INTO memories (id, frame_path, description, timestamp, capture_time, model) VALUES (?, ?, ?, ?, ?, ?)",
            (str(fid), fname, desc, now, cap_time, MODEL))
        db.commit()
        print(f"✅ ({len(desc)}字)")
    db.close()
    print(f"完成：{ok}/{len(ids)}")

if __name__ == "__main__":
    main()
