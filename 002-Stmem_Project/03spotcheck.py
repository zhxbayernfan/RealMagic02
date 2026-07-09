#!/usr/bin/env python3
"""03spotcheck.py — 单独重新描述指定帧
用法: python3 03spotcheck.py 57
      python3 03spotcheck.py 1 2 3
      python3 03spotcheck.py 10-15"""
import sqlite3, os, base64, sys, time, subprocess, tempfile, json, re

API_URL = "http://192.168.0.100:8000/v1/chat/completions"
MODEL = "MiniCPM-V-4.6"
MAX_TOKENS = 640
PROMPT = "第一行用一句温暖具体、有画面感的话描述这张图片（20字以内，包含人物或具体动作，不要以「这是一张」开头），空一行后第二段开始用流畅的中文详细描述（500~700字）：场景、人物、物体、动作。描述中需自然融入氛围形容词如「安静」「专注」「温暖」「宁静」「悠闲」「热闹」等。写完就停，不要重复。"
FRAMES_DIR = os.path.expanduser("~/Desktop/zhx/RealMagic02/002-Stmem_Project/001-Data/frames")
DB_PATH = os.path.expanduser("~/Desktop/zhx/RealMagic02/002-Stmem_Project/001-Data/memory.sqlite")

def call_minicpm(image_b64):
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": PROMPT}
            ]
        }],
        "max_tokens": MAX_TOKENS
    }
    tmp = tempfile.mktemp(suffix='.json')
    with open(tmp, 'w') as f:
        json.dump(payload, f)
    try:
        r = subprocess.run(
            ["curl", "-s", "-X", "POST", API_URL,
             "-H", "Content-Type: application/json",
             "-d", f"@{tmp}"],
            capture_output=True, text=True, timeout=200)
        resp = json.loads(r.stdout)
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"\n  [API错误] {e}")
        return ""
    finally:
        try: os.unlink(tmp)
        except: pass

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

    # 检查 API
    r = subprocess.run(["curl", "-s", "--max-time", "3", "http://192.168.0.100:8000/"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("错误：MiniCPM API 未运行"); sys.exit(1)

    db = sqlite3.connect(DB_PATH)
    ok = 0
    for fid in ids:
        fname = f"frame_{fid:03d}.jpg"
        path = os.path.join(FRAMES_DIR, fname)
        if not os.path.exists(path):
            # 也试试 .png
            fname = f"frame_{fid:03d}.png"
            path = os.path.join(FRAMES_DIR, fname)
        if not os.path.exists(path):
            print(f"❌ frame_{fid:03d} 不存在 (.jpg/.png)")
            continue
        print(f"🔄 {fname}", end=" ", flush=True)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        desc = call_minicpm(b64)
        if not desc: desc = "(生成失败)"
        else: ok += 1
        # 去掉模型输出的占位前缀（"一行""空行""### 描述部分"等）
        # 去掉模型输出的各种占位前后缀（"一行""空行""空一行""### 描述部分""详细描述："等）
        desc = re.sub(r'^(?:[空一]*行\s*\n+|###?\s*描述部分\s*\n+|以下是第一行[^：]*：\s*\n*|·\s*(?:第一行[：:])?\s*|第二段[：:]\s*\n*)', '', desc)
        desc = re.sub(r'\n\s*[空一]*行\s*\n+', '\n', desc)
        desc = re.sub(r'\n\s*第二段[^，]*，?\s*(?:详细)?描述[：:]\s*\n*', '\n', desc)
        desc = desc.strip()
        mtime = os.path.getmtime(path)
        cap_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mtime))
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        db.execute("INSERT OR REPLACE INTO memories (id, filename, description, model, capture_time, file_size) VALUES (?, ?, ?, ?, ?, ?)",
            (int(fid), fname, desc, f"minicpm/{MODEL}", cap_time, os.path.getsize(path)))
        db.commit()
        print(f"✅ ({len(desc)}字)")
    db.close()
    print(f"完成：{ok}/{len(ids)}")

if __name__ == "__main__":
    main()
