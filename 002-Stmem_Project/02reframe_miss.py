#!/usr/bin/env python3
"""补全缺失帧的描述（不删已有数据）"""
import sqlite3, os, base64, time, subprocess, tempfile, json, re

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

def main():
    # 找缺失的帧
    frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    db = sqlite3.connect(DB_PATH)
    existing = set(str(int(r[0])) for r in db.execute("SELECT id FROM memories").fetchall())
    under_len = set(str(int(r[0])) for r in db.execute("SELECT id FROM memories WHERE LENGTH(description) < 500").fetchall())
    todo = []
    for fname in frames:
        fid = str(int(fname.replace('frame_', '').rsplit('.', 1)[0]))
        if fid not in existing or fid in under_len:
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
        print(f" ({len(desc)}字)")
    elapsed = time.time() - t0
    db.close()
    print(f"\n完成！{ok}/{len(todo)} 帧，耗时 {elapsed:.0f} 秒")

if __name__ == "__main__":
    main()
