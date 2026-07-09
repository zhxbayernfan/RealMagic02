#!/usr/bin/env python3
"""01reframe.py — 批量生成全部帧的描述（全量模式）"""
import sqlite3, os, base64, sys, time, subprocess, tempfile, json, re

API_URL = "http://192.168.0.100:8000/v1/chat/completions"
MODEL = "MiniCPM-V-4.6"
MAX_TOKENS = 640
PROMPT = "第一行用一句温暖具体、有画面感的话描述这张图片（20字以内，包含人物或具体动作，不要以「这是一张」开头），空一行后第二段开始用流畅的中文详细描述（500~700字）：场景、人物、物体、动作。描述中需自然融入氛围形容词如「安静」「专注」「温暖」「宁静」「悠闲」「热闹」等。写完就停，不要重复。"
FRAMES_DIR = os.path.expanduser("~/Desktop/zhx/RealMagic02/002-Stmem_Project/001-Data/frames")
DB_PATH = os.path.expanduser("~/Desktop/zhx/RealMagic02/002-Stmem_Project/001-Data/memory.sqlite")

def call_minicpm(image_b64, retries=2):
    for attempt in range(retries + 1):
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
            desc = resp["choices"][0]["message"]["content"].strip()
            if len(desc) >= 50 or attempt >= retries:
                return desc
            print(f"  [重试{attempt+1}: 仅{len(desc)}字]")
        except Exception as e:
            print(f"\n  [API错误] {e}")
            return ""
        finally:
            try: os.unlink(tmp)
            except: pass

def main():
    # 检查 API 是否运行
    r = subprocess.run(["curl", "-s", "--max-time", "3", "http://192.168.0.100:8000/"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("错误：MiniCPM API 未运行"); sys.exit(1)

    frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))])
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
        fid = fname.replace('frame_', '').rsplit('.', 1)[0]
        path = os.path.join(FRAMES_DIR, fname)
        print(f"[{i}/{total}] {fname}", end="", flush=True)
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
        db.execute("INSERT INTO memories (id, filename, description, model, capture_time, file_size) VALUES (?, ?, ?, ?, ?, ?)",
            (int(fid), fname, desc, f"minicpm/{MODEL}", cap_time, os.path.getsize(path)))
        db.commit()
        print(f" ({len(desc)}字)")
    elapsed = time.time() - t0
    db.close()
    print(f"\n完成！{ok}/{total} 帧，耗时 {elapsed:.0f} 秒")

if __name__ == "__main__":
    main()
