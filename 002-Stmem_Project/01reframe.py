#!/usr/bin/env python3
"""01reframe.py — 重新生成全部帧的描述，写入 SQLite（调用 LinuxC MiniCPM-V 4.6）"""
import sqlite3, os, json, subprocess, base64, sys, time

API_URL = "http://192.168.0.100:8000/v1/chat/completions"
MODEL = "MiniCPM-V-4.6"
PROMPT = "用流畅的中文详细描述这张图片，不少于300字：场景、人物、物体、动作、氛围、细节。"
FRAMES_DIR = "/Users/rm010/Desktop/zhx/RealMagic02/002-Stmem_Project/001-Data/frames"
DB_PATH = "/Users/rm010/Desktop/zhx/RealMagic02/002-Stmem_Project/001-Data/memory.sqlite"

def call_minicpm(image_b64, retries=2):
    """用 curl 调用 MiniCPM API（写入临时文件避免参数过长）"""
    import tempfile
    for attempt in range(retries + 1):
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
            desc = body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if len(desc) >= 50 or attempt >= retries:
                return desc
            print(f"  [重试{attempt + 1}: 仅{len(desc)}字]")
        except subprocess.TimeoutExpired:
            print(f"\n  [API错误] 超时")
            return ""
        except Exception as e:
            if attempt < retries:
                print(f"  [重试{attempt + 1}: {e}]")
            else:
                print(f"\n  [API错误] {e}")
                return ""

def main():
    frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not frames:
        print("未找到帧文件"); return

    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("DELETE FROM memories")
    db.commit()
    print("已清空旧记忆，开始生成...")

    total = len(frames)
    ok = 0
    t0 = time.time()
    for i, fname in enumerate(frames, 1):
        fid = fname.replace('frame_', '').rsplit('.', 1)[0]
        path = os.path.join(FRAMES_DIR, fname)
        print(f"[{i}/{total}] {fname}", end="", flush=True)

        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        desc = call_minicpm(b64)
        if not desc:
            desc = "(生成失败)"
        else:
            ok += 1

        mtime = os.path.getmtime(path)
        cap_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mtime))
        now = time.strftime("%Y-%m-%dT%H:%M:%S")

        db.execute(
            "INSERT INTO memories (id, filename, description, model, capture_time, file_size) VALUES (?, ?, ?, ?, ?, ?)",
            (fid, fname, desc, "MiniCPM-V-4.6", cap_time, os.path.getsize(path))
        )
        db.commit()
        print(f" ({len(desc)}字)")

    elapsed = time.time() - t0
    db.close()
    print(f"\n完成！{ok}/{total} 帧成功，耗时 {elapsed:.0f} 秒")
    print(f"失败: {total-ok} 帧")

if __name__ == "__main__":
    main()
