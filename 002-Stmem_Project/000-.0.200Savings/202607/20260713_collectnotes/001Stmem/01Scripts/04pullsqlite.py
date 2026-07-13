#!/usr/bin/env python3
"""pullsqlite.py — 导出 SQLite 记忆描述到指定文件"""
import sqlite3, os, sys, time

datestr = time.strftime("%Y%m%d")
OUT = os.path.expanduser(f"~/lingbot-map/000Notes/001/02Produces/02Memories_{datestr}.txt")
DB = os.path.expanduser("~/lingbot-map/lingbot-jszn/data/memory.sqlite")

db = sqlite3.connect(DB)
rows = db.execute("SELECT id, capture_time, description FROM memories ORDER BY CAST(id AS INTEGER)").fetchall()
db.close()

with open(OUT, "w", encoding="utf-8") as f:
    for i, (fid, ts, desc) in enumerate(rows, 1):
        t = (ts or "2026-06-30T00:00:00").replace("T", " ")
        d = desc.replace("\n", " ") if desc else "(空)"
        f.write(f" {i:>2}  {fid}  {t[:19]}  {d}\n")

print(f"已导出 {len(rows)} 条 → {OUT}")
