#!/usr/bin/env python3
"""pullsqlite.py — 导出 SQLite 记忆描述到指定文件"""
import sqlite3, os, sys, textwrap

OUT = os.path.expanduser("~/Desktop/zhx/RealMagic02/002-Stmem_Project/001-Data/memories_export.txt")
DB = os.path.expanduser("~/Desktop/zhx/RealMagic02/002-Stmem_Project/001-Data/memory.sqlite")

db = sqlite3.connect(DB)
rows = db.execute("SELECT id, filename, capture_time, description FROM memories ORDER BY CAST(id AS INTEGER)").fetchall()
db.close()

with open(OUT, "w", encoding="utf-8") as f:
    for i, (fid, fname, ts, desc) in enumerate(rows, 1):
        t = (ts or "2026-06-30T00:00:00").replace("T", " ")
        d = desc.replace("\n", " ") if desc else "(空)"
        # 每100字换行
        d = "\n".join(textwrap.wrap(d, width=100)) if d != "(空)" else d
        f.write(f" {i:>2}  {fid:>3}  {t[:19]}\n{d}\n\n")

print(f"已导出 {len(rows)} 条 → {OUT}")
