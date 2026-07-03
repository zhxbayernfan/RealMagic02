#!/usr/bin/env python3
"""pullsqlite.py — 导出 SQLite 记忆描述到指定文件"""
import sqlite3, os, sys

OUT = "/Users/rm010/Desktop/zhx/RealMagic02/002-Stmem_Project/001-Data/memories_export.txt"
DB = "/Users/rm010/Desktop/zhx/RealMagic02/002-Stmem_Project/001-Data/memory.sqlite"

db = sqlite3.connect(DB)
rows = db.execute("SELECT id, capture_time, description FROM memories ORDER BY CAST(id AS INTEGER)").fetchall()
db.close()

def wrap_text(text, width=100):
    """每 width 字换行"""
    lines = []
    for i in range(0, len(text), width):
        lines.append(text[i:i+width])
    return "\n".join(lines)

with open(OUT, "w", encoding="utf-8") as f:
    for i, (fid, ts, desc) in enumerate(rows, 1):
        t = (ts or "2026-06-30T00:00:00").replace("T", " ")
        d = desc.replace("\n", " ") if desc else "(空)"
        f.write(f" {i:>2}  {fid}  {t[:19]}\n")
        f.write(wrap_text(d) + "\n")

print(f"已导出 {len(rows)} 条 → {OUT}")
