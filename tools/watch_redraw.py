"""watch_redraw.py -- 重绘进度监控与异常检测

## 用途

长夜无人值守时，定期检查：
  1. 进度（已完成 / 总数）
  2. 质量趋势（MAE 是否正常，有无异常值）
  3. 磁盘空间（防止写满）
  4. 进程存活

## 输出一行摘要，便于记录
"""
import os
import json
import subprocess
import sys
import time

OUT = os.path.join(_find_root(), 'output/final_v1.2.0/梅原生')
LOG = os.path.join(_find_root(), 'workspace/redraw_4000.log')
BASE = os.path.join(_find_root(), 'input/new_train/梅原生')
def main():
    total = len([f for f in os.listdir(BASE)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))])

    rows = []
    p = os.path.join(OUT, "_rows.jsonl")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass

    n = len(rows)
    print("进度 %d/%d (%.1f%%)" % (n, total, 100 * n / max(total, 1)))

    if rows:
        import numpy as np
        m = np.array([r["mae"] for r in rows])
        p99 = np.array([r["p99"] for r in rows])
        mb = sum(r["svg_mb"] for r in rows)
        sec = sum(r["sec"] for r in rows)
        print("MAE 中位 %.3f  均值 %.3f  范围 %.2f~%.2f"
              % (np.median(m), m.mean(), m.min(), m.max()))
        print("p99 中位 %.0f   SVG 合计 %.1f GB   总耗时 %.1f h"
              % (np.median(p99), mb / 1024, sec / 3600))
        # 最近 5 条
        print("最近:")
        for r in rows[-5:]:
            print("  %-28s MAE %6.3f p99 %5.0f %6.1fMB %5.0fs"
                  % (r["file"][:28], r["mae"], r["p99"], r["svg_mb"], r["sec"]))
        # 异常检测
        med = np.median(m)
        bad = [r for r in rows if r["mae"] > med * 1.8]
        if bad:
            print("异常偏高 %d 张:" % len(bad))
            for r in bad[:5]:
                print("  %-28s MAE %.3f" % (r["file"][:28], r["mae"]))
    else:
        print("尚无完成记录")

    # 磁盘
    try:
        out = subprocess.run(["df", "-h", "/f"], capture_output=True,
                             text=True).stdout
        line = out.strip().splitlines()[-1]
        print("磁盘: %s" % " ".join(line.split()[1:]))
    except Exception:
        pass

    # 进程
    try:
        nm = subprocess.run(["tasklist"], capture_output=True,
                            text=True).stdout.lower().count("python.exe")
        print("python 进程: %d" % nm)
    except Exception:
        pass

    # 日志尾部
    if os.path.exists(LOG):
        try:
            with open(LOG, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            tail = [l.rstrip() for l in lines[-3:]]
            print("日志尾:")
            for l in tail:
                print("  %s" % l[-150:])
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
