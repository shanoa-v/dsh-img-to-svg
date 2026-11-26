"""dataset_profile.py -- 建立 new_train 三组数据集的完整画像

## 为什么先做这个

不了解数据就做决定是盲目的。本项目历史教训：单张图的结论换图型后完全不成立。
必须先知道：
  1. 三组各自的图像特征（尺寸/亮度/饱和度/复杂度）
  2. 与已训练过的 batch1-3 有何差异
  3. 哪些是"困难图"（高复杂/低饱和/大尺寸）
  4. 抽样策略：每组抽多少张才能代表全体

## 输出

  output/dataset_profile/profile.json   完整画像
  output/dataset_profile/summary.md     人读摘要
"""
import os
import json
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, median_filter

def _find_root():

    """工作区根：优先环境变量，否则取仓库的上一级。



    原实现写死作者机器上的 F:\\SVG drawing。

    """

    env = os.environ.get("IMGSVG_ROOT")

    if env and os.path.isdir(env):

        return env

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))





ROOT = _find_root()
BASE = os.path.join(ROOT, "input", "new_train")
OUT = os.path.join(ROOT, "output", "dataset_profile")
os.makedirs(OUT, exist_ok=True)

GROUPS = ["Noyu", "pixiv", "梅原生"]


def profile_image(path, max_side=800):
    im = Image.open(path)
    W, H = im.size
    fmt = (im.format or "?").lower()
    im = im.convert("RGB")
    if max(W, H) > max_side:
        sc = max_side / max(W, H)
        im2 = im.resize((int(W * sc), int(H * sc)), Image.LANCZOS)
    else:
        im2 = im
    a = np.asarray(im2).astype(np.float32)
    lum = a.mean(axis=2)
    mx = a.max(axis=2); mn = a.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    g = lum
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    grad = np.hypot(gx, gy)
    # 8 级量化唯一色数（复杂度代理）
    q = (a // 32).astype(int)
    ids = q[..., 0] * 10000 + q[..., 1] * 100 + q[..., 2]
    return {
        "w": W, "h": H, "mp": round(W * H / 1e6, 2), "fmt": fmt,
        "lum": round(float(lum.mean()), 1),
        "sat_p90": round(float(np.percentile(sat, 90)), 3),
        "sat_mean": round(float(sat.mean()), 3),
        "grad_mean": round(float(grad.mean()), 2),
        "flat_pct": round(100 * float((grad < 3).mean()), 1),
        "detail_pct": round(100 * float((grad >= 15).mean()), 1),
        "colors8": int(len(np.unique(ids))),
        "bytes": os.path.getsize(path),
    }


print("=== 数据集画像 ===")
rows = {}
for gname in GROUPS:
    d = os.path.join(BASE, gname)
    if not os.path.isdir(d):
        print("  缺少组:", gname); continue
    files = sorted(f for f in os.listdir(d)
                   if f.lower().endswith((".webp", ".png", ".jpg", ".jpeg")))
    print("\n--- %s  %d 张 ---" % (gname, len(files)))
    recs = []
    for i, f in enumerate(files):
        p = os.path.join(d, f)
        try:
            r = profile_image(p)
            r["file"] = f
            recs.append(r)
        except Exception as e:
            print("  跳过 %s (%s)" % (f, e))
        if (i + 1) % 100 == 0:
            print("    已处理 %d/%d" % (i + 1, len(files)))
    if not recs:
        continue
    rows[gname] = recs
    # 组统计
    mp = np.array([r["mp"] for r in recs])
    lum = np.array([r["lum"] for r in recs])
    col = np.array([r["colors8"] for r in recs])
    flat = np.array([r["flat_pct"] for r in recs])
    det = np.array([r["detail_pct"] for r in recs])
    print("  尺寸 中位 %.2f MP  范围 %.2f~%.2f" % (np.median(mp), mp.min(), mp.max()))
    print("  亮度 中位 %.0f" % np.median(lum))
    print("  8级色数 中位 %.0f  范围 %d~%d" % (np.median(col), col.min(), col.max()))
    print("  平坦区 中位 %.1f%%   细节区 中位 %.1f%%" % (np.median(flat), np.median(det)))
    print("  格式分布: %s" % {f: sum(1 for r in recs if r["fmt"] == f)
                            for f in set(r["fmt"] for r in recs)})

# 保存
with open(os.path.join(OUT, "profile.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)

# 人读摘要 + 困难图清单
lines = ["# new_train 数据集画像", ""]
for gname, recs in rows.items():
    mp = np.array([r["mp"] for r in recs])
    col = np.array([r["colors8"] for r in recs])
    det = np.array([r["detail_pct"] for r in recs])
    flat = np.array([r["flat_pct"] for r in recs])
    lum = np.array([r["lum"] for r in recs])
    lines.append("## %s（%d 张）" % (gname, len(recs)))
    lines.append("")
    lines.append("| 指标 | 中位 | 范围 |")
    lines.append("|---|---|---|")
    lines.append("| 像素 | %.2f MP | %.2f~%.2f |" % (np.median(mp), mp.min(), mp.max()))
    lines.append("| 亮度 | %.0f | %.0f~%.0f |" % (np.median(lum), lum.min(), lum.max()))
    lines.append("| 8级色数 | %.0f | %d~%d |" % (np.median(col), col.min(), col.max()))
    lines.append("| 平坦区%% | %.1f | %.1f~%.1f |" % (np.median(flat), flat.min(), flat.max()))
    lines.append("| 细节区%% | %.1f | %.1f~%.1f |" % (np.median(det), det.min(), det.max()))
    # 最难的 5 张
    worst = sorted(recs, key=lambda r: -r["colors8"])[:5]
    lines.append("")
    lines.append("色数最高（最复杂）5 张：")
    for r in worst:
        lines.append("  %s  %dx%d %.1fMP  色数%d  细节%.1f%%"
                     % (r["file"][:40], r["w"], r["h"], r["mp"],
                        r["colors8"], r["detail_pct"]))
    lines.append("")
open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(lines))

total = sum(len(v) for v in rows.values())
print("\n总计 %d 张" % total)
print("画像 -> %s" % os.path.join(OUT, "profile.json"))
print("摘要 -> %s" % os.path.join(OUT, "summary.md"))
