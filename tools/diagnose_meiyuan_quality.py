"""diagnose_meiyuan_quality.py -- 梅原生组质量诊断与"高分辨率假设"验证

## 受访参考物

用户给的 1706919294780.svg（58.82 MB）不是手绘精品，而是本管线对
5200x11312（58.82 MP）原图的输出：
  path 168216 个、56 色、stroke-width=0.6（v1.1.1 补缝）、
  底色 #6B3F53（BASECOL 边缘中位色）

它"看起来质量高"的真实原因有两个候选：
  A. 原图分辨率极高（58.82 MP），像素信息本身丰富
  B. 处理时没有降采样（WORK_W = W0 = 5200）

其他梅原生的图多为 1~3 MP，若把工作宽度放大（VEC_W 上采样），
是否能让它们接近那种观感？这必须实测，不能假设。

## 本脚本做两件事

1. 列出梅原生 95 张的尺寸分布与当前 MAE
2. 对 3 张代表图（小/中/大），扫 VEC_W 倍数，看 MAE 与长尾怎么变
"""
import os
import json
import subprocess
import sys
import numpy as np
from PIL import Image

def _find_root():

    """工作区根：优先环境变量，否则取仓库的上一级。



    原实现写死作者机器上的 F:\\SVG drawing。

    """

    env = os.environ.get("IMGSVG_ROOT")

    if env and os.path.isdir(env):

        return env

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))





ROOT = _find_root()
BASE = os.path.join(ROOT, "input", "new_train", "梅原生")
OUT = os.path.join(ROOT, "workspace", "meiyuan_diag")
os.makedirs(OUT, exist_ok=True)
PY = sys.executable
SK = os.path.join(os.environ["DSH_HOME"], "skills", "dsh-img-to-svg", "scripts")

prof = json.load(open(os.path.join(ROOT, "output", "dataset_profile",
                                   "profile.json"), encoding="utf-8"))
recs = prof.get("梅原生", [])

print("=== 梅原生 95 张尺寸分布 ===")
mp = np.array([r["mp"] for r in recs])
print("  MP: 中位 %.2f  最小 %.2f  最大 %.2f" % (np.median(mp), mp.min(), mp.max()))
bins = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 20), (20, 999)]
print("  %-12s %6s" % ("像素区间", "数量"))
for lo, hi in bins:
    n = int(((mp >= lo) & (mp < hi)).sum())
    if n:
        print("  [%2d,%3d) MP   %4d" % (lo, hi, n))

# 当前产出质量
summ = os.path.join(ROOT, "output", "final_v1.1.1", "梅原生", "_summary.json")
cur = {}
if os.path.exists(summ):
    d = json.load(open(summ, encoding="utf-8"))
    for r in d["rows"]:
        cur[r["file"]] = r
print()
print("=== 当前产出（前 40 张）质量 ===")
if cur:
    ms = [v["mae"] for v in cur.values() if isinstance(v.get("mae"), (int, float))]
    mps = [v["mp"] for v in cur.values() if isinstance(v.get("mp"), (int, float))]
    print("  已产出 %d 张，MAE 中位 %.3f，MP 中位 %.2f"
          % (len(ms), np.median(ms), np.median(mps)))
    # 看 MAE 与 MP 的关系
    pairs = [(v["mp"], v["mae"]) for v in cur.values()
             if isinstance(v.get("mp"), (int, float))
             and isinstance(v.get("mae"), (int, float))]
    if len(pairs) > 3:
        a = np.array(pairs)
        print("  MP 与 MAE 相关系数: %+.3f" % np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        a = a[a[:, 0].argsort()]
        print("  按 MP 排序（%d 个点，取 6 个分位）:" % len(a))
        for i in np.linspace(0, len(a) - 1, 6).astype(int):
            print("    %.2f MP  MAE %.3f" % (a[i, 0], a[i, 1]))

# 选代表图做 VEC_W 扫描
cands = sorted(recs, key=lambda r: r["mp"])
picks = []
for target in (1.0, 2.5, 6.0):
    best = min(cands, key=lambda r: abs(r["mp"] - target))
    if all(best["file"] != p["file"] for p in picks):
        picks.append(best)

print()
print("=== 代表图 ===")
for p in picks:
    print("  %-40s %.2f MP  %dx%d" % (p["file"][:40], p["mp"], p["w"], p["h"]))


def run(src, tag, vec_w):
    svg = os.path.join(OUT, "%s_w%d.svg" % (tag, vec_w))
    png = svg.replace(".svg", ".png")
    env = {k: v for k, v in os.environ.items() if not k.startswith("VEC_")}
    env["VEC_TITLE"] = "d"
    env["VEC_SEED"] = "42"
    env["VEC_W"] = str(vec_w)
    r = subprocess.run([PY, os.path.join(SK, "vec.py"), src, svg],
                       capture_output=True, text=True, env=env, cwd=SK)
    if r.returncode != 0:
        return None
    subprocess.run([PY, os.path.join(SK, "render_check.py"), svg, png],
                   capture_output=True, cwd=SK)
    ref_im = Image.open(src).convert("RGB")
    W, H = ref_im.size
    ref = np.asarray(ref_im).astype(np.float32)
    v = Image.open(png).convert("RGB")
    if v.size != (W, H):
        v = v.resize((W, H), Image.LANCZOS)
    ren = np.asarray(v).astype(np.float32)
    e = np.abs(ref - ren)
    em = e.max(axis=2)
    return {
        "mae": round(float(e.mean()), 3),
        "p99": round(float(np.percentile(em, 99)), 1),
        "gt60": round(100 * float((em > 60).mean()), 2),
        "mb": round(os.path.getsize(svg) / 1048576, 2),
    }


print()
print("=== VEC_W 扫描（每张图按其原宽 W0 取倍数）===")
results = {}
for p in picks:
    src = os.path.join(BASE, p["file"])
    if not os.path.exists(src):
        print("  缺失 %s" % p["file"][:30]); continue
    W0 = p["w"]
    tag = os.path.splitext(p["file"])[0][:20]
    print()
    print("  %s（%.2f MP，W0=%d）" % (p["file"][:34], p["mp"], W0))
    print("  %-12s %8s %8s %8s %9s" % ("VEC_W", "MAE", "p99", "gt60%", "体积MB"))
    row = {}
    for mult in (1.0, 1.5, 2.0):
        w = int(W0 * mult)
        res = run(src, tag, w)
        if res is None:
            print("  %-12s 失败" % ("%dx" % mult)); continue
        row[mult] = res
        print("  %-12s %8.3f %8.1f %7.2f%% %9.2f"
              % ("%dx (%d)" % (mult, w), res["mae"], res["p99"],
                 res["gt60"], res["mb"]))
    results[p["file"]] = row

json.dump(results, open(os.path.join(OUT, "vecw_sweep.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
print()
print("注：VEC_W 有 min(W, W0*2) 硬上限，故 2.0x 是可达的最大倍数。")
print("-> %s" % os.path.join(OUT, "vecw_sweep.json"))
