"""test_physical_upscale.py -- 测试物理放大源图后处理的效果

## 动机

VEC_W 有硬上限 `WORK_W = min(WORK_W, W0 * 2)`，所以 2x 是管线内可达的最大倍数。

但参考物（58.82 MP 的 5200x11312 图）之所以质量好，是因为它的**原始像素就多**。
小图若想接近，只能凭空插值放大——这会引入假细节，但对平滑插画可能仍有效
（信号层被上采样，量化时的相对损失变小）。

## 方法

用 LANCZOS 把源图物理放大 k 倍（k=2,3,4），存临时 PNG，
再跑默认管线（此时 W0 已是放大后尺寸，VEC_W 上限随之提高）。

输出 SVG 的 viewBox 会是放大后尺寸，需在测量时缩回原尺寸比较。

## 判据

MAE / p99 / gt60 / 体积。若 4x 仍持续改善，说明插值放大的收益还没到顶。
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
OUT = os.path.join(ROOT, "workspace", "upscale_test")
os.makedirs(OUT, exist_ok=True)
PY = sys.executable
SK = os.path.join(os.environ["DSH_HOME"], "skills", "dsh-img-to-svg", "scripts")

PICKS = [
    "113117d74e5d9ab526286764290617fd2024069799_raw.jpg",   # 1.02 MP
    "1735710131383.jpg",                                      # 2.07 MP
]


def run_one(src, k):
    im = Image.open(src).convert("RGB")
    W0, H0 = im.size
    tag = os.path.splitext(os.path.basename(src))[0][:16]
    up_w, up_h = int(W0 * k), int(H0 * k)

    # 物理放大
    if k == 1:
        work_src = src
    else:
        work_src = os.path.join(OUT, "%s_up%d.png" % (tag, k))
        if not os.path.exists(work_src):
            im.resize((up_w, up_h), Image.LANCZOS).save(work_src)

    svg = os.path.join(OUT, "%s_k%d.svg" % (tag, k))
    png = svg.replace(".svg", ".png")
    env = {v: w for v, w in os.environ.items() if not v.startswith("VEC_")}
    env["VEC_TITLE"] = "k%d" % k
    env["VEC_SEED"] = "42"
    r = subprocess.run([PY, os.path.join(SK, "vec.py"), work_src, svg],
                       capture_output=True, text=True, env=env, cwd=SK)
    if r.returncode != 0:
        return None, (r.stderr.strip().splitlines() or ["?"])[-1][:80]

    subprocess.run([PY, os.path.join(SK, "render_check.py"), svg, png],
                   capture_output=True, cwd=SK)

    ref = np.asarray(im).astype(np.float32)          # 原始尺寸作为基准
    v = Image.open(png).convert("RGB")
    if v.size != (W0, H0):
        v = v.resize((W0, H0), Image.LANCZOS)        # 缩回原尺寸比较
    ren = np.asarray(v).astype(np.float32)
    e = np.abs(ref - ren)
    em = e.max(axis=2)
    mb = os.path.getsize(svg) / 1048576
    return {
        "k": k, "up": [up_w, up_h],
        "mae": round(float(e.mean()), 3),
        "p99": round(float(np.percentile(em, 99)), 1),
        "gt60": round(100 * float((em > 60).mean()), 2),
        "mb": round(mb, 2),
    }, None


print("=== 物理放大测试 ===")
allres = {}
for f in PICKS:
    src = os.path.join(BASE, f)
    if not os.path.exists(src):
        print("  缺失 %s" % f); continue
    im = Image.open(src)
    print()
    print("  %s（%dx%d = %.2f MP）"
          % (f[:38], im.width, im.height, im.width * im.height / 1e6))
    print("  %-12s %10s %8s %8s %9s %8s"
          % ("倍数", "工作尺寸", "MAE", "p99", "gt60%", "体积MB"))
    rows = []
    for k in (1, 2, 3, 4):
        res, err = run_one(src, k)
        if res is None:
            print("  %-12s 失败: %s" % ("%dx" % k, err)); continue
        rows.append(res)
        print("  %-12s %10s %8.3f %8.1f %7.2f%% %8.2f"
              % ("%dx" % k, "%dx%d" % (res["up"][0], res["up"][1]),
                 res["mae"], res["p99"], res["gt60"], res["mb"]))
    allres[f] = rows

json.dump(allres, open(os.path.join(OUT, "upscale.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print()
print("=== 判读 ===")
for f, rows in allres.items():
    if len(rows) < 2:
        continue
    m = [r["mae"] for r in rows]
    p = [r["p99"] for r in rows]
    g = [r["gt60"] for r in rows]
    print("  %s" % f[:40])
    print("    MAE: %s  最优 %dx"
          % (" -> ".join("%.2f" % x for x in m), rows[int(np.argmin(m))]["k"]))
    print("    p99: %s  最优 %dx"
          % (" -> ".join("%.0f" % x for x in p), rows[int(np.argmin(p))]["k"]))
    print("    gt60: %s  最优 %dx"
          % (" -> ".join("%.2f" % x for x in g), rows[int(np.argmin(g))]["k"]))
    print("    体积: %s"
          % " -> ".join("%.1f" % r["mb"] for r in rows))
