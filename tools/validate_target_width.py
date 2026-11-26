"""validate_target_width.py -- 小样本验证"目标宽度统一"策略

## 策略

对每张图：k = ceil(4000 / 长边)，上限 8；长边 >= 4000 则 k=1。
把工作尺寸统一到约 4000~5000px 长边。

## 为什么用 ceil 而非 round

round 会让 2667px 的图（ideal k=1.50）取 k=1，实际长边仍是 2667，
达不到 4000 目标；而同组其他图被放大到 4000+，造成质量不一致。
ceil 保证所有图的长边都 >= 4000（除本身就超过的）。

## 本脚本验证

取 6 张覆盖不同原始尺寸（含 1x 与 3x 边界），
对比「round 计划」与「ceil 计划」的实际 MAE，确认 ceil 更一致。
"""
import os
import json
import math
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
OUT = os.path.join(ROOT, "workspace", "tw_validate")
os.makedirs(OUT, exist_ok=True)
PY = sys.executable
SK = os.path.join(os.environ["DSH_HOME"], "skills", "dsh-img-to-svg", "scripts")
TARGET = 4000

# 覆盖不同原始长边：852 / 1300 / 2134 / 2489 / 2666 / 3162
PICKS = [852, 1300, 2134, 2489, 2666, 3162]


def find_by_longside(ls):
    best = None
    for f in os.listdir(BASE):
        if not f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        p = os.path.join(BASE, f)
        try:
            im = Image.open(p)
        except Exception:
            continue
        if max(im.width, im.height) == ls:
            best = (f, im.width, im.height)
            break
    return best


def run(src, k, tag):
    im = Image.open(src).convert("RGB")
    W0, H0 = im.size
    if k > 1:
        up = os.path.join(OUT, "%s_up%d.png" % (tag, k))
        if not os.path.exists(up):
            im.resize((W0 * k, H0 * k), Image.LANCZOS).save(up)
        work = up
    else:
        work = src
    svg = os.path.join(OUT, "%s_k%d.svg" % (tag, k))
    png = svg.replace(".svg", ".png")
    env = {a: b for a, b in os.environ.items() if not a.startswith("VEC_")}
    env["VEC_TITLE"] = tag
    env["VEC_SEED"] = "42"
    r = subprocess.run([PY, os.path.join(SK, "vec.py"), work, svg],
                       capture_output=True, text=True, env=env, cwd=SK)
    if r.returncode != 0:
        return None
    subprocess.run([PY, os.path.join(SK, "render_check.py"), svg, png],
                   capture_output=True, cwd=SK)
    ref = np.asarray(im).astype(np.float32)
    v = Image.open(png).convert("RGB")
    if v.size != (W0, H0):
        v = v.resize((W0, H0), Image.LANCZOS)
    ren = np.asarray(v).astype(np.float32)
    e = np.abs(ref - ren)
    em = e.max(axis=2)
    mb = os.path.getsize(svg) / 1048576
    if k > 1:
        try:
            os.remove(work)
        except Exception:
            pass
    return {
        "k": k, "work_w": W0 * k, "work_h": H0 * k,
        "mae": round(float(e.mean()), 3),
        "p99": round(float(np.percentile(em, 99)), 1),
        "mb": round(mb, 2),
    }


print("=== 目标宽度 %d 验证（round vs ceil）===" % TARGET)
print("%-34s %6s %6s %10s %8s %8s %8s"
      % ("文件", "长边", "k_round", "k_ceil", "MAE(round)", "MAE(ceil)", "差距"))

results = []
for ls in PICKS:
    found = find_by_longside(ls)
    if not found:
        print("  未找到长边 %d 的图" % ls); continue
    f, W0, H0 = found
    tag = os.path.splitext(f)[0][:14]
    kr = 1 if ls >= TARGET else max(1, min(8, int(round(TARGET / ls))))
    kc = 1 if ls >= TARGET else max(1, min(8, math.ceil(TARGET / ls)))
    a = run(os.path.join(BASE, f), kr, tag)
    b = run(os.path.join(BASE, f), kc, tag) if kc != kr else a
    if a is None or b is None:
        print("  %-34s 失败" % f[:34]); continue
    gap = (b["mae"] - a["mae"]) / a["mae"] * 100
    print("%-34s %6d %6d %10d %8.3f %8.3f %+7.2f%%"
          % (f[:34], ls, kr, kc, a["mae"], b["mae"], gap))
    results.append({"file": f, "ls": ls, "kr": kr, "kc": kc,
                    "round": a, "ceil": b})

json.dump(results, open(os.path.join(OUT, "validate.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)

if results:
    print()
    # 一致性：两者工作宽度差多少
    print("=== 工作宽度一致性 ===")
    print("%-34s %10s %10s %10s" % ("文件", "round宽", "ceil宽", "差"))
    for r in results:
        w1 = r["round"]["work_w"]
        w2 = r["ceil"]["work_w"]
        print("%-34s %10d %10d %+10d"
              % (r["file"][:34], w1, w2, w2 - w1))

    maes_r = [r["round"]["mae"] for r in results]
    maes_c = [r["ceil"]["mae"] for r in results]
    print()
    print("  round 方案 MAE: 中位 %.3f  范围 %.3f~%.3f  极差 %.3f"
          % (np.median(maes_r), min(maes_r), max(maes_r),
             max(maes_r) - min(maes_r)))
    print("  ceil  方案 MAE: 中位 %.3f  范围 %.3f~%.3f  极差 %.3f"
          % (np.median(maes_c), min(maes_c), max(maes_c),
             max(maes_c) - min(maes_c)))
    print()
    print("  判据：极差更小 = 质量更一致")
