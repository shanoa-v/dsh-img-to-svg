"""produce_final.py -- 产出最终成果：用最优配置批量矢量化

## 用途

在验证完成后，用最优配置（VEC_BASECOL=1 VEC_SEAL=1.0）处理指定图片集，
输出 SVG + 渲染 PNG + 三联对比图。

## 产出结构

  <outdir>/
    svg/<name>.svg           矢量结果
    png/<name>_render.png    渲染图
    cmp/<name>_compare.png   三联对比（原图 / 渲染 / 差异热图）
    _summary.json            全部指标
    _report.md               人读报告

## 用法

  python produce_final.py --src-dir <输入目录> --outdir <输出目录> [--limit N]
"""
import os
import io
import re
import json
import time
import argparse
import subprocess
import numpy as np
from PIL import Image, ImageDraw
import sys

def _find_python():

    """定位可用解释器：优先环境变量，否则用当前解释器。



    原实现写死作者机器上的 D:\\anaconda3\\python.exe，

    换到别的机器必然找不到。

    """

    env = os.environ.get("IMGSVG_PYTHON")

    if env and os.path.isfile(env):

        return env

    return sys.executable





PY = _find_python()
SK = os.path.join(os.environ["DSH_HOME"], "skills", "dsh-img-to-svg", "scripts")

# 不硬编码任何 VEC_* —— 统一依赖 vec3.py 的默认值，避免两处漂移。
# v1.1.1 起默认即 BASECOL=1 + SEAL=0.6（169 张全量验证：零退化）。
# 若将来要覆盖，请用环境变量显式传入，不要改这里。
OPTIMAL = {}


def metrics(ref, ren):
    err = np.abs(ref - ren)
    em = err.max(axis=2)
    L = ref.mean(axis=2)
    dark = L < 80
    serr = ren - ref
    q1 = (ref // 8).astype(np.int64)
    q2 = (ren // 8).astype(np.int64)
    n1 = len(np.unique(q1[..., 0] * 10000 + q1[..., 1] * 100 + q1[..., 2]))
    n2 = len(np.unique(q2[..., 0] * 10000 + q2[..., 1] * 100 + q2[..., 2]))
    return {
        "mae": round(float(err.mean()), 3),
        "p95": round(float(np.percentile(em, 95)), 1),
        "p99": round(float(np.percentile(em, 99)), 1),
        "gt60_pct": round(100 * float((em > 60).mean()), 2),
        "dark_bias": (round(float(serr[..., 0][dark].mean()), 2)
                      if dark.sum() > 50 else None),
        "colors_keep": round(100.0 * n2 / max(n1, 1), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--match", default="")
    args = ap.parse_args()

    src_dir = os.path.abspath(args.src_dir)
    outdir = os.path.abspath(args.outdir)
    for sub in ("svg", "png", "cmp"):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)

    files = sorted(f for f in os.listdir(src_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    if args.match:
        files = [f for f in files if args.match in f]
    if args.limit:
        files = files[:args.limit]

    print("=== 产出最终成果 ===")
    print("  输入 %s（%d 张）" % (src_dir, len(files)))
    print("  输出 %s" % outdir)
    print("  配置 %s" % OPTIMAL)

    rows = []
    t_start = time.time()
    for i, f in enumerate(files):
        src = os.path.join(src_dir, f)
        stem = os.path.splitext(f)[0]
        svg = os.path.join(outdir, "svg", stem + ".svg")
        png = os.path.join(outdir, "png", stem + "_render.png")

        env = {k: v for k, v in os.environ.items() if not k.startswith("VEC_")}
        env["VEC_TITLE"] = stem
        env.update(OPTIMAL)

        t0 = time.time()
        r = subprocess.run([PY, os.path.join(SK, "vec.py"), src, svg],
                           capture_output=True, text=True, env=env, cwd=SK)
        dt = time.time() - t0
        if r.returncode != 0:
            print("  [%3d/%d] %-34s 失败 %s"
                  % (i + 1, len(files), f[:34],
                     (r.stderr.strip().splitlines() or ['?'])[-1][:60]))
            continue

        rr = subprocess.run([PY, os.path.join(SK, "render_check.py"), svg, png],
                            capture_output=True, text=True, cwd=SK)
        if rr.returncode != 0 or not os.path.exists(png):
            print("  [%3d/%d] %-34s 渲染失败" % (i + 1, len(files), f[:34]))
            continue

        ref_im = Image.open(src).convert("RGB")
        W, H = ref_im.size
        ref = np.asarray(ref_im).astype(np.float32)
        v = Image.open(png).convert("RGB")
        if v.size != (W, H):
            v = v.resize((W, H), Image.LANCZOS)
        ren = np.asarray(v).astype(np.float32)
        m = metrics(ref, ren)

        # 三联对比
        sc = min(1.0, 700.0 / W)
        tw, th = max(1, int(W * sc)), max(1, int(H * sc))
        a = ref_im.resize((tw, th), Image.LANCZOS)
        b = v.resize((tw, th), Image.LANCZOS)
        d = np.abs(ref - ren).max(axis=2)
        heat = Image.fromarray(np.clip(d * 4, 0, 255).astype(np.uint8)).convert("RGB")
        heat = heat.resize((tw, th), Image.LANCZOS)
        LAB = 22
        cv = Image.new("RGB", (tw * 3 + 16, LAB + th), (248, 248, 248))
        dr = ImageDraw.Draw(cv)
        for j, (lb, imx) in enumerate([
                ("ORIGINAL", a),
                ("SVG  MAE %.2f" % m["mae"], b),
                ("DIFF x4  p99 %.0f" % m["p99"], heat)]):
            x = j * (tw + 8)
            dr.text((x + 4, 4), lb, fill=(0, 0, 0))
            cv.paste(imx, (x, LAB))
        cv.save(os.path.join(outdir, "cmp", stem + "_compare.png"))

        m.update({"file": f, "size": [W, H], "mp": round(W * H / 1e6, 2),
                  "sec": round(dt, 1), "svg_mb": round(os.path.getsize(svg) / 1048576, 2)})
        rows.append(m)
        el = time.time() - t_start
        eta = (el / (i + 1)) * (len(files) - i - 1) / 60
        print("  [%3d/%d] %-34s %5.2fMP %6.1fs MAE %6.3f  p99 %5.1f  暗部%+7.2f  ETA %.0fmin"
              % (i + 1, len(files), f[:34], m["mp"], dt, m["mae"], m["p99"],
                 m["dark_bias"] if m["dark_bias"] is not None else 0, eta))

    if not rows:
        print("无成功产出"); return 1

    maes = np.array([r["mae"] for r in rows])
    summary = {
        "count": len(rows),
        "config": OPTIMAL,
        "mae_median": round(float(np.median(maes)), 3),
        "mae_mean": round(float(maes.mean()), 3),
        "mae_min": round(float(maes.min()), 3),
        "mae_max": round(float(maes.max()), 3),
        "p99_median": round(float(np.median([r["p99"] for r in rows])), 1),
        "total_sec": round(time.time() - t_start, 1),
        "total_mb": round(sum(r["svg_mb"] for r in rows), 2),
        "rows": rows,
    }
    json.dump(summary, open(os.path.join(outdir, "_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("\n=== 汇总 ===")
    print("  成功 %d 张" % len(rows))
    print("  MAE 中位 %.3f  均值 %.3f  范围 %.3f~%.3f"
          % (summary["mae_median"], summary["mae_mean"],
             summary["mae_min"], summary["mae_max"]))
    print("  总耗时 %.0f min   总体积 %.1f MB"
          % (summary["total_sec"] / 60, summary["total_mb"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
