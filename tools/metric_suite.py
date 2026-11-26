"""metric_suite.py -- 补上 MAE 掩盖的局部灾难指标

## 为什么需要

MAE 是全图平均，会稀释局部灾难。实测：392×392 白底图的暗部（10.5% 面积）
R 通道偏差 +109.94，而全图 MAE 只有 24.8 —— 黑色线稿被抹掉这件事，
在 MAE 里被 89% 的白色背景稀释掉了。

类似地，早先测过「>100 误差像素 +17.8% 但 MAE 只降 1.2%」的情况。

## 本工具给出 6 个指标

  1. MAE            全图平均（传统，会被稀释）
  2. p95/p99 误差   长尾（局部灾难在这里现形）
  3. darkBias       暗部（源 L<80）的 RGB 最大通道偏差
  4. colorsKeep     色阶保留率（8级量化唯一色 / 源）
  5. worstBlockMAE  最差 3x3 网格块的平均误差（局部灾难定位）
  6. detailLoss     细节区（源 grad>=15）的误差，独立于平坦区

## 用法

  python metric_suite.py <源图> <渲染图> [--json out.json]
"""
import os
import sys
import json
import argparse
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
def metrics(ref, ren):
    W, H = ref.shape[1], ref.shape[0]
    err = np.abs(ref - ren).max(axis=2)
    serr = ren - ref
    L = ref.mean(axis=2)
    mx = ref.max(axis=2); mn = ref.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    g = L
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    grad = np.hypot(gx, gy)

    m = {}
    m["mae"] = round(float(err.mean()), 3)
    m["p90"] = round(float(np.percentile(err, 90)), 2)
    m["p95"] = round(float(np.percentile(err, 95)), 2)
    m["p99"] = round(float(np.percentile(err, 99)), 2)
    m["max"] = round(float(err.max()), 1)
    m["gt30_pct"] = round(100 * float((err > 30).mean()), 2)
    m["gt60_pct"] = round(100 * float((err > 60).mean()), 2)
    m["gt100_pct"] = round(100 * float((err > 100).mean()), 2)

    # 暗部偏差
    dark = L < 80
    if dark.sum() >= 50:
        m["dark_pct"] = round(100 * float(dark.mean()), 2)
        m["dark_bias"] = round(float(np.abs(serr[dark]).max(axis=1).mean()), 2)
        m["dark_err"] = round(float(err[dark].mean()), 2)
    else:
        m["dark_pct"] = round(100 * float(dark.mean()), 2)
        m["dark_bias"] = None
        m["dark_err"] = None

    # 细节区
    det = grad >= 15
    if det.sum() >= 100:
        m["detail_pct"] = round(100 * float(det.mean()), 2)
        m["detail_err"] = round(float(err[det].mean()), 2)
        m["detail_bias"] = round(float(np.abs(serr[det]).max(axis=1).mean()), 2)
    else:
        m["detail_err"] = None

    # 平坦区
    flat = grad < 3
    if flat.sum() >= 100:
        m["flat_pct"] = round(100 * float(flat.mean()), 2)
        m["flat_err"] = round(float(err[flat].mean()), 2)

    # 色阶
    q1 = (ref // 8).astype(np.int64)
    q2 = (ren // 8).astype(np.int64)
    n1 = len(np.unique(q1[..., 0] * 10000 + q1[..., 1] * 100 + q1[..., 2]))
    n2 = len(np.unique(q2[..., 0] * 10000 + q2[..., 1] * 100 + q2[..., 2]))
    m["colors_src"] = n1
    m["colors_ren"] = n2
    m["colors_keep"] = round(100.0 * n2 / max(n1, 1), 1)

    # 最差 3x3 网格块
    best = 0.0
    bn = max(1, min(H, W) // 3)
    for by in range(0, max(1, H - bn), bn):
        for bx in range(0, max(1, W - bn), bn):
            v = float(err[by:by + bn, bx:bx + bn].mean())
            if v > best:
                best = v
    m["worst_block"] = round(best, 2)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("ren")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    ref_im = Image.open(args.src).convert("RGB")
    W, H = ref_im.size
    ref = np.asarray(ref_im).astype(np.float32)
    v = Image.open(args.ren).convert("RGB")
    if v.size != (W, H):
        v = v.resize((W, H), Image.LANCZOS)
    ren = np.asarray(v).astype(np.float32)

    m = metrics(ref, ren)
    m["src"] = os.path.basename(args.src)
    m["size"] = [W, H]

    print("%-16s %s" % ("源图", m["src"]))
    print("%-16s %dx%d" % ("尺寸", W, H))
    print()
    print("%-18s %10s" % ("MAE", m["mae"]))
    print("%-18s %10s" % ("p90 / p95 / p99",
                          "%.1f / %.1f / %.1f" % (m["p90"], m["p95"], m["p99"])))
    print("%-18s %10s" % (">30 / >60 / >100 %",
                          "%.1f / %.1f / %.1f" % (m["gt30_pct"], m["gt60_pct"],
                                                  m["gt100_pct"])))
    print("%-18s %10s" % ("最差3x3块MAE", m["worst_block"]))
    print()
    print("%-18s %10s" % ("暗部占比%", m["dark_pct"]))
    print("%-18s %10s" % ("暗部误差/偏差",
                          "%.1f / %s" % (m["dark_err"], m["dark_bias"])
                          if m["dark_err"] is not None else "-"))
    print("%-18s %10s" % ("细节区占比%", m.get("detail_pct", "-")))
    print("%-18s %10s" % ("细节区误差",
                          m["detail_err"] if m["detail_err"] is not None else "-"))
    print("%-18s %10s" % ("平坦区误差", m.get("flat_err", "-")))
    print()
    print("%-18s %10s" % ("色阶 保留%",
                          "%d -> %d (%.1f%%)"
                          % (m["colors_src"], m["colors_ren"], m["colors_keep"])))

    if args.json:
        json.dump(m, open(args.json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("\n-> %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
