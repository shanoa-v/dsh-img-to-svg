"""smooth_damage_profile.py -- 量化平滑阶段对各尺寸图的损伤

## 目的

为「平滑阶段改造」提供方案设计依据。要回答：

  1. 两遍 3x3 中值到底抹掉了多少细节，随尺寸如何变化？
  2. 保边混合 `wk = clip(grad/gmax * EDGE_KEEP * 3, 0, 1)`
     在 EDGE_KEEP=0.16 时，实际最大 wk 是多少？（是否等于 0.48）
  3. 保边混合救回了多少？（对比 sm vs edge_sm）
  4. 损伤的量级是否真的足以解释 Noyu 组 31% 达标率的差距？

## 指标

  stroke_keep  线稿保留率（源 L<100 且 grad>20 的像素，处理后仍 <140）
  detail_err   细节区（源 grad>=15）的 MAE
  flat_err     平坦区（源 grad<3）的 MAE
  wk_max       保边混合的实际最大权重（验证 EDGE_KEEP 是否被 clip 饱和）
  wk_mean      实际平均权重
"""
import os
import sys
import json
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import median_filter

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
OUT = os.path.join(ROOT, "workspace", "smooth_diag")
os.makedirs(OUT, exist_ok=True)

# 跨尺寸、跨组抽样
CASES = [
    ("392极小", "pixiv", "illust_133681772_20261025_214817.png"),
    ("600小", "Noyu", "105819262_p0_master1200.webp"),
    ("600小2", "Noyu", "106001113_p0_master1200.webp"),
    ("1000中", "pixiv", "illust_112635661_20260808_134220.png"),
    ("1402中", "Noyu", "125091690_p0_master1200.webp"),
    ("1920中", "梅原生", "1735710131383.jpg"),
    ("2894大", "pixiv", "illust_148080192_20260911_225837.png"),
]

EDGE_KEEP = 0.16


def analyze(path):
    im = Image.open(path).convert("RGB")
    W0, H0 = im.size
    orig = np.asarray(im).astype(np.float32)
    lum = orig.mean(axis=2)

    gx = np.zeros_like(lum); gy = np.zeros_like(lum)
    gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
    gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
    grad = np.hypot(gx, gy)

    # 线稿像素定义（与 trace_lineart 一致）
    line = (lum < 100) & (grad > 20)
    n_line = int(line.sum())

    # 三种滤波
    m1 = np.asarray(im.filter(ImageFilter.MedianFilter(3))).astype(np.float32)
    m2 = np.asarray(im
                    .filter(ImageFilter.MedianFilter(3))
                    .filter(ImageFilter.MedianFilter(3))).astype(np.float32)

    # 保边混合（复刻 vec3.py 逻辑）
    gmax = np.percentile(grad, 99) + 1e-6
    wk = np.clip(np.clip(grad / gmax, 0, 1) * EDGE_KEEP * 3, 0, 1)
    edge_sm = m2 * (1 - wk[..., None]) + orig * wk[..., None]

    def stroke_keep(arr):
        if n_line < 100:
            return None
        L = arr.mean(axis=2)
        return 100.0 * float((L[line] < 140).mean())

    def zone_err(arr):
        e = np.abs(orig - arr).mean(axis=2)
        det = grad >= 15
        flat = grad < 3
        return (float(e[det].mean()) if det.sum() > 100 else None,
                float(e[flat].mean()) if flat.sum() > 100 else None)

    det_e2, flat_e2 = zone_err(m2)
    det_ee, flat_ee = zone_err(edge_sm)

    return {
        "w": W0, "h": H0, "mp": round(W0 * H0 / 1e6, 2),
        "n_line": n_line, "line_pct": round(100.0 * n_line / (W0 * H0), 2),
        "src_line_lum": round(float(lum[line].mean()), 1) if n_line > 100 else None,
        "keep_1pass": round(stroke_keep(m1), 1) if stroke_keep(m1) is not None else None,
        "keep_2pass": round(stroke_keep(m2), 1) if stroke_keep(m2) is not None else None,
        "keep_edge": round(stroke_keep(edge_sm), 1) if stroke_keep(edge_sm) is not None else None,
        "lum_2pass": round(float(m2.mean(axis=2)[line].mean()), 1) if n_line > 100 else None,
        "lum_edge": round(float(edge_sm.mean(axis=2)[line].mean()), 1) if n_line > 100 else None,
        "wk_max": round(float(wk.max()), 4),
        "wk_mean": round(float(wk.mean()), 4),
        "wk_p99": round(float(np.percentile(wk, 99)), 4),
        "flat_frac": round(100.0 * float((wk < 0.02).mean()), 1),
        "det_err_2pass": round(det_e2, 2) if det_e2 else None,
        "det_err_edge": round(det_ee, 2) if det_ee else None,
        "flat_err_2pass": round(flat_e2, 3) if flat_e2 else None,
        "flat_err_edge": round(flat_ee, 3) if flat_ee else None,
    }


print("=" * 100)
print("平滑阶段损伤画像（EDGE_KEEP=%.2f，保边混合上限=%.2f）"
      % (EDGE_KEEP, min(EDGE_KEEP * 3, 1.0)))
print("=" * 100)

rows = []
for tag, g, f in CASES:
    p = os.path.join(BASE, g, f)
    if not os.path.exists(p):
        print("  缺失 %s" % tag); continue
    r = analyze(p)
    r["tag"] = tag
    r["group"] = g
    rows.append(r)

print()
print("%-9s %5s %9s %8s %8s %8s %8s %8s"
      % ("图", "宽px", "线稿px", "1遍保留", "2遍保留", "保边后", "2遍亮度", "保边亮度"))
for r in rows:
    print("%-9s %5d %9d %7s%% %7s%% %7s%% %8s %8s"
          % (r["tag"], r["w"], r["n_line"],
             r["keep_1pass"], r["keep_2pass"], r["keep_edge"],
             r["lum_2pass"], r["lum_edge"]))

print()
print("保边混合权重（验证 EDGE_KEEP 是否被 clip 饱和）：")
print("%-9s %8s %8s %8s %10s" % ("图", "wk_max", "wk_p99", "wk_mean", "wk<0.02占比"))
for r in rows:
    print("%-9s %8.4f %8.4f %8.4f %9.1f%%"
          % (r["tag"], r["wk_max"], r["wk_p99"], r["wk_mean"], r["flat_frac"]))

print()
print("分区域误差（2遍中值 vs 保边混合后）：")
print("%-9s %10s %10s %10s %10s"
      % ("图", "细节2遍", "细节保边", "平坦2遍", "平坦保边"))
for r in rows:
    print("%-9s %10s %10s %10s %10s"
          % (r["tag"], r["det_err_2pass"], r["det_err_edge"],
             r["flat_err_2pass"], r["flat_err_edge"]))

json.dump(rows, open(os.path.join(OUT, "damage.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print()
print("-> %s" % os.path.join(OUT, "damage.json"))
