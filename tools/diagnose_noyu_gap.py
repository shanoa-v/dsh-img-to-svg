"""diagnose_noyu_gap.py -- 查清 Noyu 组为何达标率只有 31%

## 起因

v1.1.1 修复后，Noyu 组达标率（MAE<=9.454）只有 31.0%，远低于 pixiv 68.8%。
我一度归因为「平滑阶段抹掉线稿」，但实测推翻：

  600 小图（Noyu 主尺寸）线稿保留率 98.6%，几乎无损
  392 极小图才 67.1%

所以 Noyu 的误差来源不是线稿丢失。必须重新定位。

## 方法

取 Noyu 组最差 3 张与最好 3 张，逐阶段拆解误差：
  原图 → 中值 → 保边混合 → 量化 → 最终渲染

看误差主要在哪一步累积，以及各阶段对「未达标」的贡献。
"""
import os
import sys
import json
import numpy as np
from PIL import Image, ImageFilter
from sklearn.cluster import MiniBatchKMeans
from skimage import color as skcolor

def _find_root():

    """工作区根：优先环境变量，否则取仓库的上一级。



    原实现写死作者机器上的 F:\\SVG drawing。

    """

    env = os.environ.get("IMGSVG_ROOT")

    if env and os.path.isdir(env):

        return env

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))





ROOT = _find_root()
BASE = os.path.join(ROOT, "input", "new_train", "Noyu")
OUT = os.path.join(ROOT, "workspace", "noyu_gap")
os.makedirs(OUT, exist_ok=True)

EDGE_KEEP = 0.16
N_COLORS = 56

# 从 baseline/final06 里挑最差与最好
def load(p):
    d = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            d[r["key"]] = r
        except Exception:
            pass
    return d


B = load(os.path.join(ROOT, "output", "dataset_profile", "baseline.jsonl"))
F = load(os.path.join(ROOT, "output", "dataset_profile", "final06.jsonl"))
noyu = [(k, F[k]["mae"]) for k in F
        if k.startswith("Noyu/") and F[k].get("mae") is not None]
noyu.sort(key=lambda x: -x[1])
WORST = [k.split("/", 1)[1] for k, _ in noyu[:3]]
BEST = [k.split("/", 1)[1] for k, _ in noyu[-3:]]

print("=== Noyu 组最好/最差 ===")
print("  最差 3 张: %s" % [(f[:28], round(m, 2)) for (k, m), f in zip(noyu[:3], WORST)])
print("  最好 3 张: %s" % [(f[:28], round(m, 2)) for (k, m), f in zip(noyu[-3:], BEST)])


def stage_mae(path):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    orig = np.asarray(im).astype(np.float32)
    lum = orig.mean(axis=2)

    gx = np.zeros_like(lum); gy = np.zeros_like(lum)
    gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
    gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
    grad = np.hypot(gx, gy)
    gmax = np.percentile(grad, 99) + 1e-6
    wk = np.clip(np.clip(grad / gmax, 0, 1) * EDGE_KEEP * 3, 0, 1)[..., None]

    m2 = np.asarray(im.filter(ImageFilter.MedianFilter(3))
                    .filter(ImageFilter.MedianFilter(3))).astype(np.float32)
    edge = m2 * (1 - wk) + orig * wk

    # 量化（Oklab，与管线一致）
    flat = edge.reshape(-1, 3)
    lab = skcolor.rgb2lab(flat / 255.0)
    km = MiniBatchKMeans(n_clusters=N_COLORS, random_state=42,
                         batch_size=4096, n_init=3)
    idx = km.fit_predict(lab)
    cen_lab = km.cluster_centers_
    cen_rgb = np.clip(skcolor.lab2rgb(cen_lab.reshape(1, -1, 3))[0], 0, 1) * 255
    rec = cen_rgb[idx].reshape(H, W, 3)

    def mae(a, b):
        return float(np.abs(a - b).mean())

    return {
        "w": W, "h": H, "mp": round(W * H / 1e6, 2),
        "median": round(mae(orig, m2), 3),
        "edge": round(mae(orig, edge), 3),
        "quant": round(mae(orig, rec), 3),
        "edge_over_median": round(mae(orig, edge) / max(mae(orig, m2), 1e-6), 3),
    }


print()
print("=" * 96)
print("逐阶段 MAE（不含最终渲染，用于看误差累积位置）")
print("=" * 96)
print("%-30s %6s %9s %9s %9s %10s"
      % ("文件", "MP", "中值后", "保边后", "量化后", "保边/中值"))

results = {"worst": [], "best": []}
for tag, files in (("worst", WORST), ("best", BEST)):
    print()
    print("--- %s ---" % ("最差 3 张" if tag == "worst" else "最好 3 张"))
    for f in files:
        p = os.path.join(BASE, f)
        if not os.path.exists(p):
            print("  缺失 %s" % f); continue
        r = stage_mae(p)
        r["file"] = f
        results[tag].append(r)
        print("%-30s %6.2f %9.3f %9.3f %9.3f %10.3f"
              % (f[:30], r["mp"], r["median"], r["edge"],
                 r["quant"], r["edge_over_median"]))

print()
print("=" * 96)
print("判读")
print("=" * 96)
for tag in ("worst", "best"):
    rows = results[tag]
    if not rows:
        continue
    md = np.mean([r["median"] for r in rows])
    ed = np.mean([r["edge"] for r in rows])
    qt = np.mean([r["quant"] for r in rows])
    print("  %-10s 平均: 中值后 %.3f -> 保边后 %.3f -> 量化后 %.3f"
          % ("最差组" if tag == "worst" else "最好组", md, ed, qt))
    print("           中值贡献 %.1f%%  量化贡献 %.1f%%"
          % (100 * md / qt, 100 * (qt - ed) / qt))

json.dump(results, open(os.path.join(OUT, "gap.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print()
print("-> %s" % os.path.join(OUT, "gap.json"))
