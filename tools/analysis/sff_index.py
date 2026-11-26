"""small_feature_index.py -- 批量衡量「小特征保真度」，替代单图死磕

问题背景：单张图的嘴追了很多轮都没救回来，且已证实在 56 色下面积份额仅
0.0017%，代价过高。改为衡量一个可批量、可比较的指标。

指标定义（Small Feature Fidelity，SFF）：
  小特征 = 源图中「面积 4~400 px 且与周围色差 >= 40」的连通块
  它们包含：唇线、睫毛、眉、瞳孔高光、发丝分叉、纽扣、纹样细节
  SFF = 渲染图中仍然存在（色差仍 >= 25）的小特征个数 / 源图小特征个数

与已有的 mouth_census 的区别：后者用 redness 阈值，抓的是腮红/唇等红色
特征，且渲染图会产生新的红色碎块导致比值 >100%。SFF 用「与周围色差」，
且按源图特征位置逐一核对，避免统计虚高。

用法：
  python small_feature_index.py [tag] [VEC_X=Y ...]
"""
import os
import re
import sys
import json
import subprocess
import numpy as np
from PIL import Image
from scipy.ndimage import label, uniform_filter1d

def _find_root():

    """工作区根：优先环境变量，否则取仓库的上一级。



    原实现写死作者机器上的 F:\\SVG drawing。

    """

    env = os.environ.get("IMGSVG_ROOT")

    if env and os.path.isdir(env):

        return env

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))





ROOT = _find_root()
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
SK = os.path.join(os.environ.get("DSH_HOME", ""), "skills", "dsh-img-to-svg")
OUT = os.path.join(ROOT, "workspace", "sff")
os.makedirs(OUT, exist_ok=True)

# 抽样：不同图型各取几张，代表性强且耗时可接受
IMGS = [
    ("mouth", os.path.join(ROOT, "input", "collections", "batch3",
     "illust_100531443_20260808_134010.png")),
    ("portrait", os.path.join(ROOT, "input", "collections", "batch3",
     "illust_88927530_20260808_134042.png")),
    ("dark", os.path.join(ROOT, "input", "collections", "batch2",
     "93131545_p0_square1200.webp")),
    ("wuther", os.path.join(ROOT, "input", "collections", "batch1", "c_wuthering.png")),
    ("user", os.path.join(ROOT, "input", "user", "user_char.png")),
]

MIN_PX, MAX_PX = 4, 400
SRC_CONTRAST = 40.0
KEEP_CONTRAST = 25.0


def local_contrast(img):
    """每个像素与周围 15x15 均值亮度之差（绝对值）"""
    lum = img.mean(axis=2)
    mean = uniform_filter1d(uniform_filter1d(lum, 15, axis=0), 15, axis=1)
    return np.abs(lum - mean)


def find_small_features(ref):
    """源图里的小特征：(mask, 块列表[每块的像素坐标])"""
    lum = ref.mean(axis=2)
    c = local_contrast(ref)
    cand = c > (SRC_CONTRAST / 4.0)          # 先粗筛
    lb, n = label(cand)
    if n == 0:
        return []
    feats = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lb == k)
        npx = ys.size
        if npx < MIN_PX or npx > MAX_PX:
            continue
        # 与该块边界外的平均色差
        m = (lb == k)
        ring = np.zeros_like(m)
        ring[1:, :] |= m[:-1, :]; ring[:-1, :] |= m[1:, :]
        ring[:, 1:] |= m[:, :-1]; ring[:, :-1] |= m[:, 1:]
        ring &= ~m
        if not ring.any():
            continue
        inner = ref[m].mean(axis=0)
        outer = ref[ring].mean(axis=0)
        if np.abs(inner - outer).max() < SRC_CONTRAST:
            continue
        feats.append((ys, xs, inner, outer))
    return feats


def survived(feat, ren):
    """渲染图里同一位置是否还保留了这个特征（色差 >= KEEP_CONTRAST）"""
    ys, xs, inner, outer = feat
    m_in = ren[ys, xs].mean(axis=0)
    # 用原特征块的包围盒外扩 3px 计算周围色
    y0, y1 = max(0, ys.min() - 3), min(ren.shape[0], ys.max() + 4)
    x0, x1 = max(0, xs.min() - 3), min(ren.shape[1], xs.max() + 4)
    patch = ren[y0:y1, x0:x1]
    mm = np.zeros(patch.shape[:2], dtype=bool)
    mm[ys.min() - y0:ys.max() - y0 + 1, xs.min() - x0:xs.max() - x0 + 1] = True
    if not (~mm).any():
        return False
    m_out = patch[~mm].mean(axis=0)
    return float(np.abs(m_in - m_out).max()) >= KEEP_CONTRAST


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    extra = {}
    for a in sys.argv[2:]:
        if "=" in a:
            k, v = a.split("=", 1)
            extra[k] = v
    print("=== SFF 批量小特征保真度  tag=%s  参数=%s ===" % (tag, extra or "默认"))
    print("%-10s %8s %8s %8s %7s %6s"
          % ("图", "src小特征", "保留", "SFF%", "MAE", "MB"))
    print("-" * 56)
    rows = {}
    for name, src in IMGS:
        if not os.path.exists(src):
            print("%-10s 缺失" % name); continue
        ref_im = Image.open(src).convert("RGB")
        ref = np.asarray(ref_im).astype(np.float32)
        env = {k: v for k, v in os.environ.items() if not k.startswith("VEC_")}
        env.update(extra)
        svg = os.path.join(OUT, "%s_%s.svg" % (tag, name))
        png = svg.replace(".svg", ".png")
        r = subprocess.run([PY, os.path.join(SK, "scripts", "vec.py"), src, svg],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print("%-10s 失败" % name); continue
        subprocess.run([PY, os.path.join(SK, "scripts", "render_check.py"), svg, png],
                       capture_output=True, text=True)
        v = Image.open(png).convert("RGB")
        if v.size != ref_im.size:
            v = v.resize(ref_im.size, Image.LANCZOS)
        ren = np.asarray(v).astype(np.float32)

        feats = find_small_features(ref)
        kept = sum(1 for f in feats if survived(f, ren))
        sff = 100.0 * kept / max(len(feats), 1)
        mae = float(np.abs(ref - ren).mean())
        mb = os.path.getsize(svg) / 1048576
        print("%-10s %8d %8d %7.1f%% %7.3f %6.2f"
              % (name, len(feats), kept, sff, mae, mb))
        rows[name] = {"src": len(feats), "kept": kept, "sff": round(sff, 1),
                      "mae": round(mae, 3), "mb": round(mb, 2)}

    if rows:
        tot_s = sum(r["src"] for r in rows.values())
        tot_k = sum(r["kept"] for r in rows.values())
        print("\n合计: 源特征 %d，保留 %d，**总 SFF %.1f%%**"
              % (tot_s, tot_k, 100.0 * tot_k / max(tot_s, 1)))
    with open(os.path.join(OUT, "%s.json" % tag), "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "params": extra, "rows": rows}, f,
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
