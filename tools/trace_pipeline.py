"""trace_pipeline.py -- 逐阶段追踪误差累积

## 目的

Noyu 最差图（105819262）最终 MAE 21.271、暗部偏差 +28.43，
但实测：两遍中值只贡献 5.944，保边混合后 5.088。

剩余 ~16 MAE 与 +27 暗部偏差来自哪里？必须逐阶段定位。

## 复刻方式

直接 runpy 跑真实管线，导出以下中间量：
  orig           LANCZOS 缩放后的原图
  sm             两遍 3x3 中值
  edge_sm        保边混合
  idx / cen      量化结果（首次）
  idx_cur/cen_   清理与合并后的结果（若可取得）

分别重建栅格，与原图比。
"""
import os
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
SK = os.path.join(os.environ["DSH_HOME"], "skills", "dsh-img-to-svg", "scripts")
OUT = os.path.join(ROOT, "workspace", "trace")
os.makedirs(OUT, exist_ok=True)

SRC = os.path.join(ROOT, "input", "new_train", "Noyu",
                   "105819262_p0_master1200.webp")
SVG = os.path.join(OUT, "trace.svg")

sys.path.insert(0, SK)
sys.argv = [os.path.join(SK, "vec.py"), SRC, SVG]
os.environ["VEC_TITLE"] = "trace"
os.environ["VEC_SEED"] = "42"

import runpy
g = runpy.run_path(os.path.join(SK, "vec.py"), run_name="__main__")

# 取导出量
def get(k, default=None):
    return g.get(k, default)

orig = get("orig")
sm = get("sm")
edge_sm = get("edge_sm")
idx = get("idx")
cen = get("cen")
H, W = orig.shape[:2]

print("=== 阶段追踪 %s  %dx%d ===" % (os.path.basename(SRC), W, H))
print("可用导出量: %s" % [k for k in ("orig", "sm", "edge_sm", "idx", "cen", "idx_cur", "cen_") if k in g])

# 原图（同尺寸，用于对照）
ref_im = Image.open(SRC).convert("RGB")
ref = np.asarray(ref_im).astype(np.float32)
if ref.shape[:2] != (H, W):
    ref = np.asarray(ref_im.resize((W, H), Image.LANCZOS)).astype(np.float32)


def stats(tag, arr):
    if arr is None:
        print("%-24s 缺失" % tag); return
    if arr.ndim == 2:
        arr3 = np.repeat(arr[..., None], 3, axis=2)
    else:
        arr3 = arr
    e = np.abs(ref - arr3)
    s = arr3 - ref
    L = ref.mean(axis=2)
    dark = L < 80
    print("%-24s MAE %7.3f  暗部偏差 %+7.2f  全局偏差 %+6.2f"
          % (tag, e.mean(), s[..., 0][dark].mean() if dark.any() else 0.0,
             s.mean()))
    return float(e.mean())


print()
stats("LANCZOS 缩放后原图", orig if orig.shape[:2] == (H, W) else None)
stats("1. 两遍中值 sm", sm)
stats("2. 保边混合 edge_sm", edge_sm)

# 量化重建（idx + cen）
if idx is not None and cen is not None:
    rec_q = cen.astype(np.float32)[idx]
    stats("3. 量化 idx/cen", rec_q)

# 若有清理后变量
for k in ("idx_cur", "idx2", "idx_final"):
    v = g.get(k)
    if v is not None and cen is not None:
        try:
            stats("4. 清理后 %s" % k, cen.astype(np.float32)[v])
        except Exception as e:
            print("  重建 %s 失败: %s" % (k, e))

# 最终渲染图
ren_p = os.path.join(OUT, "trace.png")
import subprocess
subprocess.run([sys.executable, os.path.join(SK, "render_check.py"), SVG, ren_p],
               capture_output=True, cwd=SK)
if os.path.exists(ren_p):
    v = Image.open(ren_p).convert("RGB")
    if v.size != (W, H):
        v = v.resize((W, H), Image.LANCZOS)
    stats("5. 最终渲染", np.asarray(v).astype(np.float32))

print("\n=== 各阶段贡献（MAE 增量）===")
print("  若「量化」到「最终渲染」之间有大跳变，说明破坏在描摹/渲染层")

# 额外：量化后按亮度分档看偏差
if idx is not None and cen is not None:
    rec_q = cen.astype(np.float32)[idx]
    Lq = rec_q.mean(axis=2)
    Lo = ref.mean(axis=2)
    print("\n=== 量化后按源亮度分档 ===")
    print("  %-16s %8s %10s %10s" % ("源亮度", "占比", "量化均值", "偏差"))
    for lo, hi in [(0, 40), (40, 60), (60, 80), (80, 100), (100, 140),
                   (140, 180), (180, 220), (220, 256)]:
        m = (Lo >= lo) & (Lo < hi)
        if m.sum() < 100:
            continue
        print("  [%3d,%3d)      %7.1f%% %10.1f %+10.1f"
              % (lo, hi, 100 * m.mean(), Lq[m].mean(), Lq[m].mean() - Lo[m].mean()))
