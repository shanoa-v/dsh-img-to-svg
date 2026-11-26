"""trace_lineart.py -- 线稿在哪一步丢失（逐阶段）

## 背景

392×392 小图线稿保留率只有 39.3%，MAE 24.80。
我试过把 SPECK/MIN_AREA 按尺寸缩放（VEC_ADAPT），完全无效（改善 0.00%）。

说明线稿在这之前就没了。逐个阶段测线稿保留率。

## 判据

线稿像素 = 源图 L<100 且 grad>20
保留 = 该位置在处理后仍 <140（仍明显偏暗）
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
OUT = os.path.join(ROOT, "workspace", "traceline")
os.makedirs(OUT, exist_ok=True)

SRC = os.path.join(ROOT, "input", "new_train", "pixiv",
                   "illust_133681772_20261025_214817.png")
SVG = os.path.join(OUT, "t.svg")

sys.path.insert(0, SK)
sys.argv = [os.path.join(SK, "vec.py"), SRC, SVG]
os.environ["VEC_TITLE"] = "tl"
os.environ["VEC_SEED"] = "42"

import runpy
g = runpy.run_path(os.path.join(SK, "vec.py"), run_name="__main__")

orig = g["orig"]
sm = g["sm"]
edge_sm = g["edge_sm"]
idx = g["idx"]
cen = g["cen"]
H, W = orig.shape[:2]

ref_im = Image.open(SRC).convert("RGB")
ref = np.asarray(ref_im.resize((W, H), Image.LANCZOS)).astype(np.float32)

Lo = ref.mean(axis=2)
gx = np.zeros_like(Lo); gy = np.zeros_like(Lo)
gx[:, 1:-1] = Lo[:, 2:] - Lo[:, :-2]
gy[1:-1, :] = Lo[2:, :] - Lo[:-2, :]
grad = np.hypot(gx, gy)
line = (Lo < 100) & (grad > 20)
n_line = int(line.sum())
print("=== 线稿追踪 %s %dx%d ===" % (os.path.basename(SRC), W, H))
print("线稿像素（源 L<100 且 grad>20）：%d (%.2f%%)" % (n_line, 100 * line.mean()))

if n_line < 100:
    print("线稿太少，退出"); raise SystemExit(0)

# 源图线稿的平均亮度
print("源图线稿处亮度均值 %.1f" % Lo[line].mean())
print()


def check(tag, arr, thr=140):
    """arr: (H,W,3) 或 (H,W)"""
    if arr.ndim == 3:
        L = arr.mean(axis=2)
    else:
        L = arr
    kept = int((L[line] < thr).sum())
    print("%-28s 保留 %6d/%6d = %5.1f%%   线稿处亮度 %.1f   全图亮度 %.1f"
          % (tag, kept, n_line, 100 * kept / n_line,
             L[line].mean(), L.mean()))
    return 100.0 * kept / n_line


check("0. 源图(参照, 目标100%)", ref)
check("1. LANCZOS 工作尺寸 orig", orig)
check("2. 两遍中值 sm", sm)
check("3. 保边混合 edge_sm", edge_sm)

# 量化
rec = cen.astype(np.float32)[idx]
check("4. 量化 cen[idx]", rec)

# 量化后再各阶段（若有）
for k in ("idx_cur", "idx2", "idx_final"):
    v = g.get(k)
    if v is not None:
        check("5. %s" % k, cen.astype(np.float32)[v])

# 最终渲染
import subprocess
png = os.path.join(OUT, "t.png")
subprocess.run([sys.executable, os.path.join(SK, "render_check.py"), SVG, png],
               capture_output=True, cwd=SK)
v = Image.open(png).convert("RGB")
if v.size != (W, H):
    v = v.resize((W, H), Image.LANCZOS)
check("6. 最终渲染", np.asarray(v).astype(np.float32))

print()
print("=== 关键问题：量化用了多少个暗色中心？===")
cenL = cen.mean(axis=1)
print("  调色板 56 色中，亮度<100 的中心：%d 个" % int((cenL < 100).sum()))
print("  调色板 56 色中，亮度<60  的中心：%d 个" % int((cenL < 60).sum()))
print("  调色板最暗中心亮度：%.1f" % cenL.min())
cnt = np.bincount(idx.ravel(), minlength=len(cen))
print("  最暗 5 个中心的使用像素数：%s"
      % [(round(cenL[i], 1), int(cnt[i]))
         for i in np.argsort(cenL)[:5]])

# 源图线稿像素被分到了哪些中心
# 注意：line 是**二维布尔掩码**（与图像同形），idx 也是二维。
# 必须用 idx[line]（二维掩码按位筛选）；
# 早期写成 idx[line.ravel()]（一维掩码）会让 numpy 按扁平化解释，
# 与 idx 的形状不匹配 → IndexError: size of axis is ... but ... boolean axis is ...
# 该脚本因此从未跑通过（前 5 个阶段都正常，必崩在这一行）。
line_flat = line.ravel()
vals, c = np.unique(idx[line], return_counts=True)
print()
print("  线稿像素实际分到的中心（前 6）:")
for k in np.argsort(-c)[:6]:
    ci = vals[k]
    print("    中心#%-2d 亮度%6.1f  覆盖 %5.1f%% 线稿像素"
          % (ci, cenL[ci], 100 * c[k] / line_flat.sum()))
