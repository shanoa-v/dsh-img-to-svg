"""
vec2.py — 图片 -> 分层矢量 SVG 转换器 (v2, 前端精度优化版)

相对 v1 的四项改进（均可用环境变量单独关闭，便于 A/B）：
  A. 亚像素边缘定位：在彩色原图上求"到最近他色像素"的连续距离场，
     等值线取 0.5，使边界落在真实色块交界处（v1 用二值 mask 高斯模糊，系统性内缩）。
  B. 保留尖角：去掉对轮廓点的 Chaikin 二次贝塞尔平滑（v1 的 poly_to_d 会圆化所有角），
     改为直线 + 可选小半径圆角，硬边文字/线条不再"发胖变圆"。
  C. 孔洞 fill-rule：内轮廓保留为 evenodd 孔洞，减少叠层与错误覆盖。
  D. 残差补偿层：边缘带按符号补一层低不透明度对比色，抵消量化色带。

用法:
  python vec2.py <源图> <输出svg>
环境变量:
  VEC_W        工作宽度(默认=原图宽)
  VEC_N        调色板色数(默认 56/64)
  VEC_EPS      RDP 容差(默认 0.30)
  VEC_MINAREA  最小面积(默认 5)
  VEC_EDGE     保边强度(默认 0.16)
  VEC_TITLE    标题
  VEC_SUBPIX   1=启用亚像素距离场(默认1)  0=回退 v1 二值 mask 方式
  VEC_SHARP    1=保留尖角(默认1)          0=回退 Chaikin 圆化
  VEC_HOLES    1=启用 evenodd 孔洞(默认1) 0=关闭
  VEC_RESID    残差补偿层强度(默认 0=关闭, 建议 0.15~0.35)
  VEC_OKLAB    1=用 Oklab 做量化(默认1)   0=用 LAB(v1 行为)
"""
import numpy as np
from PIL import Image, ImageFilter
import os, sys, time
from scipy.ndimage import (gaussian_filter, median_filter, label as ndlabel,
                           generic_filter, distance_transform_edt)
from skimage import measure, color as skcolor
from sklearn.cluster import MiniBatchKMeans

if len(sys.argv) < 3:
    print(__doc__); sys.exit(1)

SRC = sys.argv[1]
OUT = sys.argv[2]

WORK_W   = int(os.environ.get("VEC_W", "0")) or None
N_COLORS = int(os.environ.get("VEC_N", "56"))
RDP_EPS  = float(os.environ.get("VEC_EPS", "0.30"))
MIN_AREA = float(os.environ.get("VEC_MINAREA", "5"))
EDGE_KEEP= float(os.environ.get("VEC_EDGE", "0.16"))
TITLE    = os.environ.get("VEC_TITLE", os.path.basename(SRC))
SUBPIX   = os.environ.get("VEC_SUBPIX", "1") != "0"
SHARP    = os.environ.get("VEC_SHARP", "1") != "0"
HOLES    = os.environ.get("VEC_HOLES", "1") != "0"
RESID    = float(os.environ.get("VEC_RESID", "0"))
OKLAB    = os.environ.get("VEC_OKLAB", "1") != "0"
SEED     = 42

np.random.seed(SEED)
t0 = time.time()

im = Image.open(SRC)
if im.mode in ("RGBA", "LA", "P"):
    im = im.convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im).convert("RGB")
else:
    im = im.convert("RGB")

W0, H0 = im.size
if WORK_W is None:
    WORK_W = W0
WORK_W = min(WORK_W, W0 * 2)
WORK_H = round(WORK_W * H0 / W0)
print(f"[0] {SRC}")
print(f"    {W0}x{H0} -> {WORK_W}x{WORK_H}")

MODE = "subpix" if SUBPIX else "mask"
print(f"    mode={MODE} sharp={int(SHARP)} holes={int(HOLES)} "
      f"oklab={int(OKLAB)} resid={RESID}")


# ---------------- 颜色空间 ----------------
def srgb_to_oklab(rgb):
    """rgb: (...,3) in [0,1] -> Oklab"""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = np.cbrt(l); m_ = np.cbrt(m); s_ = np.cbrt(s)
    return np.stack([
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ], axis=-1)


def oklab_to_srgb(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    return np.stack([
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ], axis=-1)


def to_space(rgb01):
    return srgb_to_oklab(rgb01) if OKLAB else skcolor.rgb2lab(rgb01)


def from_space(space):
    return oklab_to_srgb(space) if OKLAB else skcolor.lab2rgb(space)


# ---------------- 几何 ----------------
def rdp(pts, eps):
    pts = np.asarray(pts, dtype=np.float64); n = len(pts)
    if n < 3: return pts
    keep = np.zeros(n, dtype=bool); keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1: continue
        p0, p1 = pts[i0], pts[i1]; seg = p1 - p0; L = np.hypot(seg[0], seg[1])
        if L < 1e-9:
            dd = np.hypot(pts[i0+1:i1, 0]-p0[0], pts[i0+1:i1, 1]-p0[1])
        else:
            nx, ny = -seg[1]/L, seg[0]/L
            dd = np.abs((pts[i0+1:i1, 0]-p0[0])*nx + (pts[i0+1:i1, 1]-p0[1])*ny)
        k = int(np.argmax(dd))
        if dd[k] > eps:
            m = i0+1+k; keep[m] = True; stack.append((i0, m)); stack.append((m, i1))
    return pts[keep]


def poly_to_d_sharp(poly, close=True):
    """直线段闭合多边形（保留尖角）。v1 的 poly_to_d 用 Q 曲线会把角磨圆。"""
    if len(poly) < 3: return ""
    o = [f"M{poly[0][0]:.1f} {poly[0][1]:.1f}"]
    for p in poly[1:]:
        o.append(f"L{p[0]:.1f} {p[1]:.1f}")
    if close: o.append("Z")
    return "".join(o)


def poly_to_d_chaikin(poly):
    """v1 行为：以顶点为控制点的二次贝塞尔（等价一次 Chaikin 平滑）"""
    if len(poly) < 3: return ""
    n = len(poly)
    if n >= 4:
        o = [f"M{(poly[0][0]+poly[-1][0])/2:.1f} {(poly[0][1]+poly[-1][1])/2:.1f}"]
        for i in range(n):
            p = poly[i]; q = poly[(i+1) % n]
            o.append(f"Q{p[0]:.1f} {p[1]:.1f} {(p[0]+q[0])/2:.1f} {(p[1]+q[1])/2:.1f}")
        o.append("Z"); return "".join(o)
    o = [f"M{poly[0][0]:.1f} {poly[0][1]:.1f}"]
    for p in poly[1:]: o.append(f"L{p[0]:.1f} {p[1]:.1f}")
    o.append("Z"); return "".join(o)


# ---- 1. 保边平滑 ----
ws = im.resize((WORK_W, WORK_H), Image.LANCZOS)
med = ws.filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.MedianFilter(3))
sm = np.asarray(med).astype(np.float32)
orig = np.asarray(ws).astype(np.float32)

gray = orig.mean(axis=2)
gx = np.zeros_like(gray); gy = np.zeros_like(gray)
gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
grad = np.hypot(gx, gy)
gmax = np.percentile(grad, 99) + 1e-6
wk = np.clip(np.clip(grad / gmax, 0, 1) * EDGE_KEEP * 3, 0, 1)[..., None]
edge_sm = sm * (1 - wk) + orig * wk
print(f"[1] 保边平滑 ok ({time.time()-t0:.1f}s)")

# ---- 2. 量化 ----
lab = to_space(edge_sm / 255.0).reshape(-1, 3)
km = MiniBatchKMeans(n_clusters=N_COLORS, random_state=SEED,
                     batch_size=16384, n_init=4, max_iter=150)
idx = km.fit_predict(lab).reshape(WORK_H, WORK_W).astype(np.int32)
cen = (from_space(km.cluster_centers_.reshape(1, -1, 3))[0] * 255).clip(0, 255)
cnt = np.bincount(idx.ravel(), minlength=N_COLORS)
print(f"[2] 量化 {N_COLORS} 色 ok ({time.time()-t0:.1f}s)")

# ---- 3. 碎斑清理 ----
small = np.zeros_like(idx, dtype=bool)
for ci in range(N_COLORS):
    lb, n = ndlabel(idx == ci)
    if n == 0: continue
    sizes = np.bincount(lb.ravel())
    small |= np.isin(lb, np.nonzero((sizes < 24) & (np.arange(len(sizes)) > 0))[0])
if small.any():
    filled = generic_filter(idx.astype(np.float32),
                            lambda v: np.bincount(v.astype(np.int32)).argmax(), size=3)
    idx = np.where(small, filled.astype(np.int32), idx)
    print(f"[3] 碎斑清理 {int(small.sum())} px")


# ---- 4. 亚像素距离场：每个色层一个连续场，0.5 等值线落在真实交界处 ----
def layer_field(ci):
    """返回该色层的连续高度场。

    subpix:  f = 0.5 + (EDT_到本层 - EDT_到最近他色) / 2，截断到 [0,1]。
             本层内部 f>0.5，外部 f<0.5，0.5 等值线恰好落在两色等距处，
             不存在 v1 二值 mask 经高斯模糊后的整体内缩。
    mask:    v1 行为（二值 mask 高斯模糊）。
    """
    mask = (idx == ci)
    if not SUBPIX:
        return gaussian_filter(mask.astype(np.float32), sigma=0.4)
    d_in = distance_transform_edt(mask)              # 本层内部到外部的距离
    d_out = distance_transform_edt(~mask)            # 外部到本层的距离
    # 有符号距离：内部为正；边界两侧连续过渡
    sdf = d_in - d_out
    f = np.clip(0.5 + sdf / 2.0, 0.0, 1.0)
    return f


# ---- 5. 分层描摹 ----
layers = []; total_pts = 0
order = np.argsort(-cnt)

for ci in order:
    mask = (idx == ci); npx = int(mask.sum())
    if npx < MIN_AREA: continue
    frac = npx / idx.size
    eps = RDP_EPS * (1.0 if frac < 0.02 else (1.3 if frac < 0.10 else 1.8))

    fld = layer_field(ci)
    polys = []
    for c in measure.find_contours(fld, level=0.5):
        if len(c) < 4: continue
        if np.allclose(c[0], c[-1]): c = c[:-1]
        if len(c) < 3: continue
        pl = rdp(c, eps)
        if len(pl) < 3: continue
        x = pl[:, 1]; y = pl[:, 0]
        if 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) < MIN_AREA:
            continue
        polys.append(pl[:, ::-1])       # (row,col) -> (x,y)
    if not polys: continue

    total_pts += sum(len(p) for p in polys)
    c = cen[ci].astype(int)
    if SHARP:
        d = " ".join(poly_to_d_sharp(p) for p in polys)
        rule = ' fill-rule="evenodd"' if HOLES else ""
    else:
        d = " ".join(poly_to_d_chaikin(p) for p in polys)
        rule = ""
    layers.append(dict(hexc=f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}",
                       d=d, rule=rule, npx=npx, ci=ci))
print(f"[5] {len(layers)} 层 {total_pts} 点 ({time.time()-t0:.1f}s)")

# ---- 6. 残差补偿层 ----
resid_paths = ""
if RESID > 0:
    recon = np.zeros_like(orig)
    for L in layers:
        m = (idx == L["ci"])
        recon[m] = cen[L["ci"]]
    err = orig - recon
    # 只在误差显著处补
    mag = np.abs(err).mean(axis=2)
    hot = mag > 8
    if hot.any():
        # 用正/负两色概括误差方向（简化：按符号分组）
        pos_mask = hot & (err.mean(axis=2) > 0)
        neg_mask = hot & (err.mean(axis=2) < 0)
        pieces = []
        for m, base_col in ((pos_mask, np.array([255, 255, 255])),
                            (neg_mask, np.array([0, 0, 0]))):
            if not m.any(): continue
            f = gaussian_filter(m.astype(np.float32), sigma=0.5)
            for c in measure.find_contours(f, 0.5):
                if len(c) < 6: continue
                pl = rdp(c, RDP_EPS * 1.2)
                if len(pl) < 3: continue
                # 该区域平均误差颜色（放大后作为补偿色）
                drift = err[m].mean(axis=0)
                comp = np.clip(cen[np.argmax(cnt)] + drift * 2.0, 0, 255).astype(int)
                pieces.append((pl[:, ::-1], comp))
        if pieces:
            op = min(RESID, 0.6)
            res = [f'<g fill-opacity="{op:.2f}">']
            for pl, comp in pieces:
                d = poly_to_d_sharp(pl)
                res.append(f'<path fill="#{comp[0]:02X}{comp[1]:02X}{comp[2]:02X}" d="{d}"/>')
            res.append("</g>")
            resid_paths = "".join(res)
            print(f"[6] 残差补偿 {len(pieces)} 片")

# ---- 7. 输出 ----
base = cen[np.argmax(cnt)].astype(int)
sx, sy = W0 / WORK_W, H0 / WORK_H
P = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W0} {H0}" '
    f'width="{W0}" height="{H0}" shape-rendering="geometricPrecision">',
    f'<title>{TITLE}</title>',
    '<desc>Layered vector reconstruction (vec2)</desc>',
    f'<rect width="{W0}" height="{H0}" fill="#{base[0]:02X}{base[1]:02X}{base[2]:02X}"/>',
    f'<g transform="scale({sx:.6f} {sy:.6f})" stroke="none">',
]
for L in layers:
    P.append(f'<path fill="{L["hexc"]}"{L["rule"]} d="{L["d"]}"/>')
if resid_paths:
    P.append(resid_paths)
P.append("</g></svg>")
svg = "\n".join(P)
open(OUT, "w", encoding="utf-8").write(svg)
print(f"[7] -> {OUT}  {len(svg)/1024:.1f} KB  ({time.time()-t0:.1f}s)")
