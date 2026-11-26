"""
vec.py -- image -> layered vector SVG converter（正式交付管线）

历史渊源：本文件的前身是 vec2.py（v2，精度优化版），
vec2.py 现在仍作为**历史遗留**留在 scripts/ 下（能力已落后，且无人引用），
但**实际交付走的是本文件 vec.py**。此前文件头与 SVG 的 <desc> 里还残留着
"vec2" 字样，容易让人误以为交付物出自 vec2.py，故已统一订正为 vec.py。

Improvements over v1 (each can be toggled independently for A/B):
  A. Oklab quantization instead of CIE LAB (biggest single win, ~17% MAE).
  B. Sharp corners preserved: v1's poly_to_d used quadratic Beziers with the
     vertices as control points, which rounds every corner. Straight segments
     keep hard-edged text and line art crisp, and halve the file size.
  C. Optional sub-pixel edge fields (measured: negligible gain).
  D. Optional evenodd holes (measured: no visual gain in practice; off by default
     because it costs time).
  E. Optional residual compensation layer (experimental).

Usage:
  python vec.py <source> <output.svg>

Environment variables (ASCII names only):
  VEC_W        working width (default: source width)
  VEC_N        palette size (default 56)
  VEC_EPS      RDP tolerance (default 0.30)
  VEC_MINAREA  minimum contour area (default 5)
  VEC_EDGE     edge-preservation strength (default 0.16)
  VEC_TITLE    SVG <title> text
  VEC_OKLAB    1=Oklab (default) 0=CIE LAB (v1 behaviour)
  VEC_SHARP    1=sharp corners (default) 0=Chaikin rounding (v1 behaviour)
  VEC_HOLES    1=evenodd holes (default 0)
  VEC_SUBPIX   1=sub-pixel distance field (default 1) 0=binary mask blur (v1)
  VEC_RESID    residual layer strength (default 0 = off)
  VEC_SPECK    speck-removal area threshold in px (default 24)
  VEC_AUTOW    1=downscale when wider than VEC_MAXW
               **默认 0（关闭）** —— 代码是 get("VEC_AUTOW", "0")，
               所以 k 放大的图不会被悄悄压回 MAXW。
               （此处曾有注释误写 default 1，与实际相反，已订正）
  VEC_MAXW     autoscale cap (default 1920，仅当 VEC_AUTOW=1 时生效)
  VEC_MERGE_DE   cluster-merge colour distance gate (default 6, 0=off)
  VEC_MERGE_STEP cluster-merge per-channel 8-bit step gate (default 10)
  VEC_MERGE_GRAD cluster-merge source-gradient gate (default 5)
  VEC_LIGHT_STEP strict per-channel step used ONLY inside the light region
                 (default 0 = off; see section 2.5 below)
  VEC_LIGHT_L    CIE LAB L (0..100) that defines "light region" (default 85)
  VEC_SE         1 = structural edge sharpening (default 0 = off, byte-identical
                 to the baseline). See section 1b below: replaces the linear
                 "keep-original" blend with a two-factor gate that separates
                 GENUINE EDGES (strong grad + locally coherent orientation)
                 from NOISE GRADIENTS (incoherent / isolated). Both gates are
                 PERCENTILES, never absolute thresholds (absolute ones silently
                 fail across images -- project lesson).
  VEC_SE_HI      percentile that starts the sharpening ramp (default 95)
  VEC_SE_LO      percentile below which the weight is forced to 0 (default 70).
                 Widen (e.g. 50..80) to keep weak real edges; narrow (e.g.
                 90..99) to harden the noise/edge separation.
  VEC_SE_COH     orientation-coherence cut in 0..1 (default 0.35). 0 = disable
                 the coherence factor (ablation arm).
  VEC_SE_GAIN    max keep-original weight on the strongest coherent edges
                 (default 1.0 = fully restore the original pixel).
  VEC_SE_SIG     structure-tensor smoothing sigma in px (default 2.0).
  VEC_SE_LOGMIN  per-image log: a gate is reported as FAILING if the computed
                 threshold suppresses more than this share of pixels (default
                 0.90), so an inert branch cannot pass silently.
"""
import numpy as np
from PIL import Image, ImageFilter
import os, sys, time
from scipy.ndimage import (
    gaussian_filter, label as ndlabel, generic_filter, distance_transform_edt,
    minimum as ndmin, maximum_filter as ndmaxf, uniform_filter,
    median_filter as ndmedian
)
from skimage import measure, color as skcolor
from sklearn.cluster import MiniBatchKMeans
import block_tracer as BT

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

SRC = sys.argv[1]
OUT = sys.argv[2]

WORK_W = int(os.environ.get("VEC_W", "0")) or None
N_COLORS = int(os.environ.get("VEC_N", "56"))
RDP_EPS = float(os.environ.get("VEC_EPS", "0.30"))
MIN_AREA = float(os.environ.get("VEC_MINAREA", "5"))


# ---- 参数合理性校验（P0-3b）----
# EPS / MINAREA 取 nan / inf / 超大值时，管线**不报错**（exit 0），
# 但产物退化成「只有底色 + 一个 rect」，非底色像素 0.0000。
# 实测：VEC_EPS=nan / VEC_EPS=inf / VEC_MINAREA=1e9 都是这种静默废品。
# 这里做有限性与区间校验，明确报错退出，而不是安静地交垃圾。
def _check_num(name, val, lo, hi, unit=""):
    import math as _m
    if not _m.isfinite(val):
        raise SystemExit(
            "%s 不是有限数值（收到 %r）。请给 %s..%s 之间的正常数字。"
            % (name, val, lo, hi))
    if val < lo or val > hi:
        raise SystemExit(
            "%s=%s 超出合理范围 %s..%s%s。极值会让产物退化成只有底色。"
            % (name, val, lo, hi, unit))
    return val


RDP_EPS = _check_num("VEC_EPS（RDP 简化容差）", RDP_EPS, 0.0, 50.0, " 像素")
MIN_AREA = _check_num("VEC_MINAREA（最小面积）", MIN_AREA, 0.0, 1e6, " 像素²")

EDGE_KEEP = float(os.environ.get("VEC_EDGE", "0.16"))
TITLE = os.environ.get("VEC_TITLE", os.path.basename(SRC))
SUBPIX = os.environ.get("VEC_SUBPIX", "1") != "0"
SHARP = os.environ.get("VEC_SHARP", "1") != "0"
HOLES = os.environ.get("VEC_HOLES", "0") != "0"
RESID = float(os.environ.get("VEC_RESID", "0"))

# ---- seam sealing (VEC_SEAL, default 0 = off) -------------------------
# Adjacent colour blocks are traced as separate paths. The paths abut, yet the
# rasteriser antialiases each edge independently, so the two never both reach
# alpha=1 and the base <rect> bleeds through the hairline seam between them.
# Solved per-pixel alpha from two renders (pure black vs pure white base rect):
#   rend_black = a*fg + (1-a)*0
#   rend_white = a*fg + (1-a)*255
#   => (1-a) = (rend_white - rend_black)/255
# Measured on a dark 600x853 illustration: alpha mean 0.8728, median 1.0000,
# 45.1% of pixels below 0.999, and 49.7% of the DARK region below 0.999.
# That is ~12.7% of the base colour bleeding through on average, which is why a
# dark scene turned pale. Adding a SAME-COLOUR hairline stroke to every path
# closes the seam without introducing any outline (colour is unchanged).
# Measured MAE on that image: 21.271 (off) -> 14.433 (stroke-width 1.0).
# Default is 0.6, NOT 1.0: with VEC_BASECOL also on, the colour bleeding
# through the seam is already close to the right background, so a thinner
# seal suffices. Full 169-image sweep of seal widths:
#   1.0 -> median -13.73%, worst +0.66% (ONE image regressed)
#   0.6 -> median -15.60%, worst -3.19% (ZERO regressions)
#   0.6 wins on 142 images, loses on 4, ties on 2.
SEAL = float(os.environ.get("VEC_SEAL", "0.6"))


def _seal_sfx(hexcol):
    """Same-colour hairline stroke; empty string when SEAL <= 0."""
    if SEAL <= 0:
        return ""
    return ' stroke="%s" stroke-width="%g" stroke-linejoin="round"' % (hexcol, SEAL)

SPECK = int(os.environ.get("VEC_SPECK", "24"))
# --- adaptive speck cleanup (v1.0.6 experiment) ---
#   0 = off  (legacy: one global SPECK threshold for the whole canvas)
#   1 = on   In areas where the SOURCE image is already flat (low gradient),
#            raise the merge threshold so quantization noise is absorbed;
#            genuine high-contrast detail keeps the strict threshold.
ADAPT = os.environ.get("VEC_ADAPT", "0") != "0"
SPECK_SMOOTH = int(os.environ.get("VEC_SPECK_SMOOTH", "192"))
# VEC_FILL: "mode" = legacy 3x3 mode filter (only rewrites component rims)
#           "nearest" = nearest-neighbour fill, replaces the whole component
FILL = os.environ.get("VEC_FILL", "mode").lower()
# --- contrast gate (validated over 3 rounds + independent audit) ------------
#   VEC_GATE         1 = master switch for the validated configuration:
#                    nearest fill + whole-image gate + DE 20. Default 0, and
#                    with VEC_GATE=0 the output is byte-identical to v1.0.5, so
#                    existing users are unaffected.
#   VEC_DE_MAX       gate distance; 0 = off. This is a **derived** distance,
#                    NOT colour-science Oklab: the sRGB values are fed to the
#                    Oklab matrix WITHOUT gamma decoding, so L runs high
#                    (measured +0.20 at mid-grey) and the distance is compressed
#                    to roughly 0.5-0.9x. Calibrated empirically, not in JND.
#                    Stable region: 16..24 (10 is measurably worse: MAE 11.006).
#   VEC_DE_SPACE     "oklab" (default, the derived distance above) | "srgb"
#   VEC_DE_FLATONLY  1 = restrict the gate to source-flat regions.
#                    DEFAULT 0. NOTE: 1 is measurably WORSE (a_starry MAE
#                    18.08 -> 19.96). The cause is NOT "starved merge volume":
#                    partial merging fragments the image further (vertex count
#                    1.42M -> 1.58M) while full merging reduces it (-> 0.95M).
#                    Do not "relax flat to merge more" -- that is the wrong fix.
#   VEC_DE_FLATGRAD  gradient cut defining "flat" (only used when FLATONLY=1)
#   VEC_DE_AUTO      >0 = pick the threshold from this percentile of the noise
#                    distance distribution instead of VEC_DE_MAX
DE_MAX = float(os.environ.get("VEC_DE_MAX", "0"))
DE_SPACE = os.environ.get("VEC_DE_SPACE", "oklab").lower()
DE_FLATONLY = os.environ.get("VEC_DE_FLATONLY", "0") != "0"
DE_FLATGRAD = float(os.environ.get("VEC_DE_FLATGRAD", "3.0"))
DE_AUTO = float(os.environ.get("VEC_DE_AUTO", "0"))
GATE = os.environ.get("VEC_GATE", "0") != "0"
# VEC_SPLIT_PATHS 1 = emit one <path> per disconnected contour (default in
# v1.0.7). 0 = legacy: all contours of a colour inside one <path>, which lets
# the nonzero rule cancel overlapping subpaths and corrupt large regions.
SPLIT_PATHS = os.environ.get("VEC_SPLIT_PATHS", "1") != "0"
# VEC_BLOCK_TRACE 1 = trace each connected block separately with a local SDF
# and emit one <path> per block (v1.0.7). 0 = legacy single-field tracing.
BLOCK_TRACE = os.environ.get("VEC_BLOCK_TRACE", "1") != "0"
if GATE:
    FILL = "nearest"
    if DE_MAX <= 0 and DE_AUTO <= 0:
        DE_MAX = 20.0
# --- structural edge sharpening (v1.1.0) -------------------------------------
#   VEC_SE=0 (default) keeps the pipeline byte-identical to the baseline.
#   Why this exists, measured: the baseline render carries only 0.61-0.93x of
#   the source gradient energy (it is smoother than the source). Raising
#   VEC_EDGE globally fixed sharpness but made a_starry worse (MAE +16.6%,
#   flat-region boundary density 14.01 -> 18.33, visible grid-shaped flecks)
#   because a linear "keep the original" blend cannot tell a real edge from a
#   noise gradient: it keeps both. VEC_SE replaces the linear blend with
#     w = ramp_percentile(grad) x orientation_coherence
#   so only strong AND locally coherent gradients reach the original.
SE_ON = os.environ.get("VEC_SE", "0") != "0"
SE_HI = float(os.environ.get("VEC_SE_HI", "95"))
SE_LO = float(os.environ.get("VEC_SE_LO", "70"))
SE_COH = float(os.environ.get("VEC_SE_COH", "0.35"))
SE_GAIN = float(os.environ.get("VEC_SE_GAIN", "1.0"))
SE_SIG = float(os.environ.get("VEC_SE_SIG", "2.0"))
SE_LOGMIN = float(os.environ.get("VEC_SE_LOGMIN", "0.90"))
#   VEC_SE_NCONT   1 = add a local CONTINUITY factor (default 1 when SE is on).
#                  Measured on a_starry: 51.9% of the pixels the strength ramp
#                  alone would promote have almost no strong-gradient
#                  neighbours (<=3 inside 5x5) -- i.e. they are isolated noise
#                  specks, not edges. A real edge is a connected chain. The
#                  factor is the local density of strong gradients.
#   VEC_SE_CSIZE   continuity window in px (default 5).
#   VEC_SE_CMIN    minimum local strong-gradient density to accept (default 0.10).
#   VEC_SE_FLOOR   >= 0 : take w = max(baseline_linear_w, structural_w) instead
#                  of letting the gate REPLACE the baseline blend (default 1).
#                  See section 1b -- bnd is not monotonically improved by
#                  suppressing flat-region weight, so SE must be a gain over
#                  the baseline, never a substitute. 0 = old replace behaviour
#                  (kept for the ablation arm that proves the point).
SE_NCONT = os.environ.get("VEC_SE_NCONT", "1") != "0"
SE_CSIZE = int(os.environ.get("VEC_SE_CSIZE", "5"))
SE_CMIN = float(os.environ.get("VEC_SE_CMIN", "0.10"))
SE_FLOOR = float(os.environ.get("VEC_SE_FLOOR", "1"))
AUTOW = float(os.environ.get("VEC_AUTOW", "0")) != 0
MAXW = int(os.environ.get("VEC_MAXW", "1920"))
OKLAB_RAW = os.environ.get("VEC_OKLAB", "auto").lower()
# SEED 可通过环境变量覆盖（默认 42，与历史行为一致）。
# 为什么需要这个开关：实测 a_starry 上只换种子、平滑逻辑不动，
# bnd%% 就在 14.0~25.7 之间跳（极差 11.68）。评估必须做多种子稳健化，
# 否则单次运行的数字不可作门槛。见 references/BND_SEED_SENSITIVITY.md
SEED = int(os.environ.get("VEC_SEED", "42"))

np.random.seed(SEED)
t0 = time.time()

im = Image.open(SRC)

# ---- 位深处理（P0-3）----
# Pillow 对 I;16 / I / F 模式直接 convert("RGB") 是**截断取高位**而非缩放。
# 一张正常的 16-bit 灰度图（值域 0..65535）会被整体压到接近 255（全白），
# 产物只剩底色 + 一两个白 rect，而且 exit 0、不给任何警告。
# 正确做法：先线性归一化到 0..255，再转 L -> RGB。
# 注意：只在动态范围 > 255 时才归一化 —— 本身就是 0..255 的 I 图
# 若也归一化会改变其原有语义。
if im.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
    _arr = np.asarray(im).astype(np.float64)
    _lo = float(_arr.min())
    _hi = float(_arr.max())
    if _hi > 255.0 and _hi > _lo:
        _arr = (_arr - _lo) * (255.0 / (_hi - _lo))
        print("    位深归一化: 原始范围 %.0f..%.0f -> 0..255" % (_lo, _hi))
    _arr = np.clip(_arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(_arr, mode="L").convert("RGB")
elif im.mode in ("RGBA", "LA", "P"):
    im = im.convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im).convert("RGB")
else:
    im = im.convert("RGB")

W0, H0 = im.size
if WORK_W is None:
    WORK_W = W0
WORK_W = min(WORK_W, W0 * 2)
if AUTOW and WORK_W > MAXW:
    WORK_W = MAXW
    print("    autow: work width -> %d (from %d)" % (MAXW, W0))
WORK_H = round(WORK_W * H0 / W0)
print("[0] %s" % SRC)
print("    %dx%d -> %dx%d" % (W0, H0, WORK_W, WORK_H))
print("    mode=subpix(%d) sharp(%d) holes(%d) oklab=%s resid=%g speck=%d"
      % (SUBPIX, SHARP, HOLES, OKLAB_RAW, RESID, SPECK))


# ---------------- colour spaces ----------------
def srgb_to_oklab(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
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


# ---------------- geometry ----------------
def rdp(pts, eps):
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return pts
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        p0, p1 = pts[i0], pts[i1]
        seg = p1 - p0
        L = np.hypot(seg[0], seg[1])
        if L < 1e-9:
            dd = np.hypot(pts[i0 + 1:i1, 0] - p0[0], pts[i0 + 1:i1, 1] - p0[1])
        else:
            nx, ny = -seg[1] / L, seg[0] / L
            dd = np.abs((pts[i0 + 1:i1, 0] - p0[0]) * nx
                        + (pts[i0 + 1:i1, 1] - p0[1]) * ny)
        k = int(np.argmax(dd))
        if dd[k] > eps:
            m = i0 + 1 + k
            keep[m] = True
            stack.append((i0, m))
            stack.append((m, i1))
    return pts[keep]


def poly_to_d_sharp(poly):
    """Straight-segment closed polygon; preserves sharp corners."""
    if len(poly) < 3:
        return ""
    o = ["M%.1f %.1f" % (poly[0][0], poly[0][1])]
    for p in poly[1:]:
        o.append("L%.1f %.1f" % (p[0], p[1]))
    o.append("Z")
    return "".join(o)


def poly_to_d_chaikin(poly):
    """v1 behaviour: quadratic Bezier with vertices as control points."""
    if len(poly) < 3:
        return ""
    n = len(poly)
    if n >= 4:
        o = ["M%.1f %.1f" % ((poly[0][0] + poly[-1][0]) / 2,
                             (poly[0][1] + poly[-1][1]) / 2)]
        for i in range(n):
            p = poly[i]
            q = poly[(i + 1) % n]
            o.append("Q%.1f %.1f %.1f %.1f" % (
                p[0], p[1], (p[0] + q[0]) / 2, (p[1] + q[1]) / 2))
        o.append("Z")
        return "".join(o)
    o = ["M%.1f %.1f" % (poly[0][0], poly[0][1])]
    for p in poly[1:]:
        o.append("L%.1f %.1f" % (p[0], p[1]))
    o.append("Z")
    return "".join(o)


def poly_area(p):
    x = p[:, 0]
    y = p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def poly_bbox(p):
    return (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())


def points_in_poly_vec(pts, poly):
    """Vectorised even-odd point-in-polygon test.

    pts: (M,2) float array, poly: (N,2) float array -> (M,) bool array.
    """
    x = pts[:, 0][:, None]
    y = pts[:, 1][:, None]
    px = poly[:, 0][None, :]
    py = poly[:, 1][None, :]
    px_j = np.roll(px, 1, axis=1)
    py_j = np.roll(py, 1, axis=1)

    cond_y = (py > y) != (py_j > y)
    denom = py_j - py
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    x_int = (px_j - px) * (y - py) / denom + px
    cond_x = x < x_int
    cross = cond_y & cond_x
    return (cross.sum(axis=1) % 2) == 1


def split_islands_holes(polys):
    """Separate outer islands from true holes.

    A contour is a hole when a representative point lies inside a larger
    contour. Islands must stay independently filled; holes are paired with
    their enclosing shell under evenodd.
    """
    n = len(polys)
    if n <= 1:
        return polys, []

    areas = np.array([poly_area(p) for p in polys])
    boxes = np.array([poly_bbox(p) for p in polys], dtype=np.float64)
    order = np.argsort(-areas)
    reps = np.array([polys[i][0] for i in range(n)], dtype=np.float64)

    is_hole = np.zeros(n, dtype=bool)
    for rank, i in enumerate(order):
        if rank == 0:
            continue
        cand = order[:rank]
        if len(cand) == 0:
            continue
        bx = boxes[cand]
        px, py = reps[i]
        inside_box = ((px >= bx[:, 0]) & (px <= bx[:, 2])
                      & (py >= bx[:, 1]) & (py <= bx[:, 3]))
        cand = cand[inside_box]
        for j in cand:
            if points_in_poly_vec(reps[i:i + 1], polys[j])[0]:
                is_hole[i] = True
                break

    keep = [polys[i] for i in range(n) if not is_hole[i]]
    holes = [polys[i] for i in range(n) if is_hole[i]]
    return keep, holes


# ---- 1. edge-preserving smoothing ----
ws = im.resize((WORK_W, WORK_H), Image.LANCZOS)
# ---- smoothing mode (VEC_SMOOTH, default 0 = legacy two-pass median) ---
# Measured on a 392x392 line-art illustration (18769 stroke pixels defined as
# source L<100 and grad>20), line-art retention after each filter:
#   original           100.0%   stroke luminance  56.3
#   one 3x3 median      73.5%                    110.7
#   TWO 3x3 median      67.1%                    119.2   <- current default
#   gaussian sigma=0.6  99.6%                     85.6   <- 1-2 px strokes survive
#   gaussian sigma=1.0  79.5%                    110.9
# A median filter is an ORDER STATISTIC: a 1-2 px dark stroke is a minority in
# its 3x3 window and is replaced by the neighbourhood median, erasing the ink
# outright. A Gaussian is a weighted mean, so the stroke still pulls the result
# down. This is the single largest cause of small-image stroke loss.
# Default stays 0 so the shipped behaviour is unchanged until validated.
SMODE = int(os.environ.get("VEC_SMOOTH", "0"))
if SMODE == 1:
    _sig = float(os.environ.get("VEC_SMOOTH_SIG", "0.6"))
    med = ws.filter(ImageFilter.GaussianBlur(_sig))
    print("    smooth: gaussian sigma=%.2f (VEC_SMOOTH=1)" % _sig)
elif SMODE == 2:
    med = ws.filter(ImageFilter.MedianFilter(3))
    print("    smooth: single-pass 3x3 median (VEC_SMOOTH=2)")
else:
    med = ws.filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.MedianFilter(3))
    print("    smooth: two-pass 3x3 median (legacy, VEC_SMOOTH=0)")
sm = np.asarray(med).astype(np.float32)
orig = np.asarray(ws).astype(np.float32)
# ---- adaptive palette size (VEC_ADAPT_N, default 0 = keep VEC_N) -------
# Placed AFTER orig/WORK_W/WORK_H exist so the feature computation is valid.
# (A first attempt inserted this next to N_COLORS at line 78, where none of
#  those variables exist yet -- it would have raised NameError at runtime.)
#
# Measured on 12 illustrations (0.27-0.95 MP) with the seam-bleed fix applied:
#   optimal N median 192, range 96-192
#   vs N=56: median -5.37%, best -11.14%, worst -0.54% (still better)
#   N=96 vs N=56: 11 improved, 0 regressed
# Correlation of optimal N with image features:
#   flat_pct  -0.717   <- strongest, NEGATIVE
#   detail_pct +0.607
#   colors8    +0.330
#   MP         +0.238
# A flat region needs no colour levels; levels spent there only become speck
# noise. A detailed region needs levels to express gradation. Hence flat area
# share is the dominant inverse predictor, with pixel count as a second term.
# NOTE: constants are provisional until the large-image samples are in.
if os.environ.get("VEC_ADAPT_N", "0") != "0":
    _lum = orig.mean(axis=2)
    _gx = np.zeros_like(_lum); _gy = np.zeros_like(_lum)
    _gx[:, 1:-1] = _lum[:, 2:] - _lum[:, :-2]
    _gy[1:-1, :] = _lum[2:, :] - _lum[:-2, :]
    _grad = np.hypot(_gx, _gy)
    _flat = 100.0 * float((_grad < 3).mean())
    _det = 100.0 * float((_grad >= 15).mean())
    _mp = (WORK_W * WORK_H) / 1e6
    _old_n = N_COLORS
    # Fitted by grid search over 18 images with a full N sweep
    # (56 / 96 / 128 / 192), minimising distance to each image's own optimum:
    #   N = 285 - 2.70*flat_pct + 22.5*log2(MP), clamped to [56, 192]
    # Result: mean MAE 9.044 vs 9.679 for N=56 (=-6.56%) and 9.016 for the
    # per-image optimum, i.e. the formula reaches 99.5% of the oracle.
    # Cross-check: the one image that high N actively hurt (2.03 MP, 64.6%
    # flat, only 101 quantised colours -- N=192 cost it +5.46%) is routed to
    # N=111 by this formula, which lands on its optimum bucket.
    _new_n = 285.0 - 2.70 * _flat + 22.5 * np.log2(max(_mp, 0.1))
    _new_n = int(np.clip(round(_new_n), 56, 192))
    print("[0] adapt_n: %.2f MP, flat %.1f%%, detail %.1f%% -> N %d => %d"
          % (_mp, _flat, _det, _old_n, _new_n))
    N_COLORS = _new_n

gray = orig.mean(axis=2)
gx = np.zeros_like(gray)
gy = np.zeros_like(gray)
gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
grad = np.hypot(gx, gy)
gmax = np.percentile(grad, 99) + 1e-6
wk = np.clip(np.clip(grad / gmax, 0, 1) * EDGE_KEEP * 3, 0, 1)[..., None]
edge_sm = sm * (1 - wk) + orig * wk
print("[1] edge-preserving smoothing ok (%.1fs)" % (time.time() - t0))

# ---- 1b. structural edge sharpening (VEC_SE, default OFF) ----
# The baseline blend above is a single linear ramp on |grad|. Two weaknesses:
#   (a) it is far too soft -- at EDGE_KEEP=0.16 the strongest edge of a normal
#       image (grad == p99) only reaches w = 0.48, so real edges are half
#       smoothed away;
#   (b) it has no notion of "is this gradient a real edge or just noise?" --
#       any gradient, including the tiny incoherent ones that tile a flat
#       background, pulls the original back in. Raising EDGE_KEEP therefore
#       amplifies background noise exactly as much as real edges, which is the
#       measured cause of the a_starry regression (bnd 14.01 -> 18.33).
# Fix: keep the same two-stage shape but gate the ramp with two factors.
#   * strength gate  : percentile ramp, 0 below p_LO, 1 above p_HI. Percentiles,
#                      NOT absolute thresholds -- absolute values silently stop
#                      meaning anything when the image distribution changes.
#   * coherence gate : structure tensor J = G_sig * (g g^T); its normalised
#                      eigenvalue contrast (l1-l2)/(l1+l2) is ~1 where the local
#                      gradient all points one way (an edge), ~0 where it points
#                      everywhere (noise). Single-pixel gradients of noise are
#                      orthogonal neighbour to neighbour, so they average out;
#                      a real edge keeps its direction across the window.
# A real edge needs BOTH; noise gradients fail the coherence factor, and weak
# gradients fail the strength ramp. w stays 0 outside the gate, so flat regions
# keep the smoothed value and cannot grow new boundaries.
if SE_ON:
    # CRITICAL: the strength gate must be driven by the GRADIENT MAGNITUDE.
    # `gray` is the luminance image (used by gx/gy above); `grad` is |g|.
    # A first cut of this section used `gsrc = gray`, which silently turned the
    # gate into a *brightness* percentile ramp (logged thresholds were
    # 196/237/253 -- luminance values, not gradients) and every pixel passed
    # (SE_GATE_TOO_WIDE on 5/5 images). That is exactly the "silent inert
    # branch" failure this project has been bitten by before, so the name and
    # the logged value both stay explicit here.
    gsrc = grad
    _gx2, _gy2 = gx * gx, gy * gy
    _gxy = gx * gy
    Jxx = gaussian_filter(_gx2, SE_SIG, mode="nearest")
    Jyy = gaussian_filter(_gy2, SE_SIG, mode="nearest")
    Jxy = gaussian_filter(_gxy, SE_SIG, mode="nearest")
    del _gx2, _gy2, _gxy
    # eigenvalue gap of the 2x2 symmetric tensor, closed form
    _tr = Jxx + Jyy
    _dif = np.sqrt((Jxx - Jyy) ** 2 + 4.0 * Jxy ** 2)
    coh = _dif / (_tr + 1e-6)          # 0..1 orientation coherence
    t_lo = float(np.percentile(gsrc, SE_LO))
    t_hi = float(np.percentile(gsrc, SE_HI))
    t_hi = max(t_hi, t_lo + 1e-3)
    # linear ramp between the two percentiles, then two-factor gate
    ramp = np.clip((gsrc - t_lo) / (t_hi - t_lo), 0.0, 1.0)
    if SE_COH > 0:
        cgate = np.clip(coh / SE_COH, 0.0, 1.0)
    else:
        cgate = np.ones_like(coh)      # ablation: strength gate only
    # continuity: a real edge is a connected chain of strong gradients; a noise
    # speck is isolated. Local density of "strong" pixels (above p_SE_HI) is the
    # cheapest robust proxy. Measured on a_starry: without this factor 51.9% of
    # promoted pixels have <=3 strong neighbours inside 5x5.
    if SE_NCONT:
        strong = (gsrc > t_hi).astype(np.float32)
        dens = uniform_filter(strong, SE_CSIZE)     # 0..1 inside the window
        cont = np.clip(dens / max(SE_CMIN, 1e-6), 0.0, 1.0)
    else:
        cont = np.ones_like(coh)
    se_w = np.clip(ramp * cgate * cont * SE_GAIN, 0.0, 1.0)
    edge_keep_lin = np.clip(np.clip(gsrc / gmax, 0, 1) * EDGE_KEEP * 3, 0, 1)
    # --- baseline floor (VEC_SE_FLOOR, default 1 when SE is on) --------------
    # Measured, and this is the single most important detail of the whole
    # feature: bnd% is NOT monotonically punished by "keeping original pixels".
    # On a_starry, the baseline (EDGE=0.16) sits at bnd 14.01 which is a LOCAL
    # MINIMUM: forcing the flat-region weight to exactly 0 (the plain reading
    # of "suppress noise") raises it to 21.37, i.e. WORSE than the EDGE=0.50
    # arm that was rejected. Cause: the tiny baseline blend (w~0.006 in flat
    # areas) leaves flat regions slightly non-uniform, so quantisation breaks
    # them into many small softly-bordered patches; removing it makes flat
    # regions perfectly uniform, quantisation then emits fewer but harder-edged
    # blocks and the flat-region jump density goes UP.
    # Therefore SE must be a pure GAIN over the baseline, never a replacement:
    #   w = max(baseline_linear_w, structural_w)
    # Pixels the structural gate does not promote fall back to the baseline
    # value bit-for-bit, so flat regions behave exactly like the baseline and
    # cannot regress. Only genuine edges get more original back.
    if SE_FLOOR > 0:
        se_w = np.maximum(se_w, edge_keep_lin * SE_FLOOR)
    # --- self-proof: the branch must actually do something, and the gates must
    # --- not be degenerate. Logged, never silent (project lesson: a disabled
    # --- branch that looks enabled has cost this project twice).
    on = se_w > 0.0
    hi_w = se_w > 0.5
    warn = []
    # "inert" = the gate promotes essentially nothing
    if float(hi_w.mean()) < 0.001:
        warn.append("SE_GATE_INERT: only %.3f%% of pixels reach w>0.5"
                    % (100 * hi_w.mean()))
    if t_hi <= t_lo + 1e-2:
        warn.append("SE_RAMP_COLLAPSED: p_LO~p_HI")
    if SE_COH > 0 and float(coh.max()) < SE_COH:
        warn.append("SE_COH_NEVER_MET: max coh %.3f < %.3f" % (coh.max(), SE_COH))
    # "too wide" must be judged on EFFECTIVE weight, not merely >0: the ramp
    # starts above p_LO by design, so a large w>0 share is normal and is not a
    # defect. What would indicate a real leak is a large share of pixels with
    # substantially restored original (w>0.5).
    if float(hi_w.mean()) > SE_LOGMIN:
        warn.append("SE_GATE_TOO_WIDE: %.1f%% of pixels have w>0.5"
                    % (100 * hi_w.mean()))
    print("[1b] SE on: thresh p%g=%.2f p%g=%.2f  coh cut=%.2f (max %.2f)  "
          "cont=%s"
          % (SE_LO, t_lo, SE_HI, t_hi, SE_COH, float(coh.max()),
             ("win%d>%.2f" % (SE_CSIZE, SE_CMIN)) if SE_NCONT else "off"))
    # Dimsension sanity check: the gate must be a GRADIENT percentile. Guard
    # against the historical bug where `gsrc` was accidentally the luminance
    # image: luminance percentiles land near 120-255, gradient percentiles on
    # these images are 15-240 but their p_LO is typically far below luminance's.
    # The decisive, distribution-free test: correlation with the gradient.
    if abs(float(np.percentile(gsrc, SE_LO) - np.percentile(gsrc, 50))) < 1e-6 \
            and SE_LO != 50:
        print("[1b] !! WARNING SE_STAT_FLAT: p_LO equals the median")
    _r_g = float(np.corrcoef(gsrc.ravel()[::97], grad.ravel()[::97])[0, 1])
    if _r_g < 0.9:
        print("[1b] !! WARNING SE_WRONG_STAT: gate stat correlates only %.3f "
              "with the gradient -- it is probably NOT |grad|" % _r_g)
    print("[1b] SE share: w>0 %.2f%%  w>0.5 %.2f%%  mean w %.4f  (linear blend "
          "for reference: mean %.4f)"
          % (100 * on.mean(), 100 * hi_w.mean(), float(se_w.mean()),
             float(edge_keep_lin.mean())))
    # prove the floor really took effect: how many pixels are baseline-driven
    _flo = se_w <= edge_keep_lin * SE_FLOOR + 1e-9
    _raised = se_w > edge_keep_lin + 1e-6
    print("[1b] SE floor=%.2f: %d px promoted above baseline (%.2f%%), "
          "%.2f%% of pixels left at baseline value"
          % (SE_FLOOR, int(_raised.sum()), 100 * float(_raised.mean()),
             100 * float(_flo.mean())))
    if warn:
        for w_ in warn:
            print("[1b] !! WARNING %s" % w_)
    else:
        print("[1b] SE gates healthy (not inert, not degenerate)")
    wk = se_w[..., None]
    edge_sm = sm * (1 - wk) + orig * wk
    print("[1b] structural edge sharpening applied (%.1fs)" % (time.time() - t0))

# ---- 1c. independent detail overlay (VEC_OVL, default OFF) ------------------
# GOAL: restore small real detail (a mouth line, an eyelash, a thin crease)
# that the quantiser erases -- WITHOUT touching the flat-region connected-block
# structure. The four previously rejected attempts all failed the same way:
#   VEC_EDGE 0.16->0.50   global blend weight   -> flat-block flecks on starry
#   VEC_SE  pixel-domain gate                   -> flat areas untouched by
#                          0.0000% yet bnd still +50% (transfer inside quant)
#   VEC_AQ  symmetric budget (flat 14/detail 42)-> MAE +0.4..12.5%
#   VEC_AQ2 asymmetric +16 colours to detail    -> MAE +7.4%, bnd1 +21..52%
# VEC_AQ2 is the decisive counter-example: flat-region pixels were provably
# never edited, yet the flat metrics still regressed. The chain is
#   detail pixels change ownership -> connected blocks change -> seams move
#   -> flat-region boundaries change.
# So this section does something structurally different: the detail pixels are
# REMOVED from the image that goes into KMeans (replaced by an inpainted
# neighbourhood value), and are drawn as a separate group ON TOP of the SVG.
# The base quantiser therefore never sees the detail and cannot reshuffle flat
# blocks because of it.
#
# WHAT "REAL DETAIL" MEANS HERE -- and the three rejected detectors, measured:
#   (a) small area + inscribed radius + local high-frequency + colour distance
#       -> REJECTED. starry noise blocks have rmax p50=1.41 and area p90=20, the
#          same range as the mouth block (r=2.24, area=65). Not separable.
#   (b) prom = block amplitude / surrounding-ring amplitude
#       -> REJECTED. mouth prom p50=3.6 vs starry 1.8, but the DENSITY of the
#          distributions overlaps: starry p90=4.8. The mouth block itself
#          reaches 166, i.e. the signal lives entirely in the tail.
#   (c) prom_col = |mean signed delta| / mean |signed delta| (direction
#       consistency of the colour offset against the 21x21 local background)
#       -> REJECTED, and this one failed on the POSITIVE example: the mouth
#          block scores 1.22, BELOW mouth's own median 1.74, while starry
#          reaches p90=1.80. A detector that scores the target below its own
#          background distribution cannot be thresholded.
#   (d) dens = local COMPANION DENSITY of candidate pixels in a 31x31 window
#       -> ACCEPTED. This is the one dimension where the two are actually
#          different, and it is a statement about SPATIAL DISTRIBUTION rather
#          than about strength:
#            * noise is DENSE: wherever noise exists, nearly every 3x3 has
#              candidates, so a candidate has dozens of companions nearby.
#            * real detail is SPARSE: a lip line sits in a large flat skin
#              area, so its 31x31 neighbourhood contains almost no other
#              candidates.
#          Measured companion-density medians: mouth 0.139, wuther 0.167,
#          user 0.166, starry 0.325. At cut 0.10 the mouth block (dens=0.068)
#          is KEPT while starry retains only 0.27% of the canvas.
#          This is also exactly the right semantics for this feature: what we
#          want back is the isolated line inside a flat region. Dense texture
#          (foliage, hair, the starry noise field) never suffers from "a line
#          went missing inside a flat area", so it must be left alone entirely.
OVL = os.environ.get("VEC_OVL", "0") != "0"
OVL_PCT = float(os.environ.get("VEC_OVL_PCT", "85"))     # candidate HF percentile
OVL_DE = float(os.environ.get("VEC_OVL_DE", "5"))        # colour distance to bg
OVL_DENS = float(os.environ.get("VEC_OVL_DENS", "0.10"))  # companion-density cut
OVL_WIN = int(os.environ.get("VEC_OVL_WIN", "31"))       # density window px
OVL_RMAX = float(os.environ.get("VEC_OVL_RMAX", "2.6"))  # inscribed-radius cap
OVL_AMAX = int(os.environ.get("VEC_OVL_AMAX", "3000"))   # block area cap
OVL_AMIN = int(os.environ.get("VEC_OVL_AMIN", "5"))      # block area floor
OVL_GROW = int(os.environ.get("VEC_OVL_GROW", "0"))      # dilate the detail mask
OVL_OPA = float(os.environ.get("VEC_OVL_OPA", "1.0"))    # overlay opacity
OVL_EPS = float(os.environ.get("VEC_OVL_EPS", "0.22"))   # RDP eps for overlay
#   VEC_OVL_KCLS   tonal groups per detail block (default 3). A lip line is a
#                  near-black rim around a near-white highlight; filling the
#                  whole block with its mean colour paints grey-beige mush
#                  (measured on the mouth: mean RGB(209,181,173) over
#                  10 dark + 19 mid + 36 bright source pixels). Splitting the
#                  block by luminance keeps the rim dark and the centre bright.
#   VEC_OVL_KMIN   minimum luminance spread inside a block for the split to
#                  apply (default 24). Below this the block really is one tone.

OVL_KCLS = int(os.environ.get("VEC_OVL_KCLS", "3"))
OVL_KMIN = float(os.environ.get("VEC_OVL_KMIN", "24"))

ovl_mask = None
if OVL:
    _LO = orig.mean(axis=2)
    # local high-frequency energy: distance from a 5x5 median (the low band)
    _hf = np.abs(_LO - ndmedian(_LO, size=5, mode="nearest"))
    _hfs = uniform_filter(_hf, size=3, mode="nearest")
    # local background colour: 21x21 per-channel median. It MUST be much larger
    # than the detail itself -- a 7x7 median already returns the dark mouth
    # colour INSIDE the mouth, which made the colour-distance term vanish at the
    # centre and kept only the left rim (measured; that produced exactly the
    # "isolated dark blob" failure the task forbids).
    _bg = np.stack([ndmedian(orig[..., c], size=21, mode="nearest")
                    for c in range(3)], axis=-1)
    _dc = np.abs(orig - _bg).mean(axis=2)
    _t_hf = float(np.percentile(_hfs, OVL_PCT))
    _cand = (_hfs > _t_hf) & (_dc > OVL_DE)
    # (d) companion density -- the accepted separator
    _dens = uniform_filter(_cand.astype(np.float32), size=OVL_WIN)
    _lb, _nb = ndlabel(_cand)
    if _nb:
        _sz = np.bincount(_lb.ravel())
        _dt = distance_transform_edt(_cand)
        _rmx = np.zeros(_nb + 1, dtype=np.float32)
        np.maximum.at(_rmx, _lb[_cand], _dt[_cand])
        _ok = np.zeros(_nb + 1, dtype=bool)
        from scipy.ndimage import find_objects as _fobj
        _objs = _fobj(_lb)
        _kept_blocks = 0
        for _k in range(1, _nb + 1):
            _a = int(_sz[_k])
            if _a < OVL_AMIN or _a > OVL_AMAX:
                continue
            if float(_rmx[_k]) > OVL_RMAX:
                continue
            _sl = _objs[_k - 1]
            _d = float(_dens[_sl][_lb[_sl] == _k].mean())
            if _d >= OVL_DENS:
                continue
            _ok[_k] = True
            _kept_blocks += 1
        ovl_mask = np.isin(_lb, np.nonzero(_ok)[0])
    else:
        ovl_mask = np.zeros(_cand.shape, dtype=bool)
    if OVL_GROW > 0:
        from scipy.ndimage import binary_dilation as _bd
        ovl_mask = _bd(ovl_mask, iterations=OVL_GROW)
    _npx = int(ovl_mask.sum())
    _tot = ovl_mask.size
    _candpx = int(_cand.sum())
    print("[1c] OVL detail detect: cand %d px (%.2f%%) in %d blocks, "
          "hf_thr p%g=%.2f de>%g dens<%g(win%d) -> detail %d px (%.3f%%), "
          "%d blocks"
          % (_candpx, 100.0 * _candpx / _tot, _nb, OVL_PCT, _t_hf, OVL_DE,
             OVL_DENS, OVL_WIN, _npx, 100.0 * _npx / _tot, _kept_blocks))
    # SELF-PROOF: an inert branch must never pass silently (project lesson).
    if _npx == 0:
        print("[1c] !! WARNING OVL_INERT: no detail pixels detected; "
              "overlay will be empty and the output equals the baseline")
    elif _npx > 0.10 * _tot:
        print("[1c] !! WARNING OVL_TOO_WIDE: detail covers %.2f%% of the "
              "canvas; the base quantiser is losing too much area"
              % (100.0 * _npx / _tot))
    else:
        print("[1c] OVL detector healthy (detail share %.3f%% in range)"
              % (100.0 * _npx / _tot))

# ---- 1.5 choose colour space adaptively (unless the caller forces one) ----
def _quant_error(space_fn, back_fn, samples, n, seed=42):
    """Quantize a sample set under one colour space and return mean abs error
    in sRGB 0..1 units."""
    km = MiniBatchKMeans(n_clusters=min(n, len(samples)), random_state=seed,
                         batch_size=4096, n_init=3, max_iter=80)
    idx = km.fit_predict(space_fn(samples))
    cen = np.clip(back_fn(km.cluster_centers_.reshape(1, -1, 3))[0], 0, 1)
    return float(np.abs(samples - cen[idx]).mean())


if OKLAB_RAW in ("1", "true", "yes", "on"):
    OKLAB = True
    print("[1.5] colour space: Oklab (forced)")
elif OKLAB_RAW in ("0", "false", "no", "off"):
    OKLAB = False
    print("[1.5] colour space: CIE LAB (forced)")
else:
    # Sample a small, stride-based subset. Deterministic, cheap, representative.
    flat_sm = (edge_sm / 255.0).reshape(-1, 3)
    step = max(1, len(flat_sm) // 20000)
    smp = np.ascontiguousarray(flat_sm[::step][:40000])
    e_lab = _quant_error(lambda x: skcolor.rgb2lab(x), skcolor.lab2rgb, smp, N_COLORS)
    e_ok = _quant_error(srgb_to_oklab, oklab_to_srgb, smp, N_COLORS)
    OKLAB = e_ok < e_lab
    print("[1.5] colour space auto: LAB=%.4f Oklab=%.4f -> %s"
          % (e_lab, e_ok, "Oklab" if OKLAB else "CIE LAB"))


# ---- 2. quantization ----
# NOTE: light-end sample weighting was tried here and REMOVED. Giving near-white
# pixels extra weight (VEC_LIGHTW, tested 0.5/1.0/2.0) changed cluster POSITIONS
# but not their COUNT: the 225..245 band kept 73.9% single-colour coverage and
# band MAE improved <1% (noise). With few centres, KMeans finds it cheap to
# cover the low-variance near-white cloud with one centre, so weighting cannot
# force more centres there. Fixing that needs a structural quota, not a weight.
#
# VEC_OVL (section 1c): the detail pixels are REPLACED here, before KMeans, by
# the value of their nearest non-detail pixel. This is the whole architectural
# point -- the base quantiser must never see the detail at all, so the flat
# regions it segments are exactly what they would have been without any detail
# present. VEC_AQ2 proved the opposite direction is hopeless: leaving the base
# untouched and only adding detail colours still moved flat-region boundaries
# (bnd1 +21..52%) because the detail pixels' ownership in the base changed.
base_img = edge_sm
if OVL and ovl_mask is not None and ovl_mask.any():
    _ind = distance_transform_edt(ovl_mask, return_distances=False,
                                  return_indices=True)
    # nearest NON-detail pixel value (the EDT indices point at the nearest
    # zero of the mask, i.e. the nearest kept pixel)
    _fill = edge_sm[tuple(_ind)]
    base_img = np.where(ovl_mask[..., None], _fill, edge_sm)
    _chg = float(np.abs(base_img - edge_sm)[ovl_mask].mean())
    print("[2] OVL: %d detail px (%d/%d ch) replaced by nearest non-detail "
          "value before KMeans; mean change %.1f per channel"
          % (int(ovl_mask.sum()),
             int(base_img[ovl_mask].shape[0]),
             int(base_img[ovl_mask].shape[0]), _chg))
else:
    print("[2] OVL off: quantiser input is edge_sm unchanged")

lab = to_space(base_img / 255.0).reshape(-1, 3)
km = MiniBatchKMeans(n_clusters=N_COLORS, random_state=SEED,
                     batch_size=16384, n_init=4, max_iter=150)
idx = km.fit_predict(lab).reshape(WORK_H, WORK_W).astype(np.int32)
cen = (from_space(km.cluster_centers_.reshape(1, -1, 3))[0] * 255).clip(0, 255)
cnt = np.bincount(idx.ravel(), minlength=N_COLORS)
print("[2] quantized %d colours ok (%.1fs)" % (N_COLORS, time.time() - t0))
# ---- 2.06 asymmetric budget: add colours to detail only (VEC_AQ2) ------
# VEC_AQ（对称重分配）实测失败：平坦区砍到 14 色 -> 渐变断层 -> MAE 恶化
# 12.5%，而 grad 只涨 2~4.6%（mouth 甚至 -6.4%）。唯一亮点是 starry 的
# bnd 从 18.93 腰斩到 9.22。
#
# 本版改为「只加不减」：平坦区像素一个不动（保住不受伤的收益），
# 只用 EXTRA 个新中心对细节区做二次细化。这样平坦区的 bnd 理论上不变，
# 细节区获得额外色阶。
AQ2 = int(os.environ.get("VEC_AQ2", "0"))
if AQ2 > 0:
    _lf2 = np.abs(edge_sm.mean(axis=2) - ndmedian(edge_sm.mean(axis=2), size=5))
    _lfm2 = uniform_filter(_lf2, size=9)
    _cut2 = float(np.percentile(_lfm2, float(os.environ.get("VEC_AQ2_PCT", "75"))))
    det2 = (_lfm2 > _cut2).ravel()
    n_det2 = int(det2.sum())
    print("[2.06] AQ2: detail px %d (%.2f%%), extra %d colours requested"
          % (n_det2, 100.0 * n_det2 / det2.size, AQ2))
    if n_det2 < 800:
        print("[2.06] AQ2 INERT: too few detail pixels")
    else:
        _lab3 = to_space(edge_sm / 255.0).reshape(-1, 3)
        _kd2 = MiniBatchKMeans(n_clusters=AQ2, random_state=SEED,
                               batch_size=4096, n_init=2).fit(_lab3[det2])
        _new_lab = _kd2.cluster_centers_
        _new_cen = (from_space(_new_lab.reshape(1, -1, 3))[0] * 255).clip(0, 255)
        # 只对细节区像素做重分配（平坦区 idx 保持原样）
        _d2 = ((_lab3[det2][:, None, :] - _new_lab[None, :, :]) ** 2).sum(axis=2)
        _best2 = _d2.argmin(axis=1)
        _flat_idx = idx.ravel().copy()
        _flat_idx[det2] = N_COLORS + _best2
        idx = _flat_idx.reshape(WORK_H, WORK_W).astype(np.int32)
        cen = np.vstack([cen.astype(np.float64), _new_cen.astype(np.float64)])
        cnt = np.bincount(idx.ravel(), minlength=len(cen))
        _hit = int((cnt[N_COLORS:] > 0).sum())
        print("[2.06] AQ2 ok: pool %d centres (%d existing + %d new), "
              "%d of the new centres actually used; flat pixels untouched"
              % (len(cen), N_COLORS, AQ2, _hit))

# ---- 2.05 adaptive palette budget (VEC_AQ, default OFF) ----------------
# 依据：已验证像素域门控（VEC_SE）无效——平坦区一像素未碰，bnd 仍涨 50%，
# 说明传导在量化域的全局重排。要真正解耦「细节锐度」与「平坦区碎块」，
# 必须在量化阶段按区域复杂度分配预算。
# 不做空间分块（tile 边界会产生接缝）；改为两档独立聚类 + 统一中心池重赋值。
AQ = os.environ.get("VEC_AQ", "0") != "0"
if AQ:
    # 局部高频能量：|L - 中值滤波(L)| 的邻域均值，用分位数切两档
    _lf = np.abs(edge_sm.mean(axis=2) - ndmedian(edge_sm.mean(axis=2), size=5))
    _lfm = uniform_filter(_lf, size=9)
    _aq_pct = float(os.environ.get("VEC_AQ_PCT", "75"))
    _cut = float(np.percentile(_lfm, _aq_pct))
    detail = _lfm > _cut
    n_detail = int(detail.sum())
    print("[2.05] AQ: detail px %d (%.2f%%), cut=%.2f (p%g)"
          % (n_detail, 100.0 * n_detail / detail.size, _cut, _aq_pct))
    if n_detail < 500 or n_detail > 0.85 * detail.size:
        print("[2.05] AQ INERT: detail share out of range, falling back")
    else:
        # 预算：细节档拿 detail 份额的 3 倍权重（它像素少但需要细）
        _share = n_detail / float(detail.size)
        _k_detail = int(round(N_COLORS * min(0.75, max(0.35, _share * 3.0))))
        _k_flat = max(8, N_COLORS - _k_detail)
        print("[2.05] AQ budget: detail %d colours, flat %d colours"
              % (_k_detail, _k_flat))
        _lab2 = to_space(edge_sm / 255.0).reshape(-1, 3)
        _flatmask = ~detail.ravel()
        _kms = []
        if _k_detail >= 2:
            _kd = MiniBatchKMeans(n_clusters=_k_detail, random_state=SEED,
                                  batch_size=4096, n_init=2).fit(_lab2[detail.ravel()])
            _kms.append(_kd.cluster_centers_)
        if _k_flat >= 2 and _flatmask.any():
            _kf = MiniBatchKMeans(n_clusters=_k_flat, random_state=SEED,
                                  batch_size=4096, n_init=2).fit(_lab2[_flatmask])
            _kms.append(_kf.cluster_centers_)
        if _kms:
            _pool = np.vstack(_kms)                 # 共享中心池
            # 全局重赋值：每个像素取池中最近中心 -> 消除边界接缝
            _d = ((_lab2[:, None, :] - _pool[None, :, :]) ** 2).sum(axis=2) \
                if _pool.shape[0] <= 96 else None
            if _d is None:
                # 中心多时用分块距离，避免 1M x 96 x 3 的内存峰值
                _best = np.empty(_lab2.shape[0], dtype=np.int32)
                _step = 200000
                for _s0 in range(0, _lab2.shape[0], _step):
                    _s1 = min(_s0 + _step, _lab2.shape[0])
                    _dd = ((_lab2[_s0:_s1, None, :] - _pool[None, :, :]) ** 2).sum(axis=2)
                    _best[_s0:_s1] = _dd.argmin(axis=1)
            else:
                _best = _d.argmin(axis=1)
            idx = _best.reshape(WORK_H, WORK_W).astype(np.int32)
            cen = (from_space(_pool.reshape(1, -1, 3))[0] * 255).clip(0, 255)
            cnt = np.bincount(idx.ravel(), minlength=len(cen))
            _used = int((cnt > 0).sum())
            print("[2.05] AQ ok: pool %d centres, %d actually used, "
                  "pixel assignment is globally re-computed (no tile seams)"
                  % (len(cen), _used))


# ---- 2.5 smarter cluster merging (v1.0.7) -------------------------------
# ROOT CAUSE of the "worm-eaten" tessellation, measured: a 180x120 patch of
# plain blue background is quantised into 25 distinct labels forming 1049
# connected components, with centroid colours differing by only 2-4 RGB units.
# Every one of those internal seams gets traced, producing the blocky mosaic.
# The tracer and the merge gate were never the problem.
#
# Fix: merge two ADJACENT clusters only when the merge is invisible. Three
# guards must all hold -- colour distance, per-channel 8-bit step, and (the
# decisive one) the mean gradient of the SOURCE along their contact line.
# A boundary a viewer can see has a real gradient; a quantiser artefact does
# not. Measured on a_starry this cuts flat-region seams by 78% while MAE also
# improves (18.09 -> 15.42), so it is not a quality trade.
# MERGE_STEP default changed 10 -> 3 (v1.0.9). The per-channel step guard is
# what separates "redundant duplicate" (e.g. #FBFBFD vs #FAFAFC, 1 unit apart --
# merging these is correct) from "real tonal step" (#FBFBFD vs #EAE4E2, 17 units
# -- merging these bleaches real shading). At 10 the near-white region collapsed:
# on a white-on-white illustration 75.6% of the source 225..245 band was painted
# pure white and the light gradation disappeared.
#
# Validated on 10 images, no regression (MAE): white-on-white -26.6%,
# 91130473 -8.2%, 149160955 -10.1%, 96196782 -0.07%, user_char -3.5%,
# dark-bg -4.1%, light-bg -10.1%, a_starry +0.08%, c_wuthering 0.00%.
# The two images most sensitive to over-merging (noise field, flat poster) are
# unchanged, so the worm-eaten protection is intact.
#
# Cost, stated honestly: boundary density rises (edges get harder) because more
# near-identical tones now stay separate. Measured on every test image the
# result still sits BELOW the source's own boundary density, so this is a
# style-level shift (crisper edges), not invented tessellation.
MERGE_DE = float(os.environ.get("VEC_MERGE_DE", "6"))
MERGE_STEP = float(os.environ.get("VEC_MERGE_STEP", "3"))
MERGE_GRAD = float(os.environ.get("VEC_MERGE_GRAD", "5"))
# --- light-region merge guard (default OFF: VEC_LIGHT_STEP=0) --------------
# MEASURED root cause of "light compression" on a white-on-white illustration:
# AFTER merging, the single near-white group holds 74.2% of the source
# 225..245 luminance band, and its representative colour is the most populous
# MEMBER (#FBFBFD, 600027 px), not a mean. So #F2EAEE / #EAE4E2 / #F4EDE9 --
# real, faintly tinted shading -- are all painted 251,251,253.
#
# The merge gate is colour-distance based, so it cannot tell a redundant pair
# (#FBFBFD vs #FAFAFC, 1 unit apart) from a real tonal step (#FBFBFD vs
# #EAE4E2, 17 units apart). The global MERGE_STEP=10 must stay loose to keep
# the worm-eaten tessellation suppressed, but inside the near-white region a
# 10-unit step is far too coarse to survive.
#
# Attempted fix: keep DE/STEP/GRAD as shipped, but apply a STRICTER per-channel
# step to pairs where BOTH clusters are light (CIE LAB L > VEC_LIGHT_L).
# MEASURED OUTCOME -- this switch isolates nothing, so it is left at 0.
# Counting every adjacent pair on the white-on-white test image:
#   pairs passing dE<6 and grad<5 with a per-channel step in (3,10]: 76
#   of those, light-light pairs (both L>85):                        76 (100%)
#   non-light pairs:                                                 0 (  0%)
# Every pair STEP=3 refuses to merge is already a light-light pair, so
# "restrict the strict step to light regions" is equivalent to applying it
# globally: VEC_LIGHT_STEP=3 and VEC_MERGE_STEP=3 emit byte-identical SVGs
# (title aside). The coloured edge lines that made STEP=3 unacceptable are not
# collateral damage FROM the light region -- they originate inside it, because
# hair and costume outlines carry a thin antialiasing rim of near-white pixels.
# Kept as a documented dead end so the measurement is not repeated.
# Default 0 = disabled, and with it disabled this block never runs, so the
# output stays byte-identical.
LIGHT_STEP = float(os.environ.get("VEC_LIGHT_STEP", "0"))
LIGHT_L = float(os.environ.get("VEC_LIGHT_L", "85"))
# --- light-layer restoration (default OFF: VEC_LIGHT_BLK=0) ---------------
# MEASURED, and this is why the STEP-based guard above was NOT enough:
# tightening STEP only inside the light region produced a file that is
# byte-identical (title aside) to a GLOBAL MERGE_STEP=3. In other words every
# pair that STEP=3 refuses to merge is already a light-light pair, so
# "restrict it to light regions" isolates nothing. The coloured edge lines that
# made STEP=3 unacceptable therefore originate *inside* the light region
# itself: hair and costume outlines carry a thin antialiasing rim of near-white
# pixels, and keeping those rim clusters apart draws a line of them.
#
# So try splitting by SHAPE instead: a real tonal step is a LARGE connected
# block, a rim/noise artefact is a sliver. This restores only the original
# clusters' connected components whose area is >= VEC_LIGHT_BLK, leaving
# everything smaller in the merged state. Boundaries are clipped to the
# original cluster mask by construction (the selector runs on the pre-merge
# label map), so a restored layer provably cannot spread.
#
# MEASURED OUTCOME -- better than the STEP guard on colour, but NOT a win:
#   bandMAE 11.796 -> 4.522   ->white% 75.6 -> 18.6   topCol% 73.9 -> 13.6
#   bnd%     5.45 -> 13.01    (STEP=3 is 8.98, so this is 45% WORSE than the
#                              candidate it was meant to beat)
#   speckle (band px differing from their 3x3 neighbourhood) 19.3% -> 33.9%
#   volume 8145 -> 11483 KB (+41%), runtime +18%
# Mechanism: the restored regions cover 58.1% of the canvas and 67.9% of their
# boundary pixels land in SOURCE-FLAT areas (90.9% of restored pixels are in
# flat areas). bnd% measures exactly that quantity, so every restored cluster
# adds a seam right where bnd is counted. The restored layers are also
# block-shaped colour patches, not the smooth shading being sought.
#
# ROOT CONFLICT, which is why no K can fix this: 75.6% of the source's 225..245
# band has gradient < 3, i.e. the light band IS a flat region under this metric.
# bandMAE rewards tonal variety there; bnd% punishes variety in flat regions.
# Sweeping K only walks between the two ends (K=64: bandMAE 3.96/bnd 13.91 ...
# K=1024: 5.30/11.79 ... off: 11.80/5.45) with no plateau in between.
#
# Default 0 = disabled, output byte-identical. Safe to leave in place: it is
# provably dormant (identical hashes on all three test images with and without
# it), and it does not trigger at all on a_starry or user_char.
LIGHT_BLK = int(os.environ.get("VEC_LIGHT_BLK", "0"))


def merge_invisible_clusters(idx_, cen_, orig_, grad_, de_thr, step_thr, grad_thr,
                             light_step=0.0, light_L=85.0):
    if de_thr <= 0:
        return idx_, cen_, 0, 0
    labels = np.unique(idx_)
    parent = {int(l): int(l) for l in labels}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    a = np.concatenate([idx_[:, :-1].ravel(), idx_[:-1, :].ravel()])
    b = np.concatenate([idx_[:, 1:].ravel(), idx_[1:, :].ravel()])
    g = np.concatenate([grad_[:, :-1].ravel(), grad_[:-1, :].ravel()])
    m = a != b
    a, b, g = a[m], b[m], g[m]
    if a.size == 0:
        return idx_, cen_, 0, 0

    ok = srgb_to_oklab(cen_.astype(np.float64) / 255.0)
    key = a.astype(np.int64) * 4096 + b.astype(np.int64)
    order = np.argsort(key)
    a, b, g, key = a[order], b[order], g[order], key[order]
    uniq_key, start = np.unique(key, return_index=True)
    end = np.append(start[1:], len(key))

    # --- light-region guard (only when explicitly enabled) ----------------
    # Precompute the real CIE LAB L of every cluster so the guard keys off
    # human-perceived lightness, not the derived (gamma-skipped) Oklab above.
    # Cost: one rgb2lab on N cluster centres, i.e. negligible next to KMeans.
    if light_step > 0:
        _lab = skcolor.rgb2lab(cen_.astype(np.float64).reshape(1, -1, 3) / 255.0)[0]
        cen_L = _lab[:, 0]
        is_light = cen_L > light_L
    else:
        is_light = None

    npair = 0
    nguard = 0
    for i, s in enumerate(start):
        e = end[i]
        la, lb = int(a[s]), int(b[s])
        if la == lb:
            continue
        d_ok = float(np.linalg.norm(ok[la] - ok[lb]) * 100.0)
        d_step = float(np.abs(cen_[la] - cen_[lb]).max())
        g_soft = float(g[s:e].mean())
        # The step gate is the weakest link inside the near-white cloud: at 10
        # units it happily welds #FBFBFD to #EAE4E2. Inside the light region
        # demand a much smaller step. Everything else is unchanged, so the
        # worm-eaten suppression (which lives in mid/dark flat areas) and the
        # coloured-edge-line behaviour are untouched.
        step_eff = step_thr
        if light_step > 0 and is_light[la] and is_light[lb]:
            step_eff = min(step_thr, light_step)
        if d_ok < de_thr and d_step <= step_eff and g_soft < grad_thr:
            union(la, lb)
            npair += 1
        elif step_eff < step_thr and d_step <= step_thr:
            # This pair would have merged under the shipped gate but is being
            # held apart by the light guard -- count it for the log.
            nguard += 1

    remap = np.arange(len(cen_), dtype=np.int32)
    for l in labels:
        remap[int(l)] = find(int(l))
    new_idx = remap[idx_]
    # Representative colour of a merged group: use the DOMINANT member (the one
    # with the most pixels), not the arithmetic mean. Mean was measured to
    # shift the flat background by +15 RGB because it averages bright and dark
    # blues together; picking the most populous member keeps the colour the
    # background actually has, which is the whole point of the merge.
    counts = np.bincount(idx_.ravel(), minlength=len(cen_)).astype(np.float64)
    new_cen = cen_.astype(np.float64).copy()
    for l in np.unique(new_idx):
        members = [int(x) for x in labels if find(int(x)) == int(l)]
        if len(members) > 1:
            dom = max(members, key=lambda m: counts[m])
            new_cen[int(l)] = cen_[dom]
    return new_idx.astype(np.int32), new_cen, npair, nguard


if MERGE_DE > 0:
    _gs = orig.mean(axis=2)
    _gx = np.zeros_like(_gs)
    _gy = np.zeros_like(_gs)
    _gx[:, 1:-1] = _gs[:, 2:] - _gs[:, :-2]
    _gy[1:-1, :] = _gs[2:, :] - _gs[:-2, :]
    _grad = np.hypot(_gx, _gy)
    _n_before = len(np.unique(idx))
    # Snapshot the PRE-merge label map and palette. Section 2.6 needs them to
    # find what the merge swallowed; taking the snapshot any later would only
    # see the merged labels, in which the absorbed clusters have already
    # disappeared (that mistake made the restore block a silent no-op).
    _quant_idx = idx.copy()
    _quant_cen = cen.astype(np.float64).copy()
    idx, cen, _npair, _nguard = merge_invisible_clusters(
        idx, cen, orig, _grad, MERGE_DE, MERGE_STEP, MERGE_GRAD,
        LIGHT_STEP, LIGHT_L)
    print("[2.5] merged %d invisible cluster pairs: %d -> %d colours (%.1fs)"
          % (_npair, _n_before, len(np.unique(idx)), time.time() - t0))
    if LIGHT_STEP > 0:
        print("[2.5] light guard: L>%g 区内 STEP 收紧到 %g，拦下 %d 对"
              " (浅色层保留 %d -> %d)"
              % (LIGHT_L, min(MERGE_STEP, LIGHT_STEP), _nguard, _n_before,
                 len(np.unique(idx))))

# ---- 2.6 restore large light components (default OFF) -------------------
# Split by SHAPE, not colour (see the VEC_LIGHT_BLK comment above for why the
# colour-based guard provably isolates nothing). For every quantised cluster
# whose CIE LAB L is above VEC_LIGHT_L that the merge demonstrably swallowed,
# restore its connected components of area >= VEC_LIGHT_BLK as their own
# layers. Because the selector runs over the ORIGINAL label map, each restored
# region is clipped exactly to the pixels that belonged to that cluster; it
# cannot bleed outside them. Components below the threshold keep the merged
# colour, so 1-px rims and quantisation crumbs stay suppressed.
#
# Representative colour is the ORIGINAL cluster centre, i.e. the tone that was
# actually present, not the dominant member of the merged group.
if MERGE_DE > 0 and LIGHT_BLK > 0:
    # `_quant_idx` / `_quant_cen` were snapshotted before the merge (see 2.5).
    # The merged `cen` holds only the dominant member's colour, so reading
    # tones from it would restore a layer painted the very colour the merge
    # already chose -- no tonal step comes back. The original centres are the
    # tones the image actually contains.
    _src_lab = skcolor.rgb2lab(_quant_cen.reshape(1, -1, 3) / 255.0)[0]
    _base_n = len(cen)              # 合并后 idx 的取值都 < _base_n
    _idx_new = idx.copy()
    _new_cens = []
    _restored_px = 0
    _restored_ci = 0
    for _ci in range(len(_quant_cen)):
        if _src_lab[_ci, 0] <= LIGHT_L:
            continue
        _m = (_quant_idx == _ci)
        if not _m.any():
            continue
        # 该簇若在合并后仍独占自己的像素，则本来就是独立层，无需补
        _uniq = np.unique(idx[_m])
        if _uniq.size == 1 and int(_uniq[0]) == _ci:
            continue
        _lb, _nl = ndlabel(_m)
        if _nl == 0:
            continue
        _sz = np.bincount(_lb.ravel())
        _keep = np.nonzero((_sz >= LIGHT_BLK) & (np.arange(_sz.size) > 0))[0]
        if _keep.size == 0:
            continue
        _sel = np.isin(_lb, _keep)
        # 新标签从 _base_n 起分配，不会撞上任何现有标签
        _idx_new[_sel] = _base_n + _restored_ci
        _new_cens.append(_quant_cen[_ci])
        _restored_ci += 1
        _restored_px += int(_sel.sum())
    if _restored_ci:
        cen = np.vstack([cen.astype(np.float64),
                         np.array(_new_cens, dtype=np.float64)])
        idx = _idx_new
        # cnt 供后面按面积排序与底色判定使用，必须同步扩展
        cnt = np.bincount(idx.ravel(), minlength=len(cen)).astype(np.int64)
        print("[2.6] light layers restored: %d clusters, %d px"
              " (block >= %d, L>%g) -> %d colours (%.1fs)"
              % (_restored_ci, _restored_px, LIGHT_BLK, LIGHT_L,
                 len(np.unique(idx)), time.time() - t0))

# ---- 3. speck cleanup ----
# Adaptive mode uses a graded threshold that depends on the ORIGINAL image's
# local gradient. In flat regions a component must be large to survive (quant
# noise is cheap to drop and dropping it kills the "worm-eaten" artifact);
# on real edges and hard detail the strict threshold is kept so text and line
# art are untouched. Sizes are compared against a per-pixel threshold map, so
# each component is judged with the threshold at its own location.
if ADAPT:
    gsrc = orig.mean(axis=2)
    ggx = np.zeros_like(gsrc)
    ggy = np.zeros_like(gsrc)
    ggx[:, 1:-1] = gsrc[:, 2:] - gsrc[:, :-2]
    ggy[1:-1, :] = gsrc[2:, :] - gsrc[:-2, :]
    gloc = np.hypot(ggx, ggy)
    # 3x3 max so a component touching an edge keeps the strict threshold
    gloc = ndmaxf(gloc, size=3)
    # GRAD_SMOOTH is a PERCENTILE of the gradient distribution, not an absolute
    # value. Measured: the median local gradient is ~7 and p75 ~20 on real
    # images, so an absolute cut of 3 would saturate t at 1 almost everywhere
    # and silently disable this whole branch (that bug shipped once already).
    gpct = float(os.environ.get("VEC_GRAD_PCT", "70"))
    gcut = float(np.percentile(gloc, gpct)) + 1e-6
    # t=0 where gradient is well below the cut (flat) -> SPECK_SMOOTH
    # t=1 at/above the cut (real detail)              -> SPECK
    t = np.clip(gloc / gcut, 0.0, 1.0)
    thr_map = SPECK + (SPECK_SMOOTH - SPECK) * (1.0 - t)
    print("[3] adapt: grad p%d=%.2f -> thr %.0f..%.0f (flat px %.1f%%)"
          % (gpct, gcut, thr_map.min(), thr_map.max(),
             100.0 * (t < 0.05).mean()))
else:
    thr_map = np.full(idx.shape, float(SPECK), dtype=np.float32)

# ---- 3/4. iterative speck cleanup + contrast gate ----------------------
# VEC_ITER > 1 repeats the detect->gate->fill cycle. Rationale: one pass
# removes the small components, but the fill itself can create NEW small
# islands at the seams, and those survive to tracing time. Repeating the cycle
# attacks the residual. Measured on a_starry: this is the lever that actually
# moves the flat-region boundary count, which global SPECK did not.
ITER = max(1, int(os.environ.get("VEC_ITER", "1")))


def find_small(idx_cur):
    """Components below SPECK, optionally with a gradient-driven threshold."""
    mask = np.zeros_like(idx_cur, dtype=bool)
    # len(cen) rather than N_COLORS: section 2.6 may append restored light
    # labels past the original palette. In the default path the two are equal,
    # so this is a no-op there.
    for ci in range(len(cen)):
        lb, n = ndlabel(idx_cur == ci)
        if n == 0:
            continue
        sizes = np.bincount(lb.ravel()).astype(np.float64)
        ids = np.arange(len(sizes))
        if ADAPT and n > 0:
            mins = np.zeros(len(sizes), dtype=np.float64)
            mins[1:] = ndmin(thr_map, lb, index=np.arange(1, n + 1))
            keep_out = (sizes < mins) & (ids > 0)
        else:
            keep_out = (sizes < SPECK) & (ids > 0)
        mask |= np.isin(lb, np.nonzero(keep_out)[0])
    return mask


def apply_gate(small_mask, idx_cur):
    """Keep only components close in colour to what would replace them."""
    if not small_mask.any() or (DE_MAX <= 0 and DE_AUTO <= 0):
        return small_mask
    _ind = distance_transform_edt(small_mask, return_distances=False,
                                  return_indices=True)
    _near = idx_cur[tuple(_ind)]
    if DE_SPACE == "srgb":
        _d = cen[idx_cur] - cen[_near]
        _dist = np.sqrt((_d ** 2).sum(axis=2))
    else:
        _a = srgb_to_oklab(cen[idx_cur] / 255.0)
        _b = srgb_to_oklab(cen[_near] / 255.0)
        _dist = np.sqrt(((_a - _b) ** 2).sum(axis=2)) * 100.0
    cand = small_mask
    if DE_FLATONLY:
        gsrc = orig.mean(axis=2)
        _gx = np.zeros_like(gsrc)
        _gy = np.zeros_like(gsrc)
        _gx[:, 1:-1] = gsrc[:, 2:] - gsrc[:, :-2]
        _gy[1:-1, :] = gsrc[2:, :] - gsrc[:-2, :]
        cand = cand & (np.hypot(_gx, _gy) < DE_FLATGRAD)
    thr = DE_MAX
    if DE_AUTO > 0:
        vals = _dist[cand]
        thr = float(np.percentile(vals, DE_AUTO)) if vals.size else 0.0
    return cand & (_dist < thr)


for it in range(ITER):
    small = find_small(idx)
    if not small.any():
        break
    small = apply_gate(small, idx)
    if not small.any():
        break
    if FILL == "nearest":
        ind = distance_transform_edt(small, return_distances=False,
                                     return_indices=True)
        idx = idx[tuple(ind)]
    else:
        filled = generic_filter(idx.astype(np.float32),
                                lambda v: np.bincount(v.astype(np.int32)).argmax(),
                                size=3)
        idx = np.where(small, filled.astype(np.int32), idx)
    print("[3] pass %d/%d: merged=%d px (fill=%s speck=%d gate=%d)"
          % (it + 1, ITER, int(small.sum()), FILL, SPECK,
             int(DE_MAX > 0 or DE_AUTO > 0)))
    # NOTE: an earlier revision had a `de_protected` restore step here. It was
    # measurably a no-op (VEC_DE_RESTORE=1 and =0 produced byte-identical SVGs
    # across all four test images) because pixels excluded by the gate were
    # never in the merge mask to begin with, so there was nothing to restore.
    # Removed rather than left as misleading dead code.
    print("[3] speck cleanup merged=%d px (fill=%s gate=%d)"
          % (int(small.sum()), FILL, int(DE_MAX > 0 or DE_AUTO > 0)))


# ---- 4. per-layer field ----
def layer_field(ci):
    mask = (idx == ci)
    if not SUBPIX:
        return gaussian_filter(mask.astype(np.float32), sigma=0.4)
    d_in = distance_transform_edt(mask)
    d_out = distance_transform_edt(~mask)
    sdf = d_in - d_out
    return np.clip(0.5 + sdf / 2.0, 0.0, 1.0)


# ---- 5. trace layers ----
layers = []
total_pts = 0

for ci in np.argsort(-cnt):
    mask = (idx == ci)
    npx = int(mask.sum())
    if npx < MIN_AREA:
        continue
    frac = npx / idx.size
    eps = RDP_EPS * (1.0 if frac < 0.02 else (1.3 if frac < 0.10 else 1.8))

    if BLOCK_TRACE:
        # v1.0.7: one contour per connected block, each in its own <path>.
        # The old code put every contour of a colour into a single <path>,
        # and with 1000+ pixel-adjacent subpaths the nonzero rule cancelled
        # whole regions (a 26k px blue block rendered as flat grey).
        polys = BT.block_polys(mask, eps=eps, min_area=MIN_AREA,
                               sdf=True, pad=2)
    else:
        fld = layer_field(ci)
        polys = []
        for c in measure.find_contours(fld, level=0.5):
            if len(c) < 4:
                continue
            if np.allclose(c[0], c[-1]):
                c = c[:-1]
            if len(c) < 3:
                continue
            pl = rdp(c, eps)
            if len(pl) < 3:
                continue
            x = pl[:, 1]
            y = pl[:, 0]
            if 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) < MIN_AREA:
                continue
            polys.append(pl[:, ::-1])      # (row,col) -> (x,y)
    if not polys:
        continue

    total_pts += sum(len(p) for p in polys)
    c = cen[ci].astype(int)
    hexc = "#%02X%02X%02X" % (c[0], c[1], c[2])

    if BLOCK_TRACE:
        # every block already has its own polygon, so emit one path each;
        # no fill-rule interaction is possible.
        rule = ""
        d = " ".join(poly_to_d_sharp(p) for p in polys)
    elif SHARP:
        if HOLES and len(polys) > 1:
            keep_list, hole_list = split_islands_holes(polys)
            segs = [poly_to_d_sharp(p) for p in keep_list]
            segs += [poly_to_d_sharp(p) for p in hole_list]
            d = " ".join(segs)
            rule = ' fill-rule="evenodd"'
        else:
            d = " ".join(poly_to_d_sharp(p) for p in polys)
            rule = ""
    else:
        d = " ".join(poly_to_d_chaikin(p) for p in polys)
        rule = ""

    # v1.0.7 fix -- split into one <path> per disconnected contour.
    #
    # Putting every contour of a colour into a single <path> is unsafe: the
    # default nonzero rule makes overlapping subpaths cancel, and the contour
    # direction that find_contours emits is not consistent, so a colour could
    # punch holes in itself. That is what rendered a large blue background
    # region as flat grey (162,162,162) instead of (4,25,189).
    #
    # This was a latent defect long before v1.0.7 (both v1.0.5 and v1.0.6 SVGs
    # carry 1400+ subpaths in a single path with no fill-rule), it simply became
    # visible once cluster merging made the regions bigger.
    # One path per contour removes the interaction entirely.
    if SPLIT_PATHS:
        layers.append(dict(hexc=hexc,
                           pieces=[(poly_to_d_sharp(p) if SHARP
                                    else poly_to_d_chaikin(p)) for p in polys],
                           rule=rule, npx=npx, ci=ci))
    else:
        layers.append(dict(hexc=hexc, d=d, rule=rule, npx=npx, ci=ci))

print("[5] %d layers %d points (%.1fs)"
      % (len(layers), total_pts, time.time() - t0))

# ---- 5b. trace the independent detail overlay (VEC_OVL) ---------------------
# The overlay is drawn LAST, on top of every quantised layer, so its shape is
# not blended into anything. Colour comes from the SOURCE image (not from the
# palette and not from edge_sm) -- that is what makes it sharper than the base:
# the base only has 56 cluster centres, the overlay has the actual source pixel.
#
# SHAPE FIDELITY -- measured, and the first implementation got this wrong.
# The first version emitted ONE polygon per detected block, filled with the
# block's MEAN colour. On the mouth that produced two grey smudges instead of a
# lip line, and the reason is quantitative: the detector merged the entire mouth
# into a single 65 px block whose source pixels are 10 dark (<120) + 19 mid +
# 36 bright (>235). The mean of that is RGB(209,181,173) -- a grey-beige, while
# the source mouth is a near-BLACK rim around a near-WHITE highlight. A lip line
# and its highlight must never be averaged into one colour.
#
# So the overlay is now split BY COLOUR inside each block: the block's source
# pixels are clustered into at most OVL_KCLS tonal groups, each group becomes its
# own polygon filled with that group's mean SOURCE colour. A dark rim and a
# bright centre therefore stay dark and bright, and the emitted shape can be a
# rim around a highlight rather than a blob of average.
ovl_paths = []
if OVL and ovl_mask is not None and ovl_mask.any():
    _lb2, _nb2 = ndlabel(ovl_mask)
    _sz2 = np.bincount(_lb2.ravel())
    from scipy.ndimage import find_objects as _fobj2
    _objs2 = _fobj2(_lb2)
    _nblk = _npts = _npiece = 0
    _colsrc = orig              # overlay colour is sampled from the SOURCE
    _ktot = 0
    for _k in range(1, _nb2 + 1):
        if _sz2[_k] < OVL_AMIN:
            continue
        _sl = _objs2[_k - 1]
        _m0 = _lb2[_sl] == _k
        _src = _colsrc[_sl]
        _lsrc = _src.mean(axis=2)
        # split the block into tonal groups so a rim and its highlight do not
        # collapse into one average colour
        if OVL_KCLS >= 2:
            _lmin = float(_lsrc[_m0].min())
            _lmax = float(_lsrc[_m0].max())
            if _lmax - _lmin >= OVL_KMIN:
                _edges = np.linspace(_lmin, _lmax, OVL_KCLS + 1)[1:-1]
                _grp = np.digitize(_lsrc, _edges)      # 0..OVL_KCLS-1
                _nsp = OVL_KCLS
            else:
                _grp = np.zeros(_lsrc.shape, dtype=np.int32)
                _nsp = 1
        else:
            _grp = np.zeros(_lsrc.shape, dtype=np.int32)
            _nsp = 1
        for _g in range(_nsp):
            _m = _m0 & (_grp == _g)
            if int(_m.sum()) < OVL_AMIN:
                continue
            _c = _src[_m].mean(axis=0)
            _polys = BT.block_polys(_m, eps=OVL_EPS, min_area=2.0,
                                    sdf=True, pad=2)
            if not _polys:
                continue
            _y0 = _sl[0].start - 2
            _x0 = _sl[1].start - 2
            _nblk += 1
            _ktot = max(_ktot, _g + 1)
            for _p in _polys:
                # block_polys returns coordinates local to the padded window
                _p = _p + np.array([_x0, _y0], dtype=np.float64)
                _npts += len(_p)
                _npiece += 1
                ovl_paths.append((_c, poly_to_d_sharp(_p) if SHARP
                                  else poly_to_d_chaikin(_p)))
    print("[5b] OVL overlay: %d blocks -> %d pieces, %d points, "
          "%d tonal groups/block, opacity=%.2f (drawn on top of all %d base "
          "layers)"
          % (_nblk, _npiece, _npts, OVL_KCLS, OVL_OPA, len(layers)))
elif OVL:
    print("[5b] OVL overlay: EMPTY (no detail blocks passed the detector)")

# ---- 6. optional residual layer ----
resid_paths = ""
if RESID > 0:
    recon = np.zeros_like(orig)
    for L in layers:
        recon[idx == L["ci"]] = cen[L["ci"]]
    err = orig - recon
    mag = np.abs(err).mean(axis=2)
    hot = mag > 8
    if hot.any():
        pieces = []
        drift = err[hot].mean(axis=0)
        f = gaussian_filter(hot.astype(np.float32), sigma=0.5)
        base_col = cen[np.argmax(cnt)]
        comp = np.clip(base_col + drift * 2.0, 0, 255).astype(int)
        for c in measure.find_contours(f, 0.5):
            if len(c) < 6:
                continue
            pl = rdp(c, RDP_EPS * 1.2)
            if len(pl) < 3:
                continue
            pieces.append(pl[:, ::-1])
        if pieces:
            op = min(RESID, 0.6)
            res = ['<g fill-opacity="%.2f">' % op]
            for pl in pieces:
                res.append('<path fill="#%02X%02X%02X" d="%s"/>'
                           % (comp[0], comp[1], comp[2], poly_to_d_sharp(pl)))
            res.append("</g>")
            resid_paths = "".join(res)
            print("[6] residual compensation: %d pieces" % len(pieces))

# ---- 7. write ----
# The layer whose colour matches the base <rect> MUST be drawn first. It is
# the background plate; drawing it last (as argsort(-cnt) can do when its
# pixel count is not the maximum after cluster merging) paints a full-canvas
# block of background colour OVER the whole subject. Measured: on a white-
# background portrait the entire character vanished under a #FEFEFE layer.
# ---- base colour selection (VEC_BASECOL, default 0 = legacy) --------
# Legacy: argmax(cnt) picks the most populous cluster. Measured failure:
# on a dark illustration (Noyu/105819262, 600x853, 33% of pixels below L80)
# it wrote <rect fill="#FCE1B1"> (luminance 218) over a scene whose median
# colour is luminance 122. Changing ONLY that rect attribute cut MAE from
# 21.271 to 15.328 (-27.9%) and the dark-region bias from +28.43 to +7.51.
# The most populous cluster is not the background: mode-fill during speck
# cleanup can inflate an unrepresentative colour into the majority.
#
# New: derive the background colour from the SOURCE image's border band
# (top/bottom/left/right 6%), which is almost always background, and take
# the median so the few subject pixels that touch the band do not pull it.
# Then pick the palette cluster NEAREST to that colour, so base_ci stays a
# real palette entry and the layer draw-order guard (pitfall 18) is intact.
BASECOL = os.environ.get("VEC_BASECOL", "1") != "0"
if BASECOL:
    _bh = max(2, int(round(WORK_H * 0.06)))
    _bw = max(2, int(round(WORK_W * 0.06)))
    _edge = np.concatenate([
        orig[:_bh].reshape(-1, 3),
        orig[-_bh:].reshape(-1, 3),
        orig[:, :_bw].reshape(-1, 3),
        orig[:, -_bw:].reshape(-1, 3),
    ])
    _med = np.median(_edge, axis=0)
    # nearest palette cluster (Euclidean in RGB; palette is small)
    _d = ((cen - _med[None, :]) ** 2).sum(axis=1)
    _nb = int(np.argmin(_d))
    _old = int(np.argmax(cnt))
    base_ci = _nb
    print("[7] basecol: edge median rgb(%.0f,%.0f,%.0f) -> cluster #%d "
          "(rgb %.0f,%.0f,%.0f, dist %.1f); legacy argmax was #%d "
          "(rgb %.0f,%.0f,%.0f)"
          % (_med[0], _med[1], _med[2], _nb,
             cen[_nb][0], cen[_nb][1], cen[_nb][2],
             float(np.sqrt(_d[_nb])),
             _old, cen[_old][0], cen[_old][1], cen[_old][2]))
else:
    base_ci = int(np.argmax(cnt))
base = cen[base_ci].astype(int)
sx, sy = W0 / WORK_W, H0 / WORK_H
# draw base-colour layers first, everything else after, keeping the
# size-descending order inside each group (stable, deterministic)
layers.sort(key=lambda L: (0 if L["ci"] == base_ci else 1, -L["npx"]))
P = [
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
    'width="%d" height="%d" shape-rendering="geometricPrecision">' % (W0, H0, W0, H0),
    '<title>%s</title>' % TITLE,
    '<desc>Layered vector reconstruction (vec.py)</desc>',
    '<rect width="%d" height="%d" fill="#%02X%02X%02X"/>'
    % (W0, H0, base[0], base[1], base[2]),
    '<g transform="scale(%.6f %.6f)" stroke="none">' % (sx, sy),
]
for L in layers:
    if SPLIT_PATHS and "pieces" in L:
        for d in L["pieces"]:
            if d:
                P.append('<path fill="%s"%s d="%s"%s/>' % (L["hexc"], L["rule"], d, _seal_sfx(L["hexc"])))
    else:
        P.append('<path fill="%s"%s d="%s"%s/>' % (L["hexc"], L["rule"], L["d"], _seal_sfx(L["hexc"])))
if resid_paths:
    P.append(resid_paths)
# ---- 6b. the independent detail overlay goes on TOP OF EVERYTHING ----------
# This is the whole architectural claim of VEC_OVL: detail is not a layer among
# the quantised ones, it is painted over them. Nothing below can move its shape,
# and it cannot move anything below (its pixels never entered the base image).
if OVL and ovl_paths:
    P.append('<g fill-opacity="%.2f">' % OVL_OPA)
    for _c, _d in ovl_paths:
        _ci = np.clip(_c, 0, 255).astype(int)
        P.append('<path fill="#%02X%02X%02X" d="%s"/>'
                 % (_ci[0], _ci[1], _ci[2], _d))
    P.append("</g>")
P.append("</g></svg>")
svg = "\n".join(P)
open(OUT, "w", encoding="utf-8").write(svg)
print("[7] -> %s  %.1f KB  (%.1fs)"
      % (OUT, len(svg) / 1024, time.time() - t0))
