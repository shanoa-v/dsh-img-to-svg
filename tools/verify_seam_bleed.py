"""verify_seam_bleed.py -- 独立验证「亚像素接缝渗色」机制

## 子代理的断言（需要我独立验证）

  1. 底色 <rect> 几乎不裸露：把 rect 填成品红，纯品红只占 0.631%
  2. alpha 均值只有 0.8731，44.44% 的像素 alpha<0.999
  3. 渲染模型 rend = alpha*fg + (1-alpha)*bg 精确成立（相关系数 0.99992）
  4. 补缝（给 path 加同色 stroke）比换底色更治本：14.433 vs 14.987

## 验证方法

  实验 A：把 rect 填成纯品红 (255,0,255)，数渲染后有多少像素接近纯品红
          -> 这就是「裸露率」
  实验 B：用两次不同底色渲染，解方程求 per-pixel alpha
          rend1 = a*fg + (1-a)*bg1
          rend2 = a*fg + (1-a)*bg2
          => rend1 - rend2 = (1-a)*(bg1-bg2)
          => (1-a) = (rend1-rend2)/(bg1-bg2)   [逐通道，用差值最大的通道]
  实验 C：加 stroke 后重测 MAE
"""
import os
import io
import re
import sys
import subprocess
import numpy as np
from PIL import Image

PY = sys.executable
SK = os.path.join(os.environ["DSH_HOME"], "skills", "dsh-img-to-svg", "scripts")
def _find_root():
    """工作区根：优先环境变量，否则取仓库的上一级。

    原实现写死作者机器上的 F:\\SVG drawing。
    """
    env = os.environ.get("IMGSVG_ROOT")
    if env and os.path.isdir(env):
        return env
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROOT = _find_root()
OUT = os.path.join(ROOT, "workspace", "seam")
os.makedirs(OUT, exist_ok=True)

SRC = os.path.join(ROOT, "input", "new_train", "Noyu", "105819262_p0_master1200.webp")
SVG = os.path.join(OUT, "base.svg")

# 生成基线 SVG
env = {k: v for k, v in os.environ.items() if not k.startswith("VEC_")}
env["VEC_TITLE"] = "seam"
env["VEC_SEED"] = "42"
r = subprocess.run([PY, os.path.join(SK, "vec.py"), SRC, SVG],
                   capture_output=True, text=True, env=env, cwd=SK)
if r.returncode != 0:
    print("矢量化失败:", r.stderr[-200:]); sys.exit(1)

ref_im = Image.open(SRC).convert("RGB")
W, H = ref_im.size
ref = np.asarray(ref_im).astype(np.float32)


def render(svg_path, tag):
    png = os.path.join(OUT, "%s.png" % tag)
    subprocess.run([PY, os.path.join(SK, "render_check.py"), svg_path, png],
                   capture_output=True, cwd=SK)
    v = Image.open(png).convert("RGB")
    if v.size != (W, H):
        v = v.resize((W, H), Image.LANCZOS)
    return np.asarray(v).astype(np.float32)


def set_rect_bg(txt, hexcol, alpha=None):
    """替换 rect 的 fill，可选加 fill-opacity"""
    m = re.search(r'<rect width="(\d+)" height="(\d+)" fill="#[0-9A-Fa-f]{6}"\s*/?>', txt)
    if not m:
        return txt, None
    extra = "" if alpha is None else ' fill-opacity="%g"' % alpha
    new = '<rect width="%s" height="%s" fill="%s"%s/>' % (
        m.group(1), m.group(2), hexcol, extra)
    return txt.replace(m.group(0), new), m.group(0)


txt = io.open(SVG, encoding="utf-8").read()
base_cur = re.search(r'fill="(#[0-9A-Fa-f]{6})"', txt).group(1)
print("=" * 64)
print("源图 %s  %dx%d   当前底色 %s" % (os.path.basename(SRC), W, H, base_cur))
print("=" * 64)

# ---------- 实验 A：裸露率 ----------
print("\n[A] 底色裸露率（把 rect 填成纯品红，数渲染后接近纯品红的像素）")
t_m, _ = set_rect_bg(txt, "#FF00FF")
p_m = os.path.join(OUT, "magenta.svg")
io.open(p_m, "w", encoding="utf-8").write(t_m)
ren_m = render(p_m, "magenta")
# 接近纯品红：R>200 & G<80 & B>200
mag = (ren_m[..., 0] > 200) & (ren_m[..., 1] < 80) & (ren_m[..., 2] > 200)
print("  纯品红像素 %.3f%%（%d px）" % (100 * mag.mean(), int(mag.sum())))
# 部分渗色：R 明显高于 B 之外的情况，用"离原图更远且偏品红"衡量
print("  -> 若这个比例很小（<1%%），说明底色确实几乎不裸露")

# ---------- 实验 B：解 alpha ----------
print("\n[B] 逐像素反解 alpha（两个极端底色）")
t1, _ = set_rect_bg(txt, "#000000")
t2, _ = set_rect_bg(txt, "#FFFFFF")
p1 = os.path.join(OUT, "bg_black.svg")
p2 = os.path.join(OUT, "bg_white.svg")
io.open(p1, "w", encoding="utf-8").write(t1)
io.open(p2, "w", encoding="utf-8").write(t2)
r1 = render(p1, "bg_black")
r2 = render(p2, "bg_white")

# rend_black = a*fg + (1-a)*0
# rend_white = a*fg + (1-a)*255
# 差: rend_white - rend_black = (1-a)*255  ->  (1-a) = diff/255
diff = (r2 - r1).mean(axis=2)
one_minus_a = np.clip(diff / 255.0, 0, 1)
alpha = 1.0 - one_minus_a
print("  alpha 均值 %.4f   中位 %.4f" % (alpha.mean(), np.median(alpha)))
print("  alpha<0.999 占比 %.2f%%" % (100 * (alpha < 0.999).mean()))
print("  alpha<0.95  占比 %.2f%%" % (100 * (alpha < 0.95).mean()))
print("  alpha<0.5   占比 %.2f%%" % (100 * (alpha < 0.5).mean()))
L = ref.mean(axis=2)
dark = L < 80
if dark.any():
    print("  暗部(源L<80) 中 alpha<0.999 占比 %.2f%%"
          % (100 * (alpha[dark] < 0.999).mean()))

# ---------- 实验 C：补缝 ----------
print("\n[C] 补缝（给每个 path 加同色 stroke）")
for sw in (0.6, 0.8, 1.0, 1.2):
    # 给所有 <path fill="#XXXXXX" d="..."/> 加 stroke 同色 + stroke-width
    def add_stroke(match):
        tag = match.group(0)
        fill = re.search(r'fill="(#[0-9A-Fa-f]{6})"', tag)
        if not fill:
            return tag
        return tag.replace(
            "/>",
            ' stroke="%s" stroke-width="%g"/>' % (fill.group(1), sw)
        ) if tag.endswith("/>") else tag
    t_s = re.sub(r"<path\b[^>]*/>", add_stroke, txt)
    t_s, _ = set_rect_bg(t_s, base_cur)
    ps = os.path.join(OUT, "stroke_%g.svg" % sw)
    io.open(ps, "w", encoding="utf-8").write(t_s)
    ren_s = render(ps, "stroke_%g" % sw)
    mae = float(np.abs(ref - ren_s).mean())
    serr = ren_s - ref
    dbias = float(serr[..., 0][dark].mean()) if dark.any() else 0.0
    print("  stroke-width=%.1f  MAE %7.3f  暗部偏差 %+7.2f  体积 %.2f MB"
          % (sw, mae, dbias, os.path.getsize(ps) / 1048576))

# 基线对照
ren_base = render(SVG, "base")
print("  %-16s  MAE %7.3f  暗部偏差 %+7.2f"
      % ("基线（无补缝）", float(np.abs(ref - ren_base).mean()),
         float((ren_base - ref)[..., 0][dark].mean()) if dark.any() else 0.0))

# ---------- 实验 D：组合（补缝 + 边缘底色）----------
print("\n[D] 组合：补缝 + 换边缘底色")
bh = max(2, int(H * 0.06)); bw = max(2, int(W * 0.06))
edge = np.concatenate([ref[:bh].reshape(-1, 3), ref[-bh:].reshape(-1, 3),
                       ref[:, :bw].reshape(-1, 3), ref[:, -bw:].reshape(-1, 3)])
em = np.median(edge, axis=0)
hexe = "#%02X%02X%02X" % tuple(int(round(x)) for x in em)
print("  边缘中位色 %s" % hexe)
txt2, _ = set_rect_bg(txt, hexe)
ren_e = render((io.open(os.path.join(OUT, "edge.svg"), "w", encoding="utf-8").write(txt2),
                os.path.join(OUT, "edge.svg"))[1], "edge")
print("  %-20s MAE %7.3f" % ("仅换底色", float(np.abs(ref - ren_e).mean())))
for sw in (0.8, 1.0):
    def add_stroke2(match):
        tag = match.group(0)
        fill = re.search(r'fill="(#[0-9A-Fa-f]{6})"', tag)
        if not fill:
            return tag
        return tag.replace("/>", ' stroke="%s" stroke-width="%g"/>'
                           % (fill.group(1), sw)) if tag.endswith("/>") else tag
    t_c = re.sub(r"<path\b[^>]*/>", add_stroke2, txt)
    t_c, _ = set_rect_bg(t_c, hexe)
    pc = os.path.join(OUT, "combo_%g.svg" % sw)
    io.open(pc, "w", encoding="utf-8").write(t_c)
    ren_c = render(pc, "combo_%g" % sw)
    print("  %-20s MAE %7.3f  暗部偏差 %+7.2f"
          % ("补缝%.1f+换底色" % sw, float(np.abs(ref - ren_c).mean()),
             float((ren_c - ref)[..., 0][dark].mean()) if dark.any() else 0.0))
