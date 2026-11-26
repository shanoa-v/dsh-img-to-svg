"""render_check.py — 渲染 SVG 为 PNG，尺寸跟随 SVG viewBox（不拉伸）

用法: render_check.py <svg> <out.png> [目标宽度]
"""
import sys, re, os
from PIL import Image
import cairosvg

svg_path = sys.argv[1]
out_png = sys.argv[2] if len(sys.argv) > 2 else svg_path.replace(".svg", "_render.png")
target_w = int(sys.argv[3]) if len(sys.argv) > 3 else None

head = open(svg_path, encoding="utf-8").read(600)
m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', head)
if m:
    vw, vh = float(m.group(1)), float(m.group(2))
else:
    vw, vh = 1920.0, 1080.0

w = target_w or int(vw)
h = round(w * vh / vw)          # 按原始宽高比，绝不拉伸

png = cairosvg.svg2png(url=svg_path, output_width=w, output_height=h)
open(out_png, "wb").write(png)
print(f"render {w}x{h} (viewBox {vw:g}x{vh:g}) -> {out_png}")
