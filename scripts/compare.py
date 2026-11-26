"""compare.py — 渲染图 vs 原图 定量误差 + 差异图（自动匹配宽高比）

用法: compare.py <svg> <tag> [参考图]
"""
import sys, io, os, re
import numpy as np
from PIL import Image
import cairosvg

svg = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else "x"
if len(sys.argv) <= 3:
    print("需要参考图路径：compare.py <svg> <tag> <ref>"); sys.exit(1)
SRC = sys.argv[3]

ref = Image.open(SRC).convert("RGB")
RW, RH = ref.size

# 按 SVG 自身 viewBox 比例渲染，然后统一到参考图尺寸比较
head = open(svg, encoding="utf-8").read(600)
m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', head)
vw, vh = (float(m.group(1)), float(m.group(2))) if m else (RW, RH)
svg_ar, ref_ar = vw / vh, RW / RH

if abs(svg_ar - ref_ar) > 0.01:
    print(f"  !! 宽高比不一致: svg={svg_ar:.4f} ref={ref_ar:.4f}（对比可能失真）")

png = cairosvg.svg2png(url=svg, output_width=RW, output_height=RH)
b_img = Image.open(io.BytesIO(png)).convert("RGB")

# 对比用的缩略尺寸
CW = 960
CH = round(CW * RH / RW)
a = np.asarray(ref.resize((CW, CH), Image.LANCZOS)).astype(np.float32)
b = np.asarray(b_img.resize((CW, CH), Image.LANCZOS)).astype(np.float32)

diff = np.abs(a - b)
mae = diff.mean()
rmse = np.sqrt((diff ** 2).mean())
psnr = 20 * np.log10(255.0 / max(rmse, 1e-6))
sim = (1 - mae / 255) * 100
print(f"[{tag}] MAE={mae:6.2f}  RMSE={rmse:6.2f}  PSNR={psnr:5.2f}dB  相似度={sim:5.2f}%")
print(f"       ref={os.path.basename(SRC)} {RW}x{RH}")

d = diff.mean(axis=2)
heat = np.zeros((*d.shape, 3), dtype=np.uint8)
heat[..., 0] = np.clip(d * 3, 0, 255)
heat[..., 1] = np.clip(255 - d * 3, 60, 255)
heat[..., 2] = np.clip(255 - d * 3, 60, 255)
Image.fromarray(heat).save(f"diff_{tag}.png")

comp = Image.new("RGB", (CW, CH * 2 + 10), (255, 255, 255))
comp.paste(ref.resize((CW, CH), Image.LANCZOS), (0, 0))
comp.paste(b_img.resize((CW, CH), Image.LANCZOS), (0, CH + 10))
comp.save(f"side_{tag}.png")
print(f"  -> diff_{tag}.png, side_{tag}.png")
