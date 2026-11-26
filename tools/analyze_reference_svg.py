"""analyze_reference_svg.py -- 分析参考 SVG 的技术规格

## 目的

用户给了一个 71.9 MB 的参考 SVG（1706919294780.svg，对应 5200x11312 的 58.82MP 图），
说"这个的大小远超同文件夹内的其他 svg"，要求梅原生全部达到这种质量。

必须先从那一个文件反推出"这种质量"的可量化规格：
  - 顶点数 / path 数
  - 唯一颜色数
  - 是否有 stroke（补缝）
  - 颜色空间（hex 精度）
  - 路径密度（顶点/像素）
  - 是否用了其他技术手段

再与当前管线产出对比，找出差距在哪个环节。

## 输出

  workspace/ref_analysis/reference_spec.json
"""
import io
import os
import re
import json
import numpy as np

REF = (r"F:\Program Files\Deepseek-Harness-EAC-6.0.0 0.2.0-rc2\eac-data\dpx"
       r"\dsh-environments\eac-beta\dsh-home\attachments\v1\files\68"
       r"\6869a7c1bb929393e5a1e13f28537344def72f7460e9837e28923a8cb23c1a0b"
       r"\1706919294780.svg")
OUT = os.path.join(_find_root(), 'workspace/ref_analysis')
os.makedirs(OUT, exist_ok=True)

print("=== 参考 SVG 分析 ===")
print("文件 %s" % os.path.basename(REF))
size = os.path.getsize(REF)
print("体积 %.1f MB" % (size / 1048576))

txt = io.open(REF, encoding="utf-8", errors="replace").read()
print("字符数 %.2f 亿" % (len(txt) / 1e8))

# 头部
head = txt[:1200]
m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', head)
if m:
    vw, vh = float(m.group(1)), float(m.group(2))
    print("viewBox %.0f x %.0f = %.2f MP" % (vw, vh, vw * vh / 1e6))
else:
    vw = vh = None
print()
print("--- 头部原文 ---")
print(head[:700])
print()

# path 统计
paths = re.findall(r"<path\b[^>]*/>", txt)
print("path 数量 %d" % len(paths))

fills = re.findall(r'fill="(#[0-9A-Fa-f]{6})"', txt)
uniq_fills = set(fills)
print("fill 出现 %d 次，唯一颜色 %d 个" % (len(fills), len(uniq_fills)))

strokes = re.findall(r'stroke="(#[0-9A-Fa-f]{6})"', txt)
sws = re.findall(r'stroke-width="([\d.]+)"', txt)
print("stroke 出现 %d 次，stroke-width 出现 %d 次" % (len(strokes), len(sws)))
if sws:
    from collections import Counter
    print("  stroke-width 取值分布: %s" % dict(Counter(sws).most_common(5)))

# 顶点数：数 d 属性里的命令字母
print()
print("--- 顶点密度 ---")
d_total = 0
cmd_total = 0
sample_n = min(3000, len(paths))
# 全量统计可能慢，先抽样估
step = max(1, len(paths) // sample_n)
sampled = paths[::step]
for p in sampled:
    md = re.search(r'd="([^"]*)"', p)
    if not md:
        continue
    d = md.group(1)
    d_total += len(d)
    cmd_total += len(re.findall(r"[MLCQZmlcqz]", d))
scale = len(paths) / max(len(sampled), 1)
print("抽样 %d 个 path，平均 d 长度 %.1f 字符，平均命令数 %.1f"
      % (len(sampled), d_total / max(len(sampled), 1),
         cmd_total / max(len(sampled), 1)))
print("估算总命令数 %.2f 万" % (cmd_total * scale / 1e4))

# rect 底色
mrect = re.search(r'<rect width="[\d.]+" height="[\d.]+" fill="(#[0-9A-Fa-f]{6})"', txt)
print()
print("底色 rect: %s" % (mrect.group(1) if mrect else "无"))

# 是否有其他元素类型
print()
print("--- 元素类型统计 ---")
for tag in ("g", "rect", "path", "circle", "polygon", "image", "use"):
    n = len(re.findall(r"<%s\b" % tag, txt))
    if n:
        print("  <%s> %d" % (tag, n))

spec = {
    "file": os.path.basename(REF),
    "size_mb": round(size / 1048576, 2),
    "chars": len(txt),
    "viewbox": [vw, vh],
    "mp": round(vw * vh / 1e6, 2) if vw else None,
    "path_count": len(paths),
    "unique_fills": len(uniq_fills),
    "fill_occurrences": len(fills),
    "stroke_count": len(strokes),
    "stroke_widths": sorted(set(sws)) if sws else [],
    "rect_fill": mrect.group(1) if mrect else None,
    "est_commands": int(cmd_total * scale),
}
json.dump(spec, open(os.path.join(OUT, "reference_spec.json"), "w",
                     encoding="utf-8"), ensure_ascii=False, indent=1)
print()
print("-> %s" % os.path.join(OUT, "reference_spec.json"))
