"""make_overview.py -- 生成 95 张的总览联络表

按 MAE 从好到差排序，做一张大图：每行 8 张缩略，标注文件名与 MAE。
用户早上可直接一眼看全貌。
"""
import os
import json
import numpy as np
from PIL import Image, ImageDraw

def _find_root():

    """工作区根：优先环境变量，否则取仓库的上一级。



    原实现写死作者机器上的 F:\\SVG drawing。

    """

    env = os.environ.get("IMGSVG_ROOT")

    if env and os.path.isdir(env):

        return env

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))





ROOT = _find_root()
BASE = os.path.join(ROOT, "output", "final_v1.2.0", "梅原生")
SRC = os.path.join(ROOT, "input", "new_train", "梅原生")
OUT = os.path.join(BASE, "总览_95张_按MAE排序.png")

THUMB_W = 260
COLS = 8
PAD = 6
LABEL_H = 34


def main():
    rows = json.load(open(os.path.join(BASE, "_summary.json"),
                          encoding="utf-8"))["rows"]
    rows.sort(key=lambda r: r["mae"])

    tiles = []
    for r in rows:
        png = os.path.join(BASE, "png",
                           os.path.splitext(r["file"])[0] + "_render.png")
        if not os.path.exists(png):
            continue
        try:
            im = Image.open(png).convert("RGB")
        except Exception:
            continue
        w, h = im.size
        th = int(THUMB_W * h / w)
        tiles.append((im.resize((THUMB_W, th), Image.LANCZOS), r))

    if not tiles:
        print("无缩略图"); return

    # 统一行高：取所有缩略图高度中位，超出的裁掉顶部对齐
    ths = [t[0].height for t in tiles]
    row_h = int(np.median(ths))

    nrow = (len(tiles) + COLS - 1) // COLS
    W = COLS * (THUMB_W + PAD) + PAD
    H = nrow * (row_h + LABEL_H + PAD) + PAD + 60

    cv = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(cv)

    maes = np.array([r["mae"] for _, r in tiles])
    head = ("v1.2.0 梅原生 95 张（按 MAE 升序）  "
            f"MAE 中位 {np.median(maes):.3f}  "
            f"最小 {maes.min():.3f}  最大 {maes.max():.3f}")
    d.text((10, 8), head, fill=(0, 0, 0))
    d.text((10, 26),
           "每格：渲染结果 + 文件名前缀 + MAE/p99；绿=达标(MAE<=9.454) 红=未达",
           fill=(90, 90, 90))

    for i, (im, r) in enumerate(tiles):
        cx = PAD + (i % COLS) * (THUMB_W + PAD)
        cy = 60 + PAD + (i // COLS) * (row_h + LABEL_H + PAD)
        if im.height > row_h:
            im = im.crop((0, 0, im.width, row_h))
        cv.paste(im, (cx, cy))
        # 达标标记
        ok = r["mae"] <= 9.454
        col = (0, 130, 0) if ok else (190, 0, 0)
        d.rectangle([cx, cy, cx + THUMB_W - 1, cy + row_h - 1], outline=col)
        stem = os.path.splitext(r["file"])[0][:20]
        d.text((cx + 2, cy + row_h + 2), stem, fill=(30, 30, 30))
        d.text((cx + 2, cy + row_h + 16),
               f"MAE {r['mae']:.2f}  p99 {r['p99']:.0f}  k{r['k']}",
               fill=col)

    cv.save(OUT)
    print(f"总览图 {cv.size[0]}x{cv.size[1]} -> {OUT}")
    print(f"  {len(tiles)} 张，达标 {int((maes <= 9.454).sum())} 张 "
          f"({100 * (maes <= 9.454).mean():.0f}%)")


if __name__ == "__main__":
    main()
