"""diagnose_case.py -- 单图深度诊断（通用工具）

## 用途

输入任意原图 + 渲染图，输出 6 组诊断：
  1. 误差总体分布（含长尾分档）
  2. 有符号误差（偏色方向）
  3. 分区域误差（平坦/中等纹理/强纹理/亮部/暗部/高饱和/低饱和）
  4. 色阶损失（源 vs 渲染的唯一色数）
  5. 误差最集中的 5 个 96x96 区块（出 4x 放大对比）
  6. 尺寸/复杂度相关性提示

## 用法

  python diagnose_case.py <源图> <渲染图> <输出目录>
"""
import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import median_filter

def _find_root():

    """工作区根：优先环境变量，否则取仓库的上一级。



    原实现写死作者机器上的 F:\\SVG drawing。

    """

    env = os.environ.get("IMGSVG_ROOT")

    if env and os.path.isdir(env):

        return env

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))





ROOT = _find_root()
def main():
    if len(sys.argv) < 4:
        print(__doc__); return 2
    src, ren, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)

    ref_im = Image.open(src).convert("RGB")
    W, H = ref_im.size
    ref = np.asarray(ref_im).astype(np.float32)
    ren_im = Image.open(ren).convert("RGB")
    scaled = False
    if ren_im.size != (W, H):
        ren_im = ren_im.resize((W, H), Image.LANCZOS)
        scaled = True
    r = np.asarray(ren_im).astype(np.float32)

    err = np.abs(ref - r).max(axis=2)
    serr = r - ref

    print("=" * 60)
    print("源图 %s  %dx%d" % (os.path.basename(src), W, H))
    print("渲染 %s  %s" % (os.path.basename(ren),
                          "(已缩放回原尺寸)" if scaled else "(原始尺寸)"))
    print("=" * 60)

    # 1. 总体
    print("\n[1] 误差总体")
    print("  MAE %.3f   中位 %.2f   p90 %.2f   p99 %.2f   max %.0f"
          % (err.mean(), np.median(err), np.percentile(err, 90),
             np.percentile(err, 99), err.max()))
    print("  长尾分档：>30 %.2f%%   >60 %.2f%%   >100 %.2f%%   >150 %.2f%%"
          % tuple(100 * (err > t).mean() for t in (30, 60, 100, 150)))

    # 2. 偏色
    print("\n[2] 有符号误差（渲染 - 原图）")
    print("  R %+.2f   G %+.2f   B %+.2f"
          % (serr[..., 0].mean(), serr[..., 1].mean(), serr[..., 2].mean()))
    amax = int(np.argmax(np.abs([serr[..., i].mean() for i in range(3)])))
    direction = "整体偏亮" if sum(serr[..., i].mean() for i in range(3)) > 1.5 else (
        "整体偏暗" if sum(serr[..., i].mean() for i in range(3)) < -1.5 else "无明显偏色")
    print("  判定：%s（最大通道 %s %+.2f）"
          % (direction, "RGB"[amax], [serr[..., i].mean() for i in range(3)][amax]))

    # 3. 分区域
    lum = ref.mean(axis=2)
    mx = ref.max(axis=2); mn = ref.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    g = lum
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    grad = np.hypot(gx, gy)
    zones = {
        "平坦区 grad<3": grad < 3,
        "中等纹理 3-15": (grad >= 3) & (grad < 15),
        "强纹理 >=15": grad >= 15,
        "高饱和 >0.35": sat > 0.35,
        "低饱和 <=0.12": sat <= 0.12,
        "亮部 L>200": lum > 200,
        "暗部 L<80": lum < 80,
    }
    print("\n[3] 分区域误差")
    print("  %-18s %8s %8s %9s" % ("区域", "占比", "MAE", "有符号R"))
    zone_rows = []
    for name, m in zones.items():
        if m.sum() < 100:
            continue
        z = (name, 100 * m.mean(), float(err[m].mean()), float(serr[..., 0][m].mean()))
        zone_rows.append(z)
        print("  %-18s %7.1f%% %8.2f %+9.2f" % z)

    # 4. 色阶损失
    print("\n[4] 色阶丰富度")
    q1 = (ref // 8).astype(int)
    q2 = (r // 8).astype(int)
    n1 = len(np.unique(q1[..., 0] * 10000 + q1[..., 1] * 100 + q1[..., 2]))
    n2 = len(np.unique(q2[..., 0] * 10000 + q2[..., 1] * 100 + q2[..., 2]))
    print("  源图 %d 唯一色   渲染 %d 唯一色   保留 %.1f%%"
          % (n1, n2, 100.0 * n2 / max(n1, 1)))

    # 5. 最差区块
    BS = 96
    blocks = []
    step = BS // 2
    for by in range(0, max(1, H - BS), step):
        for bx in range(0, max(1, W - BS), step):
            m = float(err[by:by + BS, bx:bx + BS].mean())
            blocks.append((m, bx, by))
    blocks.sort(reverse=True)
    print("\n[5] 误差最大的 5 个 96x96 区域")
    for i, (m, bx, by) in enumerate(blocks[:5]):
        print("  #%d (%4d,%4d) 块MAE %.2f" % (i, bx, by, m))

    # 出放大对比
    tiles = []
    for i, (m, bx, by) in enumerate(blocks[:5]):
        x1, y1 = min(W, bx + BS), min(H, by + BS)
        if x1 - bx < 20 or y1 - by < 20:
            continue
        a = ref_im.crop((bx, by, x1, y1))
        b = ren_im.crop((bx, by, x1, y1))
        sc = max(2, int(280 / max(a.width, a.height)))
        a = a.resize((a.width * sc, a.height * sc), Image.LANCZOS)
        b = b.resize((b.width * sc, b.height * sc), Image.LANCZOS)
        pair = Image.new("RGB", (a.width * 2 + 8, a.height), (245, 245, 245))
        pair.paste(a, (0, 0)); pair.paste(b, (a.width + 8, 0))
        tiles.append(("blk%d @(%d,%d) MAE%.1f" % (i, bx, by, m), pair))
    if tiles:
        LAB = 20
        tw = max(t[1].width for t in tiles)
        th = sum(t[1].height + LAB for t in tiles)
        cv = Image.new("RGB", (tw, th), (248, 248, 248))
        d = ImageDraw.Draw(cv)
        y = 0
        for lb, imx in tiles:
            d.text((3, y + 4), lb + "   LEFT=SRC RIGHT=RENDER", fill=(0, 0, 0))
            cv.paste(imx, (0, y + LAB))
            y += imx.height + LAB
        p = os.path.join(outdir, "worst_blocks.png")
        cv.save(p)
        print("\n  放大对比 -> %s" % p)

    # 6. 误差热图
    Image.fromarray(np.clip(err * 4, 0, 255).astype(np.uint8)).save(
        os.path.join(outdir, "errmap.png"))
    print("  误差热图 -> %s" % os.path.join(outdir, "errmap.png"))

    # 保存数值
    res = {
        "src": os.path.basename(src), "size": [W, H],
        "mae": round(float(err.mean()), 3),
        "p90": round(float(np.percentile(err, 90)), 1),
        "long_tail": {str(t): round(100 * float((err > t).mean()), 2)
                      for t in (30, 60, 100, 150)},
        "signed": [round(float(serr[..., i].mean()), 2) for i in range(3)],
        "zones": [{"name": z[0], "pct": round(z[1], 1),
                   "mae": round(z[2], 2), "sr": round(z[3], 2)} for z in zone_rows],
        "colors": {"src": n1, "ren": n2, "keep_pct": round(100.0 * n2 / max(n1, 1), 1)},
        "worst_blocks": [{"mae": round(m, 2), "x": bx, "y": by}
                         for m, bx, by in blocks[:5]],
    }
    json.dump(res, open(os.path.join(outdir, "diag.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n数值 -> %s" % os.path.join(outdir, "diag.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
