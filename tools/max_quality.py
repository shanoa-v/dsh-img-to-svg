"""max_quality.py -- 高质量档（超采样 + 分块重绘，接受白点换取细节）

## 触发条件

用户明确说「不计时间」「追求质量」「尽量还原」「能做到多好就多好」「有白点也没关系」
等措辞时使用。**默认流程不要用。**

## 实测数据（银杏图 1200×1993，密集纹理典型失败案例）

| 方案 | MAE | 相对默认 | 耗时 | 平坦区网点 |
|---|---|---|---|---|
| 默认 W=1200 | 11.340 | - | 62 s | 9.46% |
| W=1400 | 10.720 | -5.5% | 80 s | 9.11% |
| W=1800 | 9.985 | -11.9% | 130 s | 11.68% |
| W=2400 | 10.019 | -11.6% | 232 s | 16.57% |
| **W=1800 + 20块局部重绘** | **9.454** | **-16.6%** | 378 s | 11.86% |

## 关键结论

1. **W=1800 是超采样的最优点**（2400 反而退步，收益饱和）
2. **网点来自 W 超采样**，局部重绘几乎不增加（11.68 -> 11.86）
3. **两者可叠加**：W 的 -11.9% + 局部重绘的 -5.33% = -16.6%
4. **W=1800 组合耗时反而少于单做局部重绘**（378 s vs 648 s），
   因为基础图质量更高，局部重绘起点更好

## 白点是可接受的代价

用户明确表示「有白点就有白点，W1400 细节不行」。
平坦区网点：W=1400 是 9.11%，W=1800 是 11.68%，差 2.57 个百分点，
换来 MAE 从 10.720 降到 9.985（再多降 6.8%）。

## 用法

  python max_quality.py <源图> <输出目录> [--w 1800] [--blocks 20] [--block 96]

产出：
  <输出目录>/max.svg          基础 SVG（W=1800）
  <输出目录>/max_final.png    最终合成图（含局部重绘）
  <输出目录>/max_compare.png  三联对比
  <输出目录>/max_result.json  指标
"""
import os
import sys
import json
import time
import argparse
import subprocess
import numpy as np
from PIL import Image, ImageDraw

PY = sys.executable
SK = os.path.join(os.environ.get("DSH_HOME", ""), "skills", "dsh-img-to-svg", "scripts")


def run_vec(src, svg, extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("VEC_")}
    env["VEC_TITLE"] = os.path.basename(svg)
    if extra:
        env.update(extra)
    r = subprocess.run([PY, os.path.join(SK, "vec.py"), src, svg],
                       capture_output=True, text=True, env=env, cwd=SK)
    return r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1]


def render(svg, png):
    subprocess.run([PY, os.path.join(SK, "render_check.py"), svg, png],
                   capture_output=True, text=True, cwd=SK)
    return os.path.exists(png)


def main():
    ap = argparse.ArgumentParser(description="高质量档：W 超采样 + 局部重绘")
    ap.add_argument("src")
    ap.add_argument("outdir")
    ap.add_argument("--w", type=int, default=1800, help="超采样宽度（最优 1800）")
    ap.add_argument("--blocks", type=int, default=20, help="局部重绘块数")
    ap.add_argument("--block", type=int, default=96, help="块尺寸")
    ap.add_argument("--pad", type=int, default=16)
    ap.add_argument("--up", type=int, default=8)
    ap.add_argument("--n", type=int, default=16, help="局部色数（16 最优）")
    args = ap.parse_args()

    args.src = os.path.abspath(args.src)
    args.outdir = os.path.abspath(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    im = Image.open(args.src).convert("RGB")
    W, H = im.size
    ref = np.asarray(im).astype(np.float32)
    print("=== 高质量档 ===")
    print("  输入 %s  %dx%d" % (os.path.basename(args.src), W, H))

    # 1. W 超采样基础
    print("\n[1] W=%d 超采样" % args.w)
    base_svg = os.path.join(args.outdir, "max.svg")
    base_png = os.path.join(args.outdir, "_max_base.png")
    t0 = time.time()
    ok, err = run_vec(args.src, base_svg, {"VEC_W": str(args.w)})
    if not ok:
        print("  失败:", err); return 1
    render(base_svg, base_png)
    t_base = time.time() - t0
    b_im = Image.open(base_png).convert("RGB")
    if b_im.size != (W, H):
        b_im = b_im.resize((W, H), Image.LANCZOS)
    base = np.asarray(b_im).astype(np.float32)
    mae0 = float(np.abs(ref - base).mean())
    print("  MAE %.4f  耗时 %.0f s" % (mae0, t_base))

    # 2. 误差驱动局部重绘
    print("\n[2] 误差驱动局部重绘 top%d（块 %dx%d）" % (args.blocks, args.block, args.block))
    BL = args.block
    blocks = []
    step = BL // 2
    for by in range(0, max(1, H - BL), step):
        for bx in range(0, max(1, W - BL), step):
            m = float(np.abs(ref[by:by + BL, bx:bx + BL]
                             - base[by:by + BL, bx:bx + BL]).mean())
            blocks.append((m, bx, by))
    blocks.sort(reverse=True)
    print("  候选 %d 块" % len(blocks))

    canvas = base.copy()
    t0 = time.time()
    done = 0
    for i, (m0, bx, by) in enumerate(blocks[:args.blocks]):
        x0, y0 = max(0, bx - args.pad), max(0, by - args.pad)
        x1, y1 = min(W, bx + BL + args.pad), min(H, by + BL + args.pad)
        crop = im.crop((x0, y0, x1, y1))
        cw, ch = crop.size
        tp = os.path.join(args.outdir, "_mb%02d.png" % i)
        crop.resize((cw * args.up, ch * args.up), Image.LANCZOS).save(tp)
        sv = os.path.join(args.outdir, "_mb%02d.svg" % i)
        ok, _ = run_vec(tp, sv, {"VEC_N": str(args.n), "VEC_MERGE_STEP": "0"})
        if not ok:
            continue
        pg = sv.replace(".svg", ".png")
        if not render(sv, pg):
            continue
        v = Image.open(pg).convert("RGB").resize((cw, ch), Image.LANCZOS)
        loc = np.asarray(v).astype(np.float32)
        bh, bw = min(BL, H - by), min(BL, W - bx)
        canvas[by:by + bh, bx:bx + bw] = loc[by - y0:by - y0 + bh, bx - x0:bx - x0 + bw]
        done += 1
        if (i + 1) % 5 == 0:
            cur = float(np.abs(ref - canvas).mean())
            print("    %2d/%d  当前 MAE %.4f (%+.2f%%)"
                  % (i + 1, args.blocks, cur, 100 * (cur - mae0) / mae0))
    t_loc = time.time() - t0

    mae1 = float(np.abs(ref - canvas).mean())
    print("\n=== 结果 ===")
    print("  块数 %d" % done)
    print("  MAE %.4f -> %.4f (%+.2f%%)" % (mae0, mae1, 100 * (mae1 - mae0) / mae0))
    print("  总耗时 %.0f s（%.1f 倍默认）" % (t_base + t_loc, (t_base + t_loc) / max(t_base, 1e-6) * 1.0))

    # 3. 出图
    final_png = os.path.join(args.outdir, "max_final.png")
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).save(final_png)

    sc = min(1.0, 720.0 / W)
    tw, th = int(W * sc), int(H * sc)
    tiles = [("ORIGINAL", im.resize((tw, th), Image.LANCZOS)),
             ("W=%d  MAE %.2f" % (args.w, mae0), b_im.resize((tw, th), Image.LANCZOS)),
             ("MAX  MAE %.2f" % mae1,
              Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).resize((tw, th), Image.LANCZOS))]
    LAB = 24
    cv = Image.new("RGB", (tw * 3 + 16, LAB + th), (248, 248, 248))
    d = ImageDraw.Draw(cv)
    for i, (lb, imx) in enumerate(tiles):
        x = i * (tw + 8)
        d.text((x + 4, 5), lb, fill=(0, 0, 0))
        cv.paste(imx, (x, LAB))
    cmp_p = os.path.join(args.outdir, "max_compare.png")
    cv.save(cmp_p)

    json.dump({"src": os.path.basename(args.src), "size": [W, H],
               "w": args.w, "blocks": done,
               "mae_base_default": None, "mae_w": round(mae0, 4),
               "mae_final": round(mae1, 4),
               "t_base": round(t_base, 1), "t_local": round(t_loc, 1),
               "t_total": round(t_base + t_loc, 1)},
              open(os.path.join(args.outdir, "max_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("\n  最终图 -> %s" % final_png)
    print("  对比图 -> %s" % cmp_p)

    # ---- 清理中间产物（P2）----
    # 本脚本会生成 _max_base.png + 每块 _mb%02d.png/svg，
    # 早期版本零清理，实测单张图残留 31 MB（20 对 _mb* + _max_base）。
    # 交付物（final_png / cmp_p / max_result.json）必须保留。
    _keep = {os.path.basename(final_png), os.path.basename(cmp_p),
             "max_result.json"}
    _n_rm = 0
    try:
        for _f in os.listdir(args.outdir):
            if _f in _keep:
                continue
            if _f.startswith("_mb") or _f.startswith("_max_base"):
                try:
                    os.remove(os.path.join(args.outdir, _f))
                    _n_rm += 1
                except OSError:
                    pass
    except OSError:
        pass
    if _n_rm:
        print("  已清理中间产物 %d 个" % _n_rm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
