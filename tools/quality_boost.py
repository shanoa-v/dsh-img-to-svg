"""quality_boost.py -- 高质量模式（可选）：高误差区域局部重绘

## 何时使用

**仅当用户明确要求「不计时间」「追求质量」「尽量还原」「能做到多好就多好」时启用。**

默认流程不要用它 —— 实测代价 10~20 倍耗时，收益有限：

| 指标 | 实测值 |
|---|---|
| 20 块重绘，全图 MAE | 11.34 -> 10.65（-6.07%） |
| 被选中块的平均块 MAE | 22.75 -> 8.24（-63.8%） |
| 耗时 | 62.2 s -> 647.7 s（**10.4 倍**） |
| 副作用 | 块与块、块与底层之间有可见色阶边界 |

## 实测成本（三档，口径不同，别混用）

| 用法 | 实测对象 | 全图 MAE | 耗时 | 成本 |
|---|---|---|---|---|
| regions 2 处 | illust_100531443 805x1300 | 9.028 -> 9.009 | 30.3 s | 1.7 倍 |
| regions 单处收益 | 嘴/眼核心区 | 嘴 7.63、眼 10.69 | 每处 +6.3 s | 每处 +35% |
| auto 3 块 | user_char 1066x600 | 8.619 -> 8.166 (-5.25%) | 44.7 s | 3.9 倍 |
| auto 20 块 | 银杏图 1200x1993 | 11.34 -> 10.65 (-6.07%) | 647.7 s | 10.4 倍 |

收益只作用在被覆盖区域。小图 auto 效率更高（每块占全图面积更大）。

## 用法

  # 推荐：只做少数关键语义区域（成本 +35%/区域，收益是本质改善）
  python quality_boost.py <原图> <输出.svg> --regions "412,431,425,437;357,330,395,357"

  # 或：按误差自动选块（成本高，适合小图或不在乎时间）
  python quality_boost.py <原图> <输出.svg> --auto 20

  # 参数
  --block 96        块尺寸（auto 模式）
  --pad 16          每块外扩
  --up 8            放大倍数
  --n 16            局部色数（实测小 N 优于大 N）
  --overlap 0.5     块步进比例

## 为什么局部量化能改善

嘴部 18 个暗像素占全图 0.0017%，N=56 时每中心覆盖 1.79% -> 分不到。
局部化后占 0.226%，提高 133 倍 -> 能分到。
实测局部量化让区域 MAE 降 47.7%~62.3%（4 图 10 区域，无一例外）。

## 必须关闭簇合并

局部设 16 色时，默认 VEC_MERGE_STEP=10 会把 16 色合并成 8~12 色。
必须 VEC_MERGE_STEP=0 才能拿到完整色阶（第 3-4 轮实测）。
"""
import os
import sys
import re
import io
import json
import time
import argparse
import subprocess
import numpy as np
from PIL import Image

PY = sys.executable
SK = os.path.join(os.environ.get("DSH_HOME", ""), "skills", "dsh-img-to-svg", "scripts")


def run_vec(src, svg, extra_env=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("VEC_")}
    env["VEC_TITLE"] = os.path.basename(svg)
    if extra_env:
        env.update(extra_env)
    r = subprocess.run([PY, os.path.join(SK, "vec.py"), src, svg],
                       capture_output=True, text=True, env=env, cwd=SK)
    return r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1]


def render(svg, png):
    subprocess.run([PY, os.path.join(SK, "render_check.py"), svg, png],
                   capture_output=True, text=True, cwd=SK)
    return os.path.exists(png)


def local_redraw(im, bbox, pad, up, n, tag, workdir):
    W, H = im.size
    x0, y0, x1, y1 = bbox
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
    crop = im.crop((cx0, cy0, cx1, cy1))
    cw, ch = crop.size
    if cw < 12 or ch < 12:
        return None
    tmp = os.path.join(workdir, "_lr_%s.png" % tag)
    crop.resize((cw * up, ch * up), Image.LANCZOS).save(tmp)
    svg = os.path.join(workdir, "_lr_%s.svg" % tag)
    ok, err = run_vec(tmp, svg, {"VEC_N": str(n), "VEC_MERGE_STEP": "0"})
    if not ok:
        return None
    png = svg.replace(".svg", ".png")
    if not render(svg, png):
        return None
    v = Image.open(png).convert("RGB").resize((cw, ch), Image.LANCZOS)
    return np.asarray(v).astype(np.float32), (cx0, cy0, cx1, cy1)


def main():
    ap = argparse.ArgumentParser(description="高质量模式：高误差区域局部重绘")
    ap.add_argument("src")
    ap.add_argument("out_svg")
    ap.add_argument("--regions", default="",
                    help='手工语义区域，格式 "x0,y0,x1,y1;x0,y0,x1,y1"')
    ap.add_argument("--auto", type=int, default=0,
                    help="自动按误差选 N 个块（与 --regions 二选一）")
    ap.add_argument("--block", type=int, default=96)
    ap.add_argument("--pad", type=int, default=16)
    ap.add_argument("--up", type=int, default=8)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--overlap", type=float, default=0.5)
    args = ap.parse_args()

    workdir = os.path.dirname(os.path.abspath(args.out_svg)) or "."
    os.makedirs(workdir, exist_ok=True)

    # 内部 subprocess 的 cwd 固定到 skill/scripts，因此必须用绝对路径
    args.src = os.path.abspath(args.src)
    args.out_svg = os.path.abspath(args.out_svg)
    im = Image.open(args.src).convert("RGB")
    W, H = im.size
    ref = np.asarray(im).astype(np.float32)
    print("=== 高质量模式 ===")
    print("  输入 %s  %dx%d" % (os.path.basename(args.src), W, H))

    base_svg = os.path.join(workdir, "_qbase.svg")
    base_png = os.path.join(workdir, "_qbase.png")
    t0 = time.time()
    ok, err = run_vec(args.src, base_svg)
    if not ok:
        print("  基础矢量化失败:", err); return 1
    render(base_svg, base_png)
    t_base = time.time() - t0
    b_im = Image.open(base_png).convert("RGB")
    if b_im.size != (W, H):
        b_im = b_im.resize((W, H), Image.LANCZOS)
    base = np.asarray(b_im).astype(np.float32)
    mae0 = float(np.abs(ref - base).mean())
    print("  基础 MAE %.4f  耗时 %.1f s" % (mae0, t_base))

    canvas = base.copy()
    t0 = time.time()
    n_done = 0

    if args.regions:
        print("\n  手工语义区域模式")
        boxes = []
        for s in args.regions.split(";"):
            s = s.strip()
            if not s:
                continue
            p = [int(v) for v in s.split(",")]
            if len(p) == 4:
                boxes.append(tuple(p))
        # ---- 越界与反向坐标保护（Q1）----
        # regions 由用户手输，常见两种坏输入都会让 local_redraw 里的
        # crop / 切片抛异常：
        #   1) 坐标超出图像范围（如 533x300 图给 y=431）
        #   2) 端点写反（x1 < x0）
        # 实测文档自带的示例命令 --regions "412,431,425,437;357,330,395,357"
        # 在 533x300 图上直接 ValueError: Coordinate 'lower' is less than 'upper'，
        # 并留下 _qbase.png/svg 残骸。
        _safe = []
        for bb in boxes:
            x0, y0, x1, y1 = bb
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
                print("    跳过越界区域 (%d,%d,%d,%d)（图像 %dx%d）"
                      % (bb[0], bb[1], bb[2], bb[3], W, H))
                continue
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W - 1, x1), min(H - 1, y1)
            if x1 < x0 or y1 < y0:
                print("    跳过无效区域 (%d,%d,%d,%d)" % bb)
                continue
            _safe.append((x0, y0, x1, y1))
        if not _safe:
            # 全部区域都越界：这里直接退出，但要先清掉 base 阶段已经写出的
            # _qbase.png/svg —— 否则用户拿不到任何结果，目录里却多两个
            # 几百 KB 的孤儿中间件（实测 _qbase.png 115KB + _qbase.svg 441KB）。
            for _fn in ("_qbase.svg", "_qbase.png"):
                try:
                    os.remove(os.path.join(workdir, _fn))
                except OSError:
                    pass
            raise SystemExit(
                "所有 --regions 均超出图像范围（图像 %dx%d）。"
                "请给 x0,y0,x1,y1 形式的合法坐标，且 x1>=x0、y1>=y0。" % (W, H))
        boxes = _safe
        for i, bb in enumerate(boxes):
            got = local_redraw(im, bb, args.pad, args.up, args.n, "r%d" % i, workdir)
            if got is None:
                print("    区域%d 失败" % i); continue
            loc, box = got
            x0, y0, x1, y1 = box
            # 只贴核心，pad 区留底层（减少边界）
            bx0, by0, bx1, by1 = bb
            sx, sy = bx0 - x0, by0 - y0
            canvas[by0:by1 + 1, bx0:bx1 + 1] = loc[sy:sy + (by1 - by0 + 1),
                                                     sx:sx + (bx1 - bx0 + 1)]
            n_done += 1
            cb = float(np.abs(ref[by0:by1 + 1, bx0:bx1 + 1]
                              - canvas[by0:by1 + 1, bx0:bx1 + 1]).mean())
            print("    区域%d %s 完成  核心MAE %.2f" % (i, bb, cb))

    elif args.auto > 0:
        print("\n  自动误差选块模式 top%d" % args.auto)
        step = max(1, int(args.block * args.overlap))
        blocks = []
        for by in range(0, max(1, H - args.block + 1), step):
            for bx in range(0, max(1, W - args.block + 1), step):
                m = float(np.abs(ref[by:by + args.block, bx:bx + args.block]
                                 - base[by:by + args.block, bx:bx + args.block]).mean())
                blocks.append((m, bx, by))
        blocks.sort(reverse=True)
        print("    候选 %d 块" % len(blocks))
        for i, (m0, bx, by) in enumerate(blocks[:args.auto]):
            bb = (bx, by, min(W - 1, bx + args.block - 1), min(H - 1, by + args.block - 1))
            got = local_redraw(im, bb, args.pad, args.up, args.n, "b%d" % i, workdir)
            if got is None:
                continue
            loc, box = got
            x0, y0, x1, y1 = box
            bh = min(args.block, H - by)
            bw = min(args.block, W - bx)
            canvas[by:by + bh, bx:bx + bw] = loc[by - y0:by - y0 + bh, bx - x0:bx - x0 + bw]
            n_done += 1
            if (i + 1) % 5 == 0:
                cur = float(np.abs(ref - canvas).mean())
                print("      %2d/%d  当前全图MAE %.4f (%+.2f%%)"
                      % (i + 1, args.auto, cur, 100 * (cur - mae0) / mae0))
    else:
        print("\n  未指定 --regions 或 --auto，退出"); return 2

    t_local = time.time() - t0
    mae1 = float(np.abs(ref - canvas).mean())
    print("\n=== 结果 ===")
    print("  处理区域/块数 %d" % n_done)
    print("  全图 MAE %.4f -> %.4f (%+.2f%%)" % (mae0, mae1, 100 * (mae1 - mae0) / mae0))
    print("  耗时 基础 %.1f s + 局部 %.1f s = %.1f s (%.1f 倍)"
          % (t_base, t_local, t_base + t_local, (t_base + t_local) / max(t_base, 1e-6)))

    # 写出最终 SVG
    # 注意：这里输出的是「按块重绘后的栅格再矢量化」不现实，所以采用
    # 「基础SVG + 局部SVG 图层」的方式，与 proto_semantic_overlay 一致。
    print("\n  注意：本脚本给出的是像素域效果与指标；")
    print("        要生成可编辑 SVG，请用 proto_semantic_overlay.py 的叠加方式。")
    out_png = os.path.splitext(args.out_svg)[0] + "_render.png"
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).save(out_png)
    print("  渲染结果 -> %s" % out_png)

    json.dump({"mae_base": mae0, "mae_after": mae1, "count": n_done,
               "t_base": round(t_base, 1), "t_local": round(t_local, 1),
               "ratio": round((t_base + t_local) / max(t_base, 1e-6), 2)},
              open(os.path.splitext(args.out_svg)[0] + "_result.json", "w",
                   encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- 清理中间产物（第 8 轮 P1）----
    # 本脚本**独立运行**时（文档档位3 给的就是这条命令）没有任何清理逻辑：
    # 实测每次成功运行在输出目录留下
    #   _qbase.svg 1.21MB + _qbase.png 181KB + _lr_r0.svg 1.21MB + _lr_r0.png 314KB
    # = **2.78MB 残留**，而交付物（_render.png + _result.json）仅 0.17MB，
    # 残留是交付物的 16 倍。
    # 之前只在 watch_draw.py 的 _cleanup_mid 里清理（走 --tier 3 才有），
    # 独立运行路径完全没覆盖。这里补上，交付物一律保留。
    _stem = os.path.splitext(os.path.basename(args.out_svg))[0]
    _keep = {_stem + "_render.png", _stem + "_result.json",
             os.path.basename(args.out_svg)}
    _n_rm = 0
    _wdir = os.path.dirname(os.path.abspath(args.out_svg)) or "."
    try:
        for _f in os.listdir(_wdir):
            if _f in _keep:
                continue
            if _f.startswith("_qbase") or _f.startswith("_lr_") \
                    or _f.startswith("_lhb"):
                try:
                    os.remove(os.path.join(_wdir, _f))
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
