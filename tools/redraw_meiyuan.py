"""redraw_meiyuan.py -- 梅原生全量高质量重绘（可断点续跑）

## 目标

用户要求：把梅原生 95 张全部按"参考 SVG 那种质量"重绘。
参考物实测为 58.82 MP 原图经本管线输出（168216 path、56 色、补缝）。
其质量来源是**原始像素多**，所以小图要靠上采样逼近。

## 策略（等物理放大测试定档后填入最终值）

初步实测（VEC_W 倍数，管线内上限 2x）：
  1.02MP 852px:  1x 8.832 -> 2x 7.643  (p99 106->73)
  2.53MP 1300px: 1x 12.716 -> 2x 11.065 (p99 111->85)
  4.74MP 1498px: 1x 7.125 -> 2x 6.297  (p99 62->46)

即**越大越好，且长尾同步改善**。物理放大可突破 2x 上限。

## 参数

--k        物理放大倍数（默认 2）
--limit    只跑前 N 张
--target-w 目标工作宽度（若设置，按此反推每图的 k）

## 产出

  output/final_v1.2.0/梅原生/
    svg/  png/  cmp/  _summary.json

断点续跑：已存在且非空的 svg 跳过。
"""
import os
import io
import json
import argparse
import subprocess
import sys
import time
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
PY = sys.executable
SK = os.path.join(os.environ["DSH_HOME"], "skills", "dsh-img-to-svg", "scripts")
BASE = os.path.join(ROOT, "input", "new_train", "梅原生")
WORK = os.path.join(ROOT, "workspace", "redraw_up")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=0,
                    help="固定倍数；0 表示按 --target-w 自动计算")
    ap.add_argument("--target-w", type=int, default=4000,
                    help="目标长边像素（自动算 k 时使用）")
    ap.add_argument("--src", default="",
                    help="单张图片路径或目录；省略则扫 input/new_train/梅原生")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--match", default="")
    ap.add_argument("--outdir",
                    default=os.path.join(ROOT, "output", "final_v1.2.0", "梅原生"))
    args = ap.parse_args()

    os.makedirs(os.path.join(args.outdir, "svg"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "png"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "cmp"), exist_ok=True)
    os.makedirs(WORK, exist_ok=True)

    # 支持 --src：单文件、目录，或不传（默认扫梅原生）
    IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
    if args.src:
        srcp = os.path.abspath(args.src)
        if not os.path.exists(srcp):
            print(f"错误：路径不存在 {srcp}")
            return 1
        if os.path.isfile(srcp):
            if not srcp.lower().endswith(IMG_EXT):
                print(f"错误：不是支持的图片格式 {srcp}")
                return 1
            files = [os.path.basename(srcp)]
            base_dir = os.path.dirname(srcp)
        else:
            base_dir = srcp
            files = sorted(f for f in os.listdir(base_dir)
                           if f.lower().endswith(IMG_EXT))
    else:
        base_dir = BASE
        files = sorted(f for f in os.listdir(base_dir)
                       if f.lower().endswith(IMG_EXT))

    if args.match:
        files = [f for f in files if args.match in f]
    if args.limit:
        files = files[:args.limit]

    import math
    print("=== 梅原生高质量重绘 ===")
    print("  模式: %s" % (("固定 k=%d" % args.k) if args.k > 0
                          else "目标长边 %dpx" % args.target_w))
    print("  待处理 %d 张" % len(files))

    rows = []
    t_start = time.time()
    for i, f in enumerate(files):
        src = os.path.join(base_dir, f)
        stem = os.path.splitext(f)[0]
        svg = os.path.join(args.outdir, "svg", stem + ".svg")
        png = os.path.join(args.outdir, "png", stem + "_render.png")

        # ---- 断点续跑：必须校验参数一致，否则会报出旧参数结果 ----
        # 早期判据只看「svg>1KB 且 png 存在」，不比对本次的 k/target_w。
        # 实测后果：用 --k 1 重跑一个已含 k=4 产物的 outdir，脚本打印
        # 「模式: 固定 k=1」却直接跳过，并报出旧 k=4 的 MAE 5.541，
        # _summary.json 里 mode=fixed k=1 与 rows[0].k=4 自相矛盾。
        # 现在读 _rows.jsonl 里该图的旧 k，与本次 k 不同则重新生成。
        _prev_k = None
        try:
            _rp = os.path.join(args.outdir, "_rows.jsonl")
            if os.path.exists(_rp):
                for _ln in io.open(_rp, encoding="utf-8"):
                    _ln = _ln.strip()
                    if not _ln:
                        continue
                    _r = json.loads(_ln)
                    if str(_r.get("file", "")) == f or \
                            str(_r.get("stem", "")) == stem:
                        _prev_k = _r.get("k")
        except Exception:
            _prev_k = None

        if os.path.exists(svg) and os.path.getsize(svg) > 1024 and os.path.exists(png):
            _want_k = args.k if args.k > 0 else None
            if _prev_k is not None and _want_k is not None and _prev_k != _want_k:
                print("  [%3d/%d] %-34s 已有 k=%s 产物，本次 k=%s → 重新生成"
                      % (i + 1, len(files), f[:34], _prev_k, _want_k))
            elif _prev_k is None:
                print("  [%3d/%d] %-34s 已存在但无参数记录，保守重跑"
                      % (i + 1, len(files), f[:34]))
            else:
                print("  [%3d/%d] %-34s 已存在（k 一致），跳过"
                      % (i + 1, len(files), f[:34]))
                continue

        im = Image.open(src).convert("RGB")
        W0, H0 = im.size
        mp0 = W0 * H0 / 1e6

        # 决定放大倍数
        if args.k > 0:
            k = args.k
        else:
            ls = max(W0, H0)
            k = 1 if ls >= args.target_w else max(
                1, min(8, math.ceil(args.target_w / ls)))

        # 物理放大
        if k > 1:
            up = os.path.join(WORK, "%s_up%d.png" % (stem[:40], k))
            if not os.path.exists(up):
                im.resize((W0 * k, H0 * k), Image.LANCZOS).save(up)
            work_src = up
        else:
            work_src = src

        env = {k: v for k, v in os.environ.items() if not k.startswith("VEC_")}
        env["VEC_TITLE"] = stem
        env["VEC_SEED"] = "42"

        t0 = time.time()
        r = subprocess.run([PY, os.path.join(SK, "vec.py"), work_src, svg],
                           capture_output=True, text=True, env=env, cwd=SK)
        dt = time.time() - t0
        if r.returncode != 0:
            print("  [%3d/%d] %-34s 失败 %s"
                  % (i + 1, len(files), f[:34],
                     (r.stderr.strip().splitlines() or ["?"])[-1][:60]))
            if k > 1 and os.path.exists(work_src):
                try:
                    os.remove(work_src)
                except Exception:
                    pass
            continue

        rr = subprocess.run([PY, os.path.join(SK, "render_check.py"), svg, png],
                            capture_output=True, text=True, cwd=SK)
        if rr.returncode != 0 or not os.path.exists(png):
            print("  [%3d/%d] %-34s 渲染失败" % (i + 1, len(files), f[:34]))
            continue

        # 测量：缩回原尺寸比较
        ref = np.asarray(im).astype(np.float32)
        v = Image.open(png).convert("RGB")
        if v.size != (W0, H0):
            v = v.resize((W0, H0), Image.LANCZOS)
        ren = np.asarray(v).astype(np.float32)
        e = np.abs(ref - ren)
        em = e.max(axis=2)

        # 三联对比
        sc = min(1.0, 700.0 / W0)
        tw, th = max(1, int(W0 * sc)), max(1, int(H0 * sc))
        a = im.resize((tw, th), Image.LANCZOS)
        b = v.resize((tw, th), Image.LANCZOS)
        heat = Image.fromarray(
            np.clip(em * 4, 0, 255).astype(np.uint8)).convert("RGB")
        heat = heat.resize((tw, th), Image.LANCZOS)
        LAB = 22
        cv = Image.new("RGB", (tw * 3 + 16, LAB + th), (248, 248, 248))
        dr = ImageDraw.Draw(cv)
        for j, (lb, imx) in enumerate([
                ("ORIGINAL", a),
                ("SVG  MAE %.2f" % e.mean(), b),
                ("DIFF x4  p99 %.0f" % np.percentile(em, 99), heat)]):
            x = j * (tw + 8)
            dr.text((x + 4, 4), lb, fill=(0, 0, 0))
            cv.paste(imx, (x, LAB))
        cv.save(os.path.join(args.outdir, "cmp", stem + "_compare.png"))

        rec = {
            "file": f, "mp": round(mp0, 2), "w": W0, "h": H0,
            "k": k, "up": [W0 * k, H0 * k],
            "mae": round(float(e.mean()), 3),
            "p99": round(float(np.percentile(em, 99)), 1),
            "gt60": round(100 * float((em > 60).mean()), 2),
            "sec": round(dt, 1),
            "svg_mb": round(os.path.getsize(svg) / 1048576, 2),
        }
        rows.append(rec)

        # 清理放大临时图
        if k > 1 and os.path.exists(work_src):
            try:
                os.remove(work_src)
            except Exception:
                pass

        el = time.time() - t_start
        eta = (el / (i + 1)) * (len(files) - i - 1) / 60
        print("  [%3d/%d] %-30s %5.2fMP %6.1fs MAE %6.3f p99 %5.0f "
              "gt60 %5.2f%% %7.1fMB ETA %.0fmin"
              % (i + 1, len(files), f[:30], mp0, dt, rec["mae"], rec["p99"],
                 rec["gt60"], rec["svg_mb"], eta))

        # 增量落盘，防止中断丢数据
        with io.open(os.path.join(args.outdir, "_rows.jsonl"), "a",
                     encoding="utf-8") as fo:
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 汇总（从 jsonl 重建，兼容续跑）
    all_rows = []
    p = os.path.join(args.outdir, "_rows.jsonl")
    if os.path.exists(p):
        for line in io.open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    all_rows.append(json.loads(line))
                except Exception:
                    pass
    if all_rows:
        ms = np.array([r["mae"] for r in all_rows])
        summary = {
            "count": len(all_rows),
            "mode": ("fixed k=%d" % args.k) if args.k > 0 else ("target_w=%d" % args.target_w),
            "mae_median": round(float(np.median(ms)), 3),
            "mae_mean": round(float(ms.mean()), 3),
            "mae_min": round(float(ms.min()), 3),
            "mae_max": round(float(ms.max()), 3),
            "p99_median": round(float(np.median([r["p99"] for r in all_rows])), 1),
            "gt60_median": round(float(np.median([r["gt60"] for r in all_rows])), 2),
            "total_mb": round(sum(r["svg_mb"] for r in all_rows), 1),
            "rows": all_rows,
        }
        json.dump(summary, open(os.path.join(args.outdir, "_summary.json"), "w",
                                encoding="utf-8"), ensure_ascii=False, indent=1)
        print()
        print("=== 汇总（累计 %d 张）===" % len(all_rows))
        print("  MAE 中位 %.3f  均值 %.3f  范围 %.3f~%.3f"
              % (summary["mae_median"], summary["mae_mean"],
                 summary["mae_min"], summary["mae_max"]))
        print("  p99 中位 %.0f   gt60 中位 %.2f%%   总体积 %.1f MB"
              % (summary["p99_median"], summary["gt60_median"],
                 summary["total_mb"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
