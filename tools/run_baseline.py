"""run_baseline.py -- 基线批处理：默认参数跑抽样集，建 MAE 分布

## 设计要求

  1. **断点续跑**：每张图完成立即写 JSONL，中断后重跑跳过已完成
  2. **错误隔离**：单张失败记录 but 不中断整批
  3. **完整指标**：MAE / 耗时 / 体积 / 尺寸 / 色数 / 是否达标
  4. **固定 cwd**：vec.py 依赖 sys.path（坑 23）

## 输出

  output/dataset_profile/baseline.jsonl   逐图记录（可续跑）
  output/dataset_profile/baseline_summary.md  汇总

## 用法

  python run_baseline.py            跑抽样集全部
  python run_baseline.py --limit 20 只跑前 20 张（快速验证）
"""
import os
import io
import sys
import json
import time
import argparse
import subprocess
import numpy as np
from PIL import Image

def _find_root():

    """工作区根：优先环境变量，否则取仓库的上一级。



    原实现写死作者机器上的 F:\\SVG drawing。

    """

    env = os.environ.get("IMGSVG_ROOT")

    if env and os.path.isdir(env):

        return env

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))





ROOT = _find_root()
def _find_python():
    """定位可用解释器：优先环境变量，否则用当前解释器。

    原实现写死作者机器上的 D:\\anaconda3\\python.exe，
    换到别的机器必然找不到。
    """
    env = os.environ.get("IMGSVG_PYTHON")
    if env and os.path.isfile(env):
        return env
    return sys.executable


PY = _find_python()
SK = os.path.join(os.environ["DSH_HOME"], "skills", "dsh-img-to-svg", "scripts")
OUT = os.path.join(ROOT, "output", "dataset_profile")
WORK = os.path.join(ROOT, "output", "_baseline_work")
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

SAMPLE = os.path.join(OUT, "sample.json")
JSONL = os.path.join(OUT, "baseline.jsonl")

TARGET_MAE = 9.454          # 用户给定的质量基准


def load_done():
    done = {}
    if os.path.exists(JSONL):
        with io.open(JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done[r["key"]] = r
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", default=SAMPLE)
    args = ap.parse_args()

    sample = json.load(open(args.sample, encoding="utf-8"))
    if args.limit:
        sample = sample[:args.limit]
    done = load_done()
    todo = [s for s in sample if s["group"] + "/" + s["file"] not in done]

    print("=== 基线批处理 ===")
    print("  抽样 %d 张，已完成 %d，待跑 %d"
          % (len(sample), len(done), len(todo)))

    t_start = time.time()
    for i, s in enumerate(todo):
        key = s["group"] + "/" + s["file"]
        src = os.path.join(ROOT, s["path"])
        if not os.path.exists(src):
            print("  [%3d] 缺失 %s" % (i + 1, key)); continue
        stem = os.path.splitext(s["file"])[0]
        safe = stem.replace(" ", "_")[:80]
        svg = os.path.join(WORK, "%s__%s.svg" % (s["group"], safe))
        png = svg.replace(".svg", ".png")

        env = {k: v for k, v in os.environ.items() if not k.startswith("VEC_")}
        env["VEC_TITLE"] = safe

        t0 = time.time()
        r = subprocess.run([PY, os.path.join(SK, "vec.py"), src, svg],
                           capture_output=True, text=True, env=env, cwd=SK)
        dt = time.time() - t0

        rec = {
            "key": key, "group": s["group"], "file": s["file"],
            "w": s["w"], "h": s["h"], "mp": s["mp"],
            "colors8": s["colors8"],
            "size_bin": s["size_bin"], "col_bin": s["col_bin"],
            "flat_pct": s["flat_pct"], "detail_pct": s["detail_pct"],
            "sec": round(dt, 1),
            "ok": r.returncode == 0,
        }

        if r.returncode != 0:
            rec["error"] = (r.stderr.strip().splitlines() or ["?"])[-1][:200]
            rec["mae"] = None
        else:
            rec["svg_mb"] = round(os.path.getsize(svg) / 1048576, 2)
            rr = subprocess.run([PY, os.path.join(SK, "render_check.py"), svg, png],
                                capture_output=True, text=True, cwd=SK)
            if rr.returncode != 0 or not os.path.exists(png):
                rec["mae"] = None
                rec["error"] = "render failed"
            else:
                ref = np.asarray(Image.open(src).convert("RGB")).astype(np.float32)
                v = Image.open(png).convert("RGB")
                if v.size != (ref.shape[1], ref.shape[0]):
                    v = v.resize((ref.shape[1], ref.shape[0]), Image.LANCZOS)
                mae = float(np.abs(ref - np.asarray(v).astype(np.float32)).mean())
                rec["mae"] = round(mae, 3)
                rec["pass"] = mae <= TARGET_MAE

        with io.open(JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # 清理大文件避免占空间
        for p in (svg, png):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        mae_s = ("%.3f" % rec["mae"]) if rec["mae"] is not None else "FAIL"
        flag = "" if rec["mae"] is None else (" 达标" if rec["mae"] <= TARGET_MAE else "")
        el = time.time() - t_start
        eta = (el / (i + 1)) * (len(todo) - i - 1) if i > 0 else 0
        print("  [%3d/%d] %-34s %.2fMP %5.1fs MAE %-8s %-40s%s  ETA %.0fmin"
              % (i + 1, len(todo), s["file"][:34], s["mp"], dt, mae_s,
                 key[:40], flag, eta / 60) if rec["ok"] else
              "  [%3d/%d] %-34s 失败 %s" % (i + 1, len(todo), s["file"][:34],
                                            rec.get("error", "")[:50]))

    # 汇总
    done = load_done()
    rows = [r for r in done.values() if r.get("mae") is not None]
    if not rows:
        print("\n无有效记录"); return
    maes = np.array([r["mae"] for r in rows])
    secs = np.array([r["sec"] for r in rows])
    print("\n=== 基线汇总（%d 张有效）===" % len(rows))
    print("  MAE  中位 %.3f  均值 %.3f  min %.3f  max %.3f"
          % (np.median(maes), maes.mean(), maes.min(), maes.max()))
    print("  达标(<=%.3f) %d 张 (%.1f%%)"
          % (TARGET_MAE, int((maes <= TARGET_MAE).sum()),
             100 * (maes <= TARGET_MAE).mean()))
    print("  耗时 中位 %.1f s  总 %.1f min"
          % (np.median(secs), secs.sum() / 60))

    lines = ["# 基线结果", "",
             "样本 %d 张，基准 MAE <= %.3f" % (len(rows), TARGET_MAE), "",
             "| 指标 | 值 |", "|---|---|",
             "| MAE 中位 | %.3f |" % np.median(maes),
             "| MAE 均值 | %.3f |" % maes.mean(),
             "| MAE 范围 | %.3f ~ %.3f |" % (maes.min(), maes.max()),
             "| 达标数 | %d / %d (%.1f%%) |"
             % (int((maes <= TARGET_MAE).sum()), len(rows),
                100 * (maes <= TARGET_MAE).mean()),
             ""]
    # 按组统计
    lines += ["## 分组统计", "", "| 组 | n | 中位MAE | 达标率 |", "|---|---|---|---|"]
    for g in ("Noyu", "pixiv", "梅原生"):
        sub = [r for r in rows if r["group"] == g]
        if not sub:
            continue
        m = np.array([r["mae"] for r in sub])
        lines.append("| %s | %d | %.3f | %.1f%% |"
                     % (g, len(sub), np.median(m),
                        100 * (m <= TARGET_MAE).mean()))
    # 按尺寸统计
    lines += ["", "## 尺寸分层", "", "| 尺寸层 | n | 中位MAE | 达标率 |", "|---|---|---|---|"]
    for b in ("small", "mid", "large", "huge"):
        sub = [r for r in rows if r["size_bin"] == b]
        if not sub:
            continue
        m = np.array([r["mae"] for r in sub])
        lines.append("| %s | %d | %.3f | %.1f%% |"
                     % (b, len(sub), np.median(m), 100 * (m <= TARGET_MAE).mean()))
    # 最差的 15 张
    lines += ["", "## 最差 15 张", "", "| 文件 | 尺寸 | MP | 色数 | MAE |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: -x["mae"])[:15]:
        lines.append("| %s | %dx%d | %.2f | %d | **%.3f** |"
                     % (r["file"][:34], r["w"], r["h"], r["mp"],
                        r["colors8"], r["mae"]))
    open(os.path.join(OUT, "baseline_summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("  汇总 -> %s" % os.path.join(OUT, "baseline_summary.md"))


if __name__ == "__main__":
    main()
