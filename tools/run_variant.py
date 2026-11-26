"""run_variant.py -- 用指定环境变量跑抽样集，与基线对比

## 用途

基线（run_baseline.py）跑完 158 张后，用同样的抽样集跑改进配置，
逐图对比，确认「无退化」这个硬条件。

## 用法

  python run_variant.py --tag seal --env VEC_SEAL=1.0 --limit 40
  python run_variant.py --tag combo --env VEC_BASECOL=1 --env VEC_SEAL=1.0

## 输出

  output/dataset_profile/<tag>.jsonl
  output/dataset_profile/<tag>_vs_baseline.md
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
WORK = os.path.join(ROOT, "output", "_variant_work")
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

BASELINE = os.path.join(OUT, "baseline.jsonl")
SAMPLE = os.path.join(OUT, "sample.json")


def load_jsonl(p):
    d = {}
    if not os.path.exists(p):
        return d
    for line in io.open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            d[r["key"]] = r
        except Exception:
            pass
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-gorup", default="")
    args = ap.parse_args()

    extra = {}
    for e in args.env:
        if "=" in e:
            k, v = e.split("=", 1)
            extra[k] = v
    print("=== 变体 %s  环境 %s ===" % (args.tag, extra))

    sample = json.load(open(SAMPLE, encoding="utf-8"))
    base = load_jsonl(BASELINE)
    # 只跑基线里已完成的（才能逐图对比）
    sample = [s for s in sample if s["group"] + "/" + s["file"] in base]
    if args.limit:
        sample = sample[:args.limit]
    if not sample:
        print("没有可对比的样本（基线未完成？）"); return 1

    jsonl = os.path.join(OUT, "%s.jsonl" % args.tag)
    done = load_jsonl(jsonl)
    if done:
        # 以已有记录为准（续跑）：清空重写以便与当前 env 一致
        print("  清理旧的 %s.jsonl（%d 条）" % (args.tag, len(done)))
        os.remove(jsonl)
        done = {}

    print("  待跑 %d 张（基线已完成 %d）" % (len(sample), len(base)))
    t_start = time.time()
    for i, s in enumerate(sample):
        key = s["group"] + "/" + s["file"]
        src = os.path.join(ROOT, s["path"])
        if not os.path.exists(src):
            continue
        stem = os.path.splitext(s["file"])[0].replace(" ", "_")[:80]
        svg = os.path.join(WORK, "%s__%s__%s.svg" % (args.tag, s["group"], stem))
        png = svg.replace(".svg", ".png")

        env = {k: v for k, v in os.environ.items() if not k.startswith("VEC_")}
        env["VEC_TITLE"] = stem
        env["VEC_SEED"] = "42"
        env.update(extra)

        t0 = time.time()
        r = subprocess.run([PY, os.path.join(SK, "vec.py"), src, svg],
                           capture_output=True, text=True, env=env, cwd=SK)
        dt = time.time() - t0
        rec = {"key": key, "group": s["group"], "file": s["file"],
               "mp": s["mp"], "sec": round(dt, 1), "ok": r.returncode == 0}
        if r.returncode != 0:
            rec["mae"] = None
            rec["error"] = (r.stderr.strip().splitlines() or ["?"])[-1][:150]
        else:
            rec["svg_mb"] = round(os.path.getsize(svg) / 1048576, 2)
            rr = subprocess.run([PY, os.path.join(SK, "render_check.py"), svg, png],
                                capture_output=True, text=True, cwd=SK)
            if rr.returncode == 0 and os.path.exists(png):
                ref_im = Image.open(src).convert("RGB")
                W, H = ref_im.size
                ref = np.asarray(ref_im).astype(np.float32)
                v = Image.open(png).convert("RGB")
                if v.size != (W, H):
                    v = v.resize((W, H), Image.LANCZOS)
                ren = np.asarray(v).astype(np.float32)
                err = np.abs(ref - ren)
                rec["mae"] = round(float(err.mean()), 3)
                rec["p99"] = round(float(np.percentile(err.max(axis=2), 99)), 1)
                L = ref.mean(axis=2)
                dark = L < 80
                rec["dark_bias"] = (round(float((ren - ref)[..., 0][dark].mean()), 2)
                                    if dark.sum() > 50 else None)
            else:
                rec["mae"] = None
                rec["error"] = "render failed"

        with io.open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for p in (svg, png):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        if rec["mae"] is not None:
            b = base[key]["mae"]
            d = 100 * (rec["mae"] - b) / b if b else 0
            el = time.time() - t_start
            eta = (el / (i + 1)) * (len(sample) - i - 1) / 60
            print("  [%3d/%d] %-30s %6.2fMP  %6.3f -> %6.3f (%+6.2f%%)  ETA %.0fmin"
                  % (i + 1, len(sample), s["file"][:30], s["mp"],
                     b, rec["mae"], d, eta))
        else:
            print("  [%3d/%d] %-30s 失败 %s"
                  % (i + 1, len(sample), s["file"][:30], rec.get("error", "")[:40]))

    # ---- 对比报告 ----
    var = load_jsonl(jsonl)
    rows = []
    for k, v in var.items():
        if v.get("mae") is None or k not in base or base[k].get("mae") is None:
            continue
        b = base[k]
        rows.append({
            "key": k, "group": v["group"], "mp": v["mp"],
            "base": b["mae"], "var": v["mae"],
            "delta": 100 * (v["mae"] - b["mae"]) / b["mae"],
            "base_p99": b.get("p99"), "var_p99": v.get("p99"),
        })
    if not rows:
        print("\n无可对比结果"); return 1

    ds = np.array([r["delta"] for r in rows])
    better = int((ds < -0.5).sum())
    worse = int((ds > 0.5).sum())
    same = len(ds) - better - worse

    print("\n=== %s vs 基线（逐图对比，n=%d）===" % (args.tag, len(rows)))
    print("  改善(<-0.5%%) %d 张   持平 %d 张   退化(>0.5%%) %d 张"
          % (better, same, worse))
    print("  MAE 变化 中位 %+.2f%%  均值 %+.2f%%  最好 %+.2f%%  最差 %+.2f%%"
          % (np.median(ds), ds.mean(), ds.min(), ds.max()))
    bm = np.array([r["base"] for r in rows])
    vm = np.array([r["var"] for r in rows])
    print("  基线 MAE 中位 %.3f -> 变体 %.3f" % (np.median(bm), np.median(vm)))

    # 分组
    lines = ["# %s vs 基线" % args.tag, "",
             "环境变量：`%s`" % extra, "",
             "| 指标 | 值 |", "|---|---|",
             "| 样本 | %d |" % len(rows),
             "| 改善 / 持平 / 退化 | %d / %d / %d |" % (better, same, worse),
             "| MAE 变化中位 | %+.2f%% |" % np.median(ds),
             "| 基线 -> 变体（中位） | %.3f -> %.3f |" % (np.median(bm), np.median(vm)),
             ""]
    lines += ["## 分组", "", "| 组 | n | 中位改善 | 退化数 |", "|---|---|---|---|"]
    for g in sorted(set(r["group"] for r in rows)):
        sub = [r for r in rows if r["group"] == g]
        d2 = np.array([r["delta"] for r in sub])
        lines.append("| %s | %d | %+.2f%% | %d |"
                     % (g, len(sub), np.median(d2), int((d2 > 0.5).sum())))

    # 退化的图
    worse_rows = sorted([r for r in rows if r["delta"] > 0.5], key=lambda x: -x["delta"])
    if worse_rows:
        lines += ["", "## 退化的图（%d 张）" % len(worse_rows), "",
                  "| 文件 | MP | 基线 | 变体 | 变化 |", "|---|---|---|---|---|"]
        for r in worse_rows[:20]:
            lines.append("| %s | %.2f | %.3f | %.3f | %+.2f%% |"
                         % (r["file"] if "file" in r else r["key"].split("/")[-1],
                            r["mp"], r["base"], r["var"], r["delta"]))

    p = os.path.join(OUT, "%s_vs_baseline.md" % args.tag)
    open(p, "w", encoding="utf-8").write("\n".join(lines))
    print("  报告 -> %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
