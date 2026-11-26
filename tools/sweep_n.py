"""sweep_n.py -- VEC_N 扫描与自适应公式拟合

## 背景

v1.1.0 修好接缝渗色后，历史结论「提高全局 VEC_N 会伤害部分图型」需要重新检验。

初测（4 张图 × 5 档 N）：

| 图 | 细节% | MP | N=24 | N=56 | N=96 | N=128 | N=192 | 最优 |
|---|---|---|---|---|---|---|---|---|
| noyu1 | 54.3 | 0.51 | 14.832 | 13.763 | 13.235 | 13.224 | **13.070** | 192 |
| noyu2 | 44.4 | 0.52 | 10.954 | 10.357 | 10.132 | 10.043 | **9.980** | 192 |
| noyu3 | 29.9 | 0.56 | 7.658 | 7.323 | 7.194 | **7.144** | 7.196 | 128 |
| pixiv1 | 31.8 | 1.40 | 10.541 | 9.558 | **9.298** | 9.366 | 9.581 | 96 |

**N=96 在全部 4 张上都优于 N=56**（+2.2% ~ +3.8%）。
N=192 在 pixiv1 上开始退化 —— 说明存在拐点，不能一味加大。

## 目标

在足够样本上扫 N，拟合「最优 N 与图像特征的关系」，写出自适应公式。

候选特征：
  - 细节区占比 detail_pct（源 grad>=15 的比例）
  - 像素数 px
  - 8级量化唯一色数 colors8
  - 平坦区占比

## 用法

  python sweep_n.py --n 20 --ns 56,96,128       抽样 20 张扫 3 档
  python sweep_n.py --n 50 --ns 56,96           更大样本两档对比
"""
import os
import io
import json
import time
import argparse
import subprocess
import numpy as np
from PIL import Image
import sys

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
def _find_root():
    """工作区根：优先环境变量，否则取仓库的上一级。

    原实现写死作者机器上的 F:\\SVG drawing。
    """
    env = os.environ.get("IMGSVG_ROOT")
    if env and os.path.isdir(env):
        return env
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROOT = _find_root()
OUT = os.path.join(ROOT, "workspace", "nsweep")
os.makedirs(OUT, exist_ok=True)
SAMPLE = os.path.join(ROOT, "output", "dataset_profile", "sample.json")
RESULT = os.path.join(OUT, "results.jsonl")


def load_done():
    d = {}
    if os.path.exists(RESULT):
        for line in io.open(RESULT, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                d[(r["key"], r["n"])] = r
            except Exception:
                pass
    return d


def measure(src, svg, png):
    ref_im = Image.open(src).convert("RGB")
    W, H = ref_im.size
    ref = np.asarray(ref_im).astype(np.float32)
    v = Image.open(png).convert("RGB")
    if v.size != (W, H):
        v = v.resize((W, H), Image.LANCZOS)
    ren = np.asarray(v).astype(np.float32)
    err = np.abs(ref - ren)
    return {
        "mae": round(float(err.mean()), 3),
        "p99": round(float(np.percentile(err.max(axis=2), 99)), 1),
        "svg_mb": round(os.path.getsize(svg) / 1048576, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--ns", default="56,96,128")
    ap.add_argument("--env", action="append", default=[])
    args = ap.parse_args()

    Ns = [int(x) for x in args.ns.split(",")]
    extra = {}
    for e in args.env:
        if "=" in e:
            k, v = e.split("=", 1)
            extra[k] = v

    sample = json.load(open(SAMPLE, encoding="utf-8"))
    # 分层取：每档尺寸/复杂度各取一些
    picked = []
    seen = set()
    for want in ("small", "mid", "large", "huge"):
        for s in sample:
            if s["size_bin"] == want and s["file"] not in seen:
                picked.append(s)
                seen.add(s["file"])
                if sum(1 for p in picked if p["size_bin"] == want) >= max(3, args.n // 4):
                    break
    picked = picked[:args.n]

    done = load_done()
    todo = [(s, n) for s in picked for n in Ns
            if (s["group"] + "/" + s["file"], n) not in done]
    print("=== VEC_N 扫描 ===")
    print("  图 %d 张 × N档 %s = %d 次，已完成 %d，待跑 %d"
          % (len(picked), Ns, len(picked) * len(Ns), len(done), len(todo)))

    t0 = time.time()
    for i, (s, n) in enumerate(todo):
        key = s["group"] + "/" + s["file"]
        src = os.path.join(ROOT, s["path"])
        if not os.path.exists(src):
            continue
        stem = os.path.splitext(s["file"])[0].replace(" ", "_")[:60]
        svg = os.path.join(OUT, "%s_n%d.svg" % (stem, n))
        png = svg.replace(".svg", ".png")

        env = {k: v for k, v in os.environ.items() if not k.startswith("VEC_")}
        env["VEC_TITLE"] = stem
        env["VEC_N"] = str(n)
        env.update(extra)

        r = subprocess.run([PY, os.path.join(SK, "vec.py"), src, svg],
                           capture_output=True, text=True, env=env, cwd=SK)
        rec = {"key": key, "group": s["group"], "file": s["file"], "n": n,
               "mp": s["mp"], "size_bin": s["size_bin"],
               "colors8": s["colors8"], "flat_pct": s["flat_pct"],
               "detail_pct": s["detail_pct"], "ok": r.returncode == 0}
        if r.returncode != 0:
            rec["mae"] = None
        else:
            rr = subprocess.run([PY, os.path.join(SK, "render_check.py"), svg, png],
                                capture_output=True, text=True, cwd=SK)
            if rr.returncode == 0 and os.path.exists(png):
                rec.update(measure(src, svg, png))
            else:
                rec["mae"] = None

        with io.open(RESULT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for p in (svg, png):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        if rec.get("mae") is not None:
            el = time.time() - t0
            eta = (el / (i + 1)) * (len(todo) - i - 1) / 60
            print("  [%3d/%d] %-26s N=%-4d %5.2fMP MAE %6.3f  体积%5.1fMB  ETA %.0fmin"
                  % (i + 1, len(todo), s["file"][:26], n, s["mp"],
                     rec["mae"], rec.get("svg_mb", 0), eta))

    # ---- 分析 ----
    rows = []
    for line in io.open(RESULT, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
                if r.get("mae") is not None:
                    rows.append(r)
            except Exception:
                pass
    if not rows:
        return 0

    bykey = {}
    for r in rows:
        bykey.setdefault(r["key"], {})[r["n"]] = r
    print("\n=== 每图最优 N ===")
    print("%-30s %6s %7s %6s %8s %8s" % ("文件", "MP", "细节%", "色数", "最优N", "vs N56"))
    best_rows = []
    for k, d in sorted(bykey.items()):
        if 56 not in d:
            continue
        base = d[56]["mae"]
        bn = min(d, key=lambda n: d[n]["mae"])
        delta = 100 * (d[bn]["mae"] - base) / base
        r0 = d[56]
        best_rows.append({"key": k, "mp": r0["mp"], "det": r0["detail_pct"],
                          "flat": r0["flat_pct"], "col": r0["colors8"],
                          "best": bn, "delta": delta})
        print("%-30s %6.2f %6.1f%% %6d %8d %+7.2f%%"
              % (r0["file"][:30], r0["mp"], r0["detail_pct"], r0["colors8"],
                 bn, delta))

    if best_rows:
        bs = np.array([b["best"] for b in best_rows])
        ds = np.array([b["delta"] for b in best_rows])
        print("\n  最优 N 分布：中位 %.0f  范围 %d~%d" % (np.median(bs), bs.min(), bs.max()))
        print("  相比 N=56 的中位改善 %+.2f%%" % np.median(ds))
        # 简单相关性
        for f in ("mp", "det", "flat", "col"):
            x = np.array([b[f] for b in best_rows], dtype=float)
            if x.std() > 0:
                c = np.corrcoef(x, bs)[0, 1]
                print("  相关系数(最优N vs %-5s) = %+.3f" % (f, c))

    json.dump(best_rows, open(os.path.join(OUT, "best_n.json"), "w",
                              encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n  -> %s" % os.path.join(OUT, "best_n.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
