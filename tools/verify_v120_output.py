"""verify_v120_output.py -- v1.2.0 产出完整性与质量核验（f-string 版）

核验项：
  1. 文件数量与完整性
  2. summary 与 svg 数量一致
  3. p99 / gt60 分布
  4. 与 v1.1.1 同图对比
  5. 抽样重算 MAE，确认与 summary 一致
  6. viewBox 检查
"""
import os
import io
import re
import json
import numpy as np
from PIL import Image
from collections import Counter

def _find_root():

    """工作区根：优先环境变量，否则取仓库的上一级。



    原实现写死作者机器上的 F:\\SVG drawing。

    """

    env = os.environ.get("IMGSVG_ROOT")

    if env and os.path.isdir(env):

        return env

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))





ROOT = _find_root()
OUT = os.path.join(ROOT, "output", "final_v1.2.0", "梅原生")
OLD = os.path.join(ROOT, "output", "final_v1.1.1", "梅原生")
BASE = os.path.join(ROOT, "input", "new_train", "梅原生")


def main():
    print("=== 一、文件完整性 ===")
    for sub in ("svg", "png", "cmp"):
        d = os.path.join(OUT, sub)
        if not os.path.isdir(d):
            print(f"  {sub:4s} 目录缺失")
            continue
        files = os.listdir(d)
        empt = [f for f in files
                if os.path.getsize(os.path.join(d, f)) == 0]
        print(f"  {sub:4s} {len(files):3d} 个文件  空文件 {len(empt)}")
        if empt:
            print(f"      空文件: {empt[:5]}")

    sp = os.path.join(OUT, "_summary.json")
    if not os.path.exists(sp):
        print("  _summary.json 缺失")
        return
    summ = json.load(open(sp, encoding="utf-8"))
    rows = summ["rows"]

    print()
    print("=== 二、_summary.json ===")
    print(f"  count={summ['count']}  mode={summ.get('mode')}")
    print(f"  MAE 中位 {summ['mae_median']:.3f}  均值 {summ['mae_mean']:.3f}  "
          f"范围 {summ['mae_min']:.3f}~{summ['mae_max']:.3f}")
    print(f"  p99 中位 {summ['p99_median']:.0f}   "
          f"gt60 中位 {summ['gt60_median']:.2f}   "
          f"总体积 {summ['total_mb'] / 1024:.1f} GB")

    n_svg = len([f for f in os.listdir(os.path.join(OUT, "svg"))
                 if f.endswith(".svg")])
    same = "一致" if len(rows) == n_svg else "不一致"
    print(f"  summary {len(rows)} 条 vs svg {n_svg} 个: {same}")

    print()
    print("=== 三、p99 / gt60 分布 ===")
    p99 = np.array([r["p99"] for r in rows])
    g60 = np.array([r["gt60"] for r in rows])
    print(f"  p99   中位 {np.median(p99):.0f}  均值 {p99.mean():.1f}  "
          f"范围 {p99.min():.0f}~{p99.max():.0f}")
    print(f"  gt60  中位 {np.median(g60):.2f}  均值 {g60.mean():.2f}  "
          f"范围 {g60.min():.2f}~{g60.max():.2f}")
    n1 = int((p99 <= 60).sum())
    n2 = int((g60 <= 1).sum())
    print(f"  p99<=60 的张数: {n1}/{len(p99)} ({100 * n1 / len(p99):.0f}%)")
    print(f"  gt60<=1 的张数: {n2}/{len(g60)} ({100 * n2 / len(g60):.0f}%)")

    print()
    print("=== 四、与 v1.1.1 同图对比 ===")
    op = os.path.join(OLD, "_summary.json")
    if os.path.exists(op):
        old = json.load(open(op, encoding="utf-8"))
        oldmap = {r["file"]: r["mae"] for r in old["rows"]
                  if isinstance(r.get("mae"), (int, float))}
        pairs = [(r["file"], oldmap[r["file"]], r["mae"])
                 for r in rows if r["file"] in oldmap]
        if pairs:
            print(f"  可比 {len(pairs)} 张")
            d = np.array([100 * (n - o) / o for _, o, n in pairs])
            print(f"  改善 {int((d < -0.5).sum())} / 变差 "
                  f"{int((d > 0.5).sum())}  中位 {np.median(d):+.2f}  "
                  f"均值 {d.mean():+.2f}")
            print(f"  最好 {d.min():+.2f}  最差 {d.max():+.2f}")
            srt = sorted(pairs, key=lambda x: (x[2] - x[1]) / x[1])
            print(f"  {'文件':32s} {'v1.1.1':>8s} {'v1.2.0':>8s} {'变化':>8s}")
            for f, o, n in srt[:6]:
                print(f"  {f[:32]:32s} {o:8.3f} {n:8.3f} "
                      f"{100 * (n - o) / o:+7.1f}")
            print("  ...")
            for f, o, n in srt[-3:]:
                print(f"  {f[:32]:32s} {o:8.3f} {n:8.3f} "
                      f"{100 * (n - o) / o:+7.1f}")
        else:
            print("  无重叠文件")
    else:
        print("  v1.1.1 summary 缺失，跳过")

    print()
    print("=== 五、抽样重算 MAE ===")
    step = max(1, len(rows) // 5)
    picks = rows[::step][:5]
    ok = 0
    for r in picks:
        src = os.path.join(BASE, r["file"])
        png = os.path.join(OUT, "png",
                           os.path.splitext(r["file"])[0] + "_render.png")
        if not (os.path.exists(src) and os.path.exists(png)):
            print(f"  {r['file'][:30]:30s} 文件缺失")
            continue
        im = Image.open(src).convert("RGB")
        W0, H0 = im.size
        ref = np.asarray(im).astype(np.float32)
        v = Image.open(png).convert("RGB")
        if v.size != (W0, H0):
            v = v.resize((W0, H0), Image.LANCZOS)
        mae = float(np.abs(ref - np.asarray(v).astype(np.float32)).mean())
        diff = mae - r["mae"]
        good = abs(diff) < 0.05
        ok += 1 if good else 0
        tag = "OK" if good else "差异大"
        print(f"  {r['file'][:30]:30s} 记录 {r['mae']:.3f}  重算 {mae:.3f}  "
              f"差 {diff:+.4f}  {tag}")
    print(f"  抽样 {len(picks)} 张，{ok} 张一致")

    print()
    print("=== 六、viewBox 检查 ===")
    checked = 0
    mismatch = 0
    for r in rows[:8]:
        svg = os.path.join(OUT, "svg",
                           os.path.splitext(r["file"])[0] + ".svg")
        if not os.path.exists(svg):
            continue
        head = io.open(svg, encoding="utf-8").read(300)
        m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', head)
        if not m:
            continue
        vw, vh = int(float(m.group(1))), int(float(m.group(2)))
        ew, eh = r["w"] * r["k"], r["h"] * r["k"]
        checked += 1
        if (vw, vh) != (ew, eh):
            mismatch += 1
            print(f"  {r['file'][:30]:30s} viewBox {vw}x{vh}  "
                  f"期望 {ew}x{eh}")
    tail = "（全部正确）" if mismatch == 0 else ""
    print(f"  抽查 {checked} 个，不符 {mismatch} 个{tail}")

    print()
    print("=== 七、k 档位与实际工作尺寸 ===")
    ks = Counter(r["k"] for r in rows)
    print(f"  k 分布: {dict(sorted(ks.items()))}")
    ws = np.array([max(r["w"] * r["k"], r["h"] * r["k"]) for r in rows])
    print(f"  工作长边: 中位 {np.median(ws):.0f}  "
          f"最小 {ws.min()}  最大 {ws.max()}")
    below = int((ws < 4000).sum())
    print(f"  长边 < 4000 的: {below} 张（本身已 >=4000 或未达目标）")


if __name__ == "__main__":
    main()
