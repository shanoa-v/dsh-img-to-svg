"""validate_adapt_n.py -- 留一交叉验证：判断 N 公式是否过拟合

## 为什么必须做

公式 `N = 285 - 2.70*flat + 22.5*log2(MP)` 是在 18 张图上网格搜索出来的，
同一批数据上"达到理论上限 99.5%"可能是过拟合。

留一法（Leave-One-Out）：每次扣掉一张，用剩下 17 张重新拟合常数，
再用新常数预测被扣掉那张。若 LOO 结果仍接近原公式，说明规律稳健。

## 另外三个必查项

1. 公式 N 的分布是否集中在边界（若大量落在 56 或 192 的 clamp 上，
   说明公式实际只在做"全给"或"全不给"，没有真正的分辨力）
2. 逐图最差情况（不能只看平均）
3. 长尾 p99 的变化（历史教训：均值改善常伴随长尾恶化）
"""
import os
import io
import json
import numpy as np

def _find_root():

    """工作区根：优先环境变量，否则取仓库的上一级。



    原实现写死作者机器上的 F:\\SVG drawing。

    """

    env = os.environ.get("IMGSVG_ROOT")

    if env and os.path.isdir(env):

        return env

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))





ROOT = _find_root()
REFIT = os.path.join(ROOT, "workspace", "nsweep", "refit.json")
NS = (56, 96, 128, 192)


def cost_of(params, data):
    A, B, C = params
    tot = 0.0
    for r in data:
        n_pred = A - B * r["flat"] + C * np.log2(max(r["mp"], 0.1))
        n_pred = int(np.clip(round(n_pred), 56, 192))
        n_use = min(NS, key=lambda x: abs(x - n_pred))
        tot += (r["maes"][n_use] - r["maes"][r["best"]]) / r["mae56"]
    return tot / len(data)


def grid_fit(data):
    best = None
    bc = 1e9
    for A in range(150, 320, 5):
        for B in np.arange(0, 5.0, 0.1):
            for C in np.arange(-20, 40, 2.5):
                c = cost_of((A, B, C), data)
                if c < bc:
                    bc, best = c, (A, B, C)
    return best


def main():
    d = json.load(open(REFIT, encoding="utf-8"))
    data = d["data"]
    # JSON 会把 int 键变成 str，统一转回来
    for r in data:
        r["maes"] = {int(k): v for k, v in r["maes"].items()}
    A, B, C = d["A"], d["B"], d["C"]
    print("=== 留一交叉验证（n=%d）===" % len(data))
    print("全量拟合常数: A=%.0f B=%.2f C=%.1f" % (A, B, C))
    print()
    print("%-34s %6s %7s %8s %8s %8s" 
          % ("留出图", "MP", "flat%", "全量N", "LOO N", "LOO代价"))

    loo_cost = []
    for i, r in enumerate(data):
        rest = [x for x in data if x is not r]
        p = grid_fit(rest)
        n_loo = p[0] - p[1] * r["flat"] + p[2] * np.log2(max(r["mp"], 0.1))
        n_loo = int(np.clip(round(n_loo), 56, 192))
        n_use = min(NS, key=lambda x: abs(x - n_loo))
        c = (r["maes"][n_use] - r["maes"][r["best"]]) / r["mae56"]
        loo_cost.append(c)
        n_full = A - B * r["flat"] + C * np.log2(max(r["mp"], 0.1))
        n_full = int(np.clip(round(n_full), 56, 192))
        print("%-34s %6.2f %6.1f%% %8d %8d %+7.2f%%"
              % (r["file"][:34], r["mp"], r["flat"], n_full, n_loo, 100 * c))

    loo_cost = np.array(loo_cost)
    print()
    print("LOO 平均代价 %.3f%%  中位 %.3f%%  最差 %.3f%%"
          % (100 * loo_cost.mean(), 100 * np.median(loo_cost),
             100 * loo_cost.max()))
    print("（代价 = 公式选的 N 与逐图最优 N 的 MAE 差距，越小越好）")

    # ---- 检查 1：N 分布是否过度集中在 clamp 边界 ----
    print("\n=== 检查 1：公式 N 的分布 ===")
    preds = []
    for r in data:
        n_pred = A - B * r["flat"] + C * np.log2(max(r["mp"], 0.1))
        raw = n_pred
        n_pred = int(np.clip(round(n_pred), 56, 192))
        preds.append((n_pred, raw))
    vals = [p[0] for p in preds]
    at56 = sum(1 for v in vals if v <= 56)
    at192 = sum(1 for v in vals if v >= 192)
    print("  N 范围 %d~%d  中位 %.0f" % (min(vals), max(vals), np.median(vals)))
    print("  落在下界 56 的: %d 张 (%.0f%%)" % (at56, 100 * at56 / len(vals)))
    print("  落在上界 192 的: %d 张 (%.0f%%)" % (at192, 100 * at192 / len(vals)))
    mid = len(vals) - at56 - at192
    print("  落在中间区间的: %d 张 (%.0f%%)" % (mid, 100 * mid / len(vals)))
    if mid < len(vals) * 0.3:
        print("  警告：超过 70%% 落在边界，公式分辨力不足")
    else:
        print("  通过：中间区间有足够样本，说明公式有真实分辨力")

    # ---- 检查 2：逐图最差 ----
    print("\n=== 检查 2：逐图表现（含最差）===")
    deltas = []
    for r in data:
        n_pred = A - B * r["flat"] + C * np.log2(max(r["mp"], 0.1))
        n_pred = int(np.clip(round(n_pred), 56, 192))
        n_use = min(NS, key=lambda x: abs(x - n_pred))
        dl = 100 * (r["maes"][n_use] - r["mae56"]) / r["mae56"]
        deltas.append((dl, r["file"], r["best"], n_pred))
    deltas.sort(reverse=True)
    print("  相比 N=56 最差 5 张：")
    for dl, f, bn, np_ in deltas[:5]:
        print("    %-34s %+7.2f%%  (最优N=%d 公式N=%d)" % (f[:34], dl, bn, np_))
    ds = np.array([x[0] for x in deltas])
    print("  整体：改善 %d / 变差 %d  中位 %+.2f%%  最差 %+.2f%%"
          % ((ds < -0.2).sum(), (ds > 0.2).sum(), np.median(ds), ds.max()))
    # 比较：固定 N=96
    d96 = np.array([100 * (r["maes"][96] - r["mae56"]) / r["mae56"] for r in data])
    print("  对照固定 N=96：改善 %d / 变差 %d  中位 %+.2f%%  最差 %+.2f%%"
          % ((d96 < -0.2).sum(), (d96 > 0.2).sum(), np.median(d96), d96.max()))

    # ---- 检查 3：p99 长尾 ----
    print("\n=== 检查 3：长尾 p99（历史教训：均值改善常伴长尾恶化）===")
    # refit.json 未存 p99，从原始 results 读
    raw = {}
    for line in io.open(os.path.join(ROOT, "workspace", "nsweep", "results.jsonl"),
                        encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if r.get("mae") is not None:
                raw.setdefault(r["key"], {})[r["n"]] = r
        except Exception:
            pass
    diffs = []
    for r in data:
        k = r["key"]
        if k not in raw:
            continue
        rr = raw[k]
        n_pred = A - B * r["flat"] + C * np.log2(max(r["mp"], 0.1))
        n_pred = int(np.clip(round(n_pred), 56, 192))
        n_use = min(NS, key=lambda x: abs(x - n_pred))
        if 56 in rr and n_use in rr:
            diffs.append(rr[n_use]["p99"] - rr[56]["p99"])
    if diffs:
        diffs = np.array(diffs)
        print("  p99 变化：中位 %+.1f  变大的 %d 张  变小的 %d 张"
              % (np.median(diffs), (diffs > 0).sum(), (diffs < 0).sum()))
        print("  最差 p99 恶化 +%.1f" % diffs.max())
        print("  说明：提高 N 引入更多碎块，p99 略升是预期代价；"
              "若中位变化 >+3 则需警惕")


if __name__ == "__main__":
    main()
