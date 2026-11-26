"""refit_adapt_n.py -- 用代码内口径重新拟合 N 公式

## 为什么必须重做

先前用 `dataset_profile/profile.json` 里的 `flat_pct` 拟合公式，
但那份画像是在 **缩放到 800px** 的图上算的梯度；
而 `vec3.py` 运行时在 **工作尺寸（WORK_W×WORK_H）** 上算。

同一张图两个口径差很多：

    96e7ee9b...JPG  profile flat = 64.6%   代码内 flat = 73.0%

公式在错误口径上拟合，上线后必然偏移。

## 做法

用与 `vec3.py` 完全相同的算法（同一缩放策略、同样的 grad<3 判据）
重新计算每张图的 flat_pct / detail_pct，再重做网格搜索。

## 输出

  workspace/nsweep/refit.json   新常数 + 逐图验证
"""
import os
import io
import json
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
RESULT = os.path.join(ROOT, "workspace", "nsweep", "results.jsonl")
OUT = os.path.join(ROOT, "workspace", "nsweep", "refit.json")

# 与 vec3.py 一致的缩放上限（VEC_W 默认不缩放；这里复原实际工作尺寸）
def work_size(W0, H0, vec_w=None, autow=False, maxw=1920):
    """复刻 vec3.py 的 WORK_W 逻辑"""
    if vec_w:
        ww = int(vec_w)
    else:
        ww = W0
    ww = min(ww, W0 * 2)
    if autow and ww > maxw:
        ww = maxw
    wh = round(ww * H0 / W0)
    return ww, wh


def features(path):
    """按 vec3.py 的口径算 flat/detail：在工作尺寸上算梯度"""
    im = Image.open(path).convert("RGB")
    W0, H0 = im.size
    ww, wh = work_size(W0, H0)          # 默认：不缩放
    if (ww, wh) != (W0, H0):
        im = im.resize((ww, wh), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32)
    lum = a.mean(axis=2)
    gx = np.zeros_like(lum); gy = np.zeros_like(lum)
    gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
    gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
    grad = np.hypot(gx, gy)
    return {
        "mp": (ww * wh) / 1e6,
        "flat": 100.0 * float((grad < 3).mean()),
        "det": 100.0 * float((grad >= 15).mean()),
    }


def main():
    rows = []
    for line in io.open(RESULT, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if r.get("mae") is not None:
                rows.append(r)
        except Exception:
            pass
    bykey = {}
    for r in rows:
        bykey.setdefault(r["key"], {})[r["n"]] = r

    full = [k for k, d in bykey.items() if len(d) == 4]
    print("=== 用代码内口径重算特征 ===")
    print("已完成四档 N 的图：%d 张" % len(full))

    data = []
    for k in full:
        g, f = k.split("/", 1)
        path = os.path.join(ROOT, "input", "new_train", g, f)
        if not os.path.exists(path):
            print("  缺失 %s" % k[:50]); continue
        feat = features(path)
        d = bykey[k]
        b = d[56]["mae"]
        bn = min(d, key=lambda n: d[n]["mae"])
        old = d[56]
        data.append({
            "key": k, "file": f, "group": g,
            "mp": feat["mp"], "flat": feat["flat"], "det": feat["det"],
            "mp_profile": old["mp"], "flat_profile": old["flat_pct"],
            "best": bn, "mae56": b,
            "maes": {n: d[n]["mae"] for n in d},
        })

    print("\n口径差异（profile vs 代码内）：")
    print("%-34s %8s %8s %8s %8s" % ("文件", "flat画像", "flat代码", "差", "MP"))
    for r in data:
        print("%-34s %7.1f%% %7.1f%% %+7.1f %8.2f"
              % (r["file"][:34], r["flat_profile"], r["flat"], 
                 r["flat"] - r["flat_profile"], r["mp"]))

    # ---- 网格搜索（代码内口径）----
    NS = (56, 96, 128, 192)
    best = None
    best_cost = 1e9
    for A in range(150, 320, 5):
        for B in np.arange(0, 5.0, 0.1):
            for C in np.arange(-20, 40, 2.5):
                cost = 0.0
                for r in data:
                    n_pred = A - B * r["flat"] + C * np.log2(max(r["mp"], 0.1))
                    n_pred = int(np.clip(round(n_pred), 56, 192))
                    n_use = min(NS, key=lambda x: abs(x - n_pred))
                    cost += (r["maes"][n_use] - r["maes"][r["best"]]) / r["mae56"]
                cost /= len(data)
                if cost < best_cost:
                    best_cost = cost
                    best = (A, B, C)
    A, B, C = best
    print("\n=== 拟合结果（代码内口径）===")
    print("  N = %.0f - %.2f*flat_pct + %.1f*log2(MP)   clamp[56,192]"
          % (A, B, C))
    print("  与理论最优的平均差距 %.3f%%" % (best_cost * 100))

    # 验证
    print("\n%-34s %6s %7s %6s %6s %8s %8s" 
          % ("文件", "MP", "flat%", "最优N", "公式N", "公式MAE", "vs N56"))
    sum_p = sum_o = sum_5 = 0.0
    for r in data:
        n_pred = A - B * r["flat"] + C * np.log2(max(r["mp"], 0.1))
        n_pred = int(np.clip(round(n_pred), 56, 192))
        n_use = min(NS, key=lambda x: abs(x - n_pred))
        mp_ = r["maes"][n_use]
        sum_p += mp_; sum_5 += r["mae56"]; sum_o += r["maes"][r["best"]]
        print("%-34s %6.2f %6.1f%% %6d %6d %8.3f %+7.2f%%"
              % (r["file"][:34], r["mp"], r["flat"], r["best"], n_pred,
                 mp_, 100 * (mp_ - r["mae56"]) / r["mae56"]))
    n = len(data)
    print("\n  公式平均 MAE %.3f" % (sum_p / n))
    print("  N=56 平均 MAE %.3f" % (sum_5 / n))
    print("  最优   平均 MAE %.3f" % (sum_o / n))
    print("  公式 vs N=56: %+.2f%%" % (100 * (sum_p - sum_5) / sum_5))
    print("  公式 vs 最优: %+.2f%%" % (100 * (sum_p - sum_o) / sum_o))

    json.dump({"A": A, "B": B, "C": C, "cost": best_cost,
               "n": n, "data": data},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n  -> %s" % OUT)


if __name__ == "__main__":
    main()
