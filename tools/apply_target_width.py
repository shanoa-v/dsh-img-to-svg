"""apply_target_width.py -- 按目标工作宽度决定物理放大倍数

## 问题

物理放大测试显示「越大越好」，但边际递减：
  1.02MP: 1x 8.832 -> 2x 7.309(-17%) -> 3x 6.713(-8%) -> 4x 6.270(-7%)
  2.07MP: 1x 8.209 -> 2x 7.050(-14%) -> 3x 6.216(-12%)

若对大图也强制 4x，58.82MP 会变成 940MP，不可行。

## 策略

统一到**目标工作宽度** T（例如 4000px）：
  k = clamp(round(T / max(W0,H0)), 1, 8)

这样：
  - 小图（852px）放大到接近 T，收益最大
  - 大图（5200px）不放大（k=1），避免爆炸
  - 中等图适度放大

## 参数选择依据

由测试反推：要达到 MAE ≈ 6.2~6.3（参考物的 6.408 水平），
工作宽度需达到约 3400~5800px。
取 **T=4000** 作为目标（在收益与体积/耗时之间平衡）。

## 峰值内存风险

58MP 图（5200x11312）若 k=1，管线本身已跑通（实测 415s / 59MB SVG）。
小图 4x 后也仅相当于 10~17MP，安全。
"""
import io
import os


def compute_k(w0, h0, target_w=4000, k_max=8):
    """按目标宽度算放大倍数（取整数，至少 1）"""
    long_side = max(w0, h0)
    if long_side >= target_w:
        return 1
    k = target_w / long_side
    k = max(1, min(k_max, int(round(k))))
    return k


if __name__ == "__main__":
    import json
    import sys
    def _find_root():
        """工作区根：优先环境变量，否则取仓库的上一级。
    
        原实现写死作者机器上的 F:\\SVG drawing。
        """
        env = os.environ.get("IMGSVG_ROOT")
        if env and os.path.isdir(env):
            return env
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    
    ROOT = _find_root()
    prof = json.load(open(os.path.join(ROOT, "output", "dataset_profile",
                                       "profile.json"), encoding="utf-8"))
    recs = prof.get("梅原生", [])
    print("=== 目标宽度 4000px 下的放大倍数分布 ===")
    from collections import Counter
    ks = []
    for r in recs:
        k = compute_k(r["w"], r["h"], 4000)
        ks.append(k)
        r["k"] = k
        r["out_px"] = round(r["w"] * k * r["h"] * k / 1e6, 1)
    print("  k 分布: %s" % dict(sorted(Counter(ks).items())))
    print("  放大后像素总计 %.1f MP（原 %.1f MP）"
          % (sum(r["out_px"] for r in recs), sum(r["mp"] for r in recs)))

    print()
    print("  %-40s %8s %8s %6s %8s"
          % ("文件", "原MP", "长边", "k", "放大后MP"))
    for r in sorted(recs, key=lambda x: x["mp"]):
        print("  %-40s %8.2f %8d %6d %8.1f"
              % (r["file"][:40], r["mp"], max(r["w"], r["h"]), r["k"],
                 r["out_px"]))

    # 估算耗时：按 22.7 * MP 秒的经验拟合（本项目实测 t = 7.9 + 22.7*MP）
    total_s = sum(7.9 + 22.7 * r["out_px"] for r in recs)
    print()
    print("  估算总耗时 %.1f 小时（按 t = 7.9 + 22.7*MP）" % (total_s / 3600))
    json.dump(recs, open(os.path.join(ROOT, "workspace", "meiyuan_diag",
                                      "k_plan_4000.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
